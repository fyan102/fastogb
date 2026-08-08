"""A configurable estimator for additive rule ensembles."""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from inspect import signature
from time import perf_counter

import numpy as np

from fastogb.encoding import PropositionEncoder, as_2d_array, infer_feature_names
from fastogb.kernels import rule_ensemble_scores
from fastogb.linalg import OrthogonalBasis
from fastogb.logic import Conjunction
from fastogb.losses import loss_function
from fastogb.objectives import GradientBoostingObjectiveXGB
from fastogb.parallel import numba_threads, resolve_numba_jobs
from fastogb.rules import AdditiveRuleEnsemble, Rule
from fastogb.terms import LinearTerm, make_linear_terms
from fastogb.weights import KeepWeight, WeightUpdateMethod, initial_constant


_ENCODER_KEYS = {'categorical', 'include_missing', 'max_categories', 'min_category_count', 'without'}
_PREDICTION_PARALLEL_MIN_WORK = 100_000


class GeneralRuleBoostingEstimator:
    """Fit a configurable additive ensemble of rules and optional linear terms.

    Parameters
    ----------
    num_rules : int, default=3
        Number of learned conjunction rules, excluding the optional default rule and linear terms.
    objective_function : objective class, default=GradientBoostingObjectiveXGB
        Query-selection objective. Supported classes are ``GradientBoostingObjectiveXGB``,
        ``GradientBoostingObjectiveMWG``, ``GradientBoostingObjectiveGPE`` and
        ``OrthogonalBoostingObjective``.
    weight_update_method : WeightUpdateMethod or None, default=None
        Rule-weight updater. Use ``KeepWeight()``, ``LineSearch()`` or ``FullyCorrective()``. ``None`` is
        equivalent to ``KeepWeight()``. The estimator supplies its loss and regularisation to the updater.
    loss : {'squared', 'logistic', 'poisson'} or loss object, default='squared'
        Training loss. Logistic targets must be encoded as -1 and 1; Poisson targets must be non-negative.
    reg : float, default=1.0
        Non-negative L2 regularisation strength used by weight updates and objectives where applicable.
    search : {'exhaustive', 'greedy', 'ogb'} or search class, default='exhaustive'
        Method used to maximise the query-selection objective. ``'ogb'`` is the orthogonal beam search and
        requires an objective providing ordered prefix values, normally ``OrthogonalBoostingObjective``.
    include_default_rule : bool, default=False
        Add a constant intercept fitted from the target before learning rules.
    include_linear_terms : bool, default=False
        Add standardised numeric columns and categorical indicators as fixed base terms.
    max_col_attr : int, mapping or None, default=10
        Maximum numeric threshold propositions per feature. A mapping sets limits by feature name or index;
        ``None`` retains every distinct threshold. Categorical cardinality is controlled by ``search_params``.
    search_params : dict or None, default=None
        Feature-encoding and search-specific options. See ``docs/general_rule_boosting_estimator.md`` for the
        accepted keys for exhaustive, greedy and orthogonal beam search.
    verbose : bool, default=False
        Print the newest rule and empirical loss after every successful boosting iteration.
    basis_rtol : float or None, default=None
        Relative tolerance for rejecting zero or linearly dependent rule indicators. ``None`` uses a
        floating-point tolerance derived from machine precision.
    objective_params : dict or None, default=None
        Options passed to the objective, such as ``epsilon``, ``maximum_query_length``,
        ``length_regularisation`` or ``hessian_floor``.
    max_components : int, default=10_000
        Safety limit on learned rule components, where a rule costs one plus its number of propositions.
    n_jobs : int, default=1
        Numba CPU workers used by search and large prediction batches. ``-1`` uses the configured Numba maximum.

    Notes
    -----
    The complete interface, supported combinations, nested options, methods and fitted attributes are documented
    in ``docs/general_rule_boosting_estimator.md``. Input features are two-dimensional NumPy arrays. Mixed numeric
    and categorical data may use ``dtype=object``; feature names are supplied through ``search_params``.
    """

    def __init__(self, num_rules=3, objective_function=GradientBoostingObjectiveXGB, weight_update_method=None,
                 loss='squared', reg=1.0, search='exhaustive', include_default_rule=False,
                 include_linear_terms=False, max_col_attr=10, search_params=None, verbose=False, basis_rtol=None,
                 objective_params=None, max_components=10_000, n_jobs=1):
        self.num_rules = num_rules
        self.objective_function = objective_function
        self.weight_update_method = weight_update_method
        self.loss = loss
        self.reg = reg
        self.search = search
        self.include_default_rule = include_default_rule
        self.include_linear_terms = include_linear_terms
        self.max_col_attr = max_col_attr
        self.search_params = search_params
        self.verbose = verbose
        self.basis_rtol = basis_rtol
        self.objective_params = objective_params
        self.max_components = max_components
        self.n_jobs = n_jobs
        self.rules_ = AdditiveRuleEnsemble(n_jobs=n_jobs)

    def get_params(self, deep=True):
        """Return the estimator configuration.

        Parameters
        ----------
        deep : bool, default=True
            If true, include parameters exposed by configured child objects under ``parent__child`` names.

        Returns
        -------
        params : dict[str, object]
            Mapping from parameter names to their current values. The top-level entries correspond exactly to
            the constructor arguments.
        """
        names = [name for name in signature(self.__init__).parameters if name != 'self']
        params = {name: getattr(self, name) for name in names}
        if deep:
            for name, value in tuple(params.items()):
                if hasattr(value, 'get_params'):
                    params.update({f'{name}__{key}': item for key, item in value.get_params().items()})
        return params

    def set_params(self, **params):
        """Set estimator parameters.

        Parameters
        ----------
        **params : object
            Constructor parameters to replace. A ``parent__child`` name updates a child object that implements
            ``set_params``. For example, ``weight_update_method__solver='BFGS'`` targets a nested updater option.

        Returns
        -------
        self : GeneralRuleBoostingEstimator
            Estimator with the requested parameter values. Call ``fit`` to apply changed training parameters.
        """
        for name, value in params.items():
            if '__' in name:
                parent, child = name.split('__', 1)
                getattr(self, parent).set_params(**{child: value})
            else:
                setattr(self, name, value)
        return self

    def set_reg(self, reg):
        """Set the regularisation strength.

        Parameters
        ----------
        reg : float
            Non-negative finite L2 regularisation strength used by the next call to ``fit``.

        Returns
        -------
        self : GeneralRuleBoostingEstimator
            Estimator with the updated ``reg`` value. Existing fitted weights and histories are unchanged until
            the estimator is fitted again.
        """
        self.reg = reg
        return self

    def fit(self, data, target, has_origin_rules=False, verbose=False):
        """Fit an additive rule ensemble.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Two-dimensional feature matrix accepted by ``numpy.asarray``. Numeric matrices may use integer or
            floating-point dtypes. Categorical or mixed matrices may use string or object dtype. Dataframe-like
            objects exposing ``to_numpy`` are accepted for compatibility without making pandas a dependency.
        target : array-like of shape (n_samples,) or (n_samples, 1)
            Finite numeric response values convertible to ``float64``. Squared loss accepts real values,
            logistic loss requires labels encoded exactly as -1 and 1, and Poisson loss requires values >= 0.
        has_origin_rules : bool, default=False
            If false, discard the previous ensemble and fit from the configured base terms. If true, retain the
            existing ensemble and continue until its learned-rule count reaches ``num_rules``.
        verbose : bool, default=False
            Print the selected rule and empirical mean loss after each completed iteration. Output is enabled
            when this value or the constructor's ``verbose`` value is true.

        Returns
        -------
        self : GeneralRuleBoostingEstimator
            Fitted estimator. Learned terms are available through ``rules_`` and iteration diagnostics through
            ``history_``, ``loss_history_`` and ``time_``.

        Raises
        ------
        ValueError
            If ``data`` is not two-dimensional, its row count differs from ``target``, the target is non-finite,
            or the selected loss rejects the target values.
        FloatingPointError
            If query selection or weight optimisation produces a NaN or infinite coefficient.

        Notes
        -----
        Feature names and categorical columns are configured through ``search_params``. Prediction inputs must
        preserve the fitted column order and compatible column value types.
        """
        values = as_2d_array(data)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(values) != len(target):
            raise ValueError('Feature and target arrays must contain the same number of samples')
        if not np.all(np.isfinite(target)):
            raise ValueError('Target contains NaN or infinite values')
        continuing = has_origin_rules and hasattr(self, 'encoder_')
        params = _default_search_params(self.search_params, self.max_col_attr)
        self.search_params_ = params.copy()
        self.n_jobs_ = resolve_numba_jobs(params.get('n_jobs', self.n_jobs))
        self.rules_.n_jobs = self.n_jobs_
        runtime_params = params.copy()
        runtime_params.setdefault('n_jobs', self.n_jobs_)
        feature_names = infer_feature_names(data, values.shape[1], params.get('feature_names'))
        encoder_kwargs = {key: params[key] for key in _ENCODER_KEYS if key in params and key != 'without'}
        encoder = PropositionEncoder(max_col_attr=params.get('max_col_attr', self.max_col_attr),
                                     feature_names=feature_names, **encoder_kwargs)
        context_matrix = encoder.fit_transform(values, without=params.get('without'))
        self.encoder_ = encoder
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(feature_names)
        self.context_matrix_ = context_matrix
        if not continuing:
            self.rules_ = AdditiveRuleEnsemble(n_jobs=self.n_jobs_)
        self.history = []
        self.history_ = self.history
        self.time = []
        self.time_ = self.time
        self.loss_history_ = []
        self.stopping_reason_ = None
        if not continuing:
            self._prepare_base_terms(values, target, encoder)
        self._base_rule_count_ = getattr(self, '_base_rule_count_', 0)
        self._default_rule_count_ = getattr(self, '_default_rule_count_', 0)
        self._initialise_training_cache(values)
        self._basis_ = self._initial_basis(values)
        updater = copy(self.weight_update_method) if self.weight_update_method is not None else KeepWeight()
        updater.loss = self.loss
        updater.reg = float(self.reg)
        self.weight_update_method_ = updater
        if hasattr(updater, 'fixed_prefix'):
            updater.fixed_prefix = self._default_rule_count_
        self._initialise_base_weights(updater, values, target)
        component_count = 0
        while len(self.rules_) - self._base_rule_count_ < self.num_rules and component_count < self.max_components:
            started = perf_counter()
            query, objective, mask, reason = self._find_independent_query(values, target, context_matrix, encoder,
                                                                          runtime_params, verbose)
            if query is None:
                self.stopping_reason_ = reason
                break
            weight = objective.opt_weight(query) if hasattr(objective, 'opt_weight') else 1.0
            if not np.isfinite(weight):
                raise FloatingPointError('Initial rule weight is NaN or infinite')
            self.rules_.append(Rule(query, weight))
            self._append_training_mask(mask)
            weights = self._calculate_weights(updater, values, target)
            if len(weights) != len(self.rules_) or not np.all(np.isfinite(weights)):
                raise FloatingPointError('Weight update returned invalid coefficients')
            for rule, updated_weight in zip(self.rules_, weights):
                rule.y = float(updated_weight)
            self._training_scores_ = self.training_rule_matrix_ @ weights
            component_count += 1 + len(query)
            self.history.append(self.rules_.copy())
            self.loss_history_.append(float(np.mean(loss_function(self.loss)(target, self._training_scores_))))
            self.time.append(perf_counter() - started)
            if self.verbose or verbose:
                print(f'Rule: {self.rules_[-1]}')
                print(f'Loss: {self.loss_history_[-1]:.8g}')
        if self.stopping_reason_ is None:
            reached_target = len(self.rules_) - self._base_rule_count_ >= self.num_rules
            self.stopping_reason_ = 'maximum_rules' if reached_target else 'component_limit'
        self.orth_basis_ = self._basis_.values
        self._build_prediction_plan()
        return self

    def _initial_basis(self, data):
        basis = OrthogonalBasis(len(data), rtol=self.basis_rtol)
        active = list(range(self._default_rule_count_, len(self.rules_)))
        if active:
            basis.rebuild(self.training_rule_matrix_[:, active])
        return basis

    def _prepare_base_terms(self, data, target, encoder):
        self._default_rule_count_ = int(bool(self.include_default_rule))
        if self.include_default_rule:
            self.rules_.append(Rule(Conjunction(), initial_constant(self.loss, target)))
        if self.include_linear_terms:
            for term in make_linear_terms(data, encoder):
                self.rules_.append(Rule(term, 0.0))
        self._base_rule_count_ = len(self.rules_)

    def _initialise_base_weights(self, updater, data, target):
        linear_indices = [index for index, rule in enumerate(self.rules_) if isinstance(rule.q, LinearTerm)]
        if not linear_indices:
            return
        loss = loss_function(self.loss)
        weights = np.asarray([rule.y for rule in self.rules_], dtype=np.float64)
        scores = self.training_rule_matrix_ @ weights
        for index in linear_indices:
            column = self.training_rule_matrix_[:, index]
            gradient, hessian = loss.derivatives(target, scores) if hasattr(loss, 'derivatives') else (
                loss.g(target, scores), loss.h(target, scores)
            )
            denominator = self.reg + np.dot(hessian, np.square(column))
            weight = 0.0 if denominator <= 0 else -np.dot(gradient, column) / denominator
            weights[index] = weight
            self.rules_[index].y = float(weight)
            scores += column * weight
        weights = self._calculate_weights(updater, data, target)
        for rule, weight in zip(self.rules_, weights):
            rule.y = float(weight)
        self._training_scores_ = self.training_rule_matrix_ @ weights

    def _find_independent_query(self, data, target, context_matrix, encoder, params, verbose):
        forbidden = [self.training_rule_matrix_[:, index].astype(bool) for index, rule in enumerate(self.rules_)
                     if isinstance(rule.q, Conjunction)]
        retry_limit = max(8, min(len(encoder.propositions_) + 1, len(data) + 1))
        reason = 'no_query'
        objective = self._new_objective(data, target, context_matrix, encoder)
        for _ in range(retry_limit):
            query = objective.search(method=self.search, verbose=max(bool(verbose), bool(self.verbose)),
                                     forbidden_masks=forbidden, **params)
            if query is None:
                reason = 'duplicate_query_extent' if forbidden else 'no_query'
                return None, objective, None, reason
            mask = np.asarray(query(data), dtype=np.float64)
            if not np.any(mask):
                return None, objective, None, 'empty_query_extent'
            if self._basis_.append(mask):
                return query, objective, mask, None
            duplicate = any(np.array_equal(mask.astype(bool), prior) for prior in forbidden)
            reason = 'duplicate_query_extent' if duplicate else 'linearly_dependent_query_extent'
            forbidden.append(mask.astype(bool))
            if self.search in {'greedy'}:
                return None, objective, None, reason
        return None, objective, None, reason

    def _new_objective(self, data, target, context_matrix, encoder):
        return self.objective_function(data, target, predictions=self._training_scores_, loss=self.loss, reg=self.reg,
                                       rules=self.rules_, orth_basis=self._basis_.values,
                                       context_matrix=context_matrix, propositions=encoder.propositions_,
                                       encoder=encoder, feature_names=self.feature_names_in_,
                                       **(self.objective_params or {}))

    def _initialise_training_cache(self, data):
        if not len(self.rules_):
            self.training_rule_matrix_ = np.empty((len(data), 0), dtype=np.float64)
            self._training_scores_ = np.zeros(len(data), dtype=np.float64)
            return
        self.training_rule_matrix_ = self.rules_.query_matrix(data, self.n_jobs_)
        weights = np.asarray([rule.y for rule in self.rules_], dtype=np.float64)
        self._training_scores_ = self.training_rule_matrix_ @ weights

    def _append_training_mask(self, mask):
        mask = np.asarray(mask, dtype=np.float64).reshape(-1, 1)
        self.training_rule_matrix_ = np.ascontiguousarray(np.column_stack((self.training_rule_matrix_, mask)))

    def _calculate_weights(self, updater, data, target):
        matrix_method = getattr(type(updater), 'calc_weight_from_matrix', None)
        if matrix_method is not None and matrix_method is not WeightUpdateMethod.calc_weight_from_matrix:
            return updater.calc_weight_from_matrix(target, self.rules_, self.training_rule_matrix_)
        return updater.calc_weight(data, target, self.rules_)

    def decision_function(self, data):
        """Calculate raw additive ensemble scores.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features_in_)
            Two-dimensional feature matrix with the same positional column order and compatible value types as
            the fitting data. It may contain any number of rows, including zero rows.

        Returns
        -------
        scores : numpy.ndarray of shape (n_samples,), dtype=float64
            Sum of the fitted default term, linear terms and rule contributions before applying a loss-specific
            prediction link or classification threshold.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If ``data`` is not two-dimensional or does not contain ``n_features_in_`` columns.
        """
        self._check_fitted()
        values = as_2d_array(data)
        if self.rules_.supports_compiled(values):
            return self.rules_(values)
        if self._prediction_plan_ is None:
            return self.rules_(values)
        work = len(values) * max(len(self._prediction_encoder_indices_), 1)
        parallel_context = self.n_jobs_ > 1 and work >= _PREDICTION_PARALLEL_MIN_WORK
        offsets, proposition_indices, positive, negative = self._prediction_plan_
        work = len(values) * max(len(positive), len(proposition_indices), 1)
        parallel_scores = self.n_jobs_ > 1 and work >= _PREDICTION_PARALLEL_MIN_WORK
        manager = numba_threads(self.n_jobs_) if parallel_context or parallel_scores else nullcontext()
        with manager:
            context = self.encoder_.transform(values, self._prediction_encoder_indices_, parallel=parallel_context)
            scores = rule_ensemble_scores(context, offsets, proposition_indices, positive, negative,
                                          parallel=parallel_scores)
            for rule in self._prediction_linear_rules_:
                scores += rule.y * rule.q(values)
            return scores

    def _build_prediction_plan(self):
        lookup = {str(proposition): index for index, proposition in enumerate(self.encoder_.propositions_)}
        selected = []
        local_indices = {}
        flattened = []
        offsets = [0]
        linear_rules = []
        conjunction_rules = []
        for rule in self.rules_:
            if isinstance(rule.q, LinearTerm):
                linear_rules.append(rule)
            elif isinstance(rule.q, Conjunction):
                conjunction_rules.append(rule)
            else:
                self._prediction_plan_ = None
                return
        for rule in conjunction_rules:
            for proposition in rule.q:
                global_index = lookup.get(str(proposition))
                if global_index is None:
                    self._prediction_plan_ = None
                    return
                if global_index not in local_indices:
                    local_indices[global_index] = len(selected)
                    selected.append(global_index)
                flattened.append(local_indices[global_index])
            offsets.append(len(flattened))
        positive = [rule.y for rule in conjunction_rules]
        negative = [rule.z for rule in conjunction_rules]
        self._prediction_linear_rules_ = linear_rules
        self._prediction_encoder_indices_ = np.asarray(selected, dtype=np.int64)
        arrays = ((offsets, np.int64), (flattened, np.int64), (positive, np.float64), (negative, np.float64))
        self._prediction_plan_ = tuple(np.asarray(values, dtype=dtype) for values, dtype in arrays)

    def predict(self, data):
        """Calculate loss-specific predictions.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features_in_)
            Two-dimensional feature matrix following the fitted positional column schema.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples,)
            Squared loss returns ``float64`` raw scores. Logistic loss returns labels -1 or 1. Poisson loss
            returns non-negative ``float64`` conditional means obtained from the exponential link.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If ``data`` has an invalid shape or feature count.
        """
        return loss_function(self.loss).predictions(self.decision_function(data))

    def predict_proba(self, data):
        """Calculate class probabilities for logistic loss.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features_in_)
            Two-dimensional feature matrix following the fitted positional column schema.

        Returns
        -------
        probabilities : numpy.ndarray of shape (n_samples, 2), dtype=float64
            Probabilities whose first column corresponds to class -1 and second column corresponds to class 1.
            Every row sums to one apart from floating-point rounding.

        Raises
        ------
        AttributeError
            If the configured loss does not define class probabilities.
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If ``data`` has an invalid shape or feature count.
        """
        loss = loss_function(self.loss)
        if not hasattr(loss, 'probabilities'):
            raise AttributeError(f'{loss} does not define class probabilities')
        return loss.probabilities(self.decision_function(data))

    def _check_fitted(self):
        if not hasattr(self, 'encoder_'):
            raise RuntimeError('Estimator must be fitted before prediction')


def _default_search_params(params, max_col_attr=10):
    defaults = {'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'max_col_attr': max_col_attr}
    if params is not None:
        defaults.update(params)
    return defaults
