"""A configurable estimator for additive rule ensembles."""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy, deepcopy
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
from fastogb.replacement import no_replacement
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

    def fit(self, data, target, has_origin_rules=None, verbose=False, *, replacement_selector=no_replacement):
        """Initialise, grow or adapt the ensemble according to its current learned-rule count.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Two-dimensional feature matrix accepted by ``numpy.asarray``. Numeric matrices may use integer or
            floating-point dtypes. Categorical or mixed matrices may use string or object dtype. Dataframe-like
            objects exposing ``to_numpy`` are accepted for compatibility without making pandas a dependency.
        target : array-like of shape (n_samples,) or (n_samples, 1)
            Finite numeric response values convertible to ``float64``. Squared loss accepts real values,
            logistic loss requires labels encoded exactly as -1 and 1, and Poisson loss requires values >= 0.
        has_origin_rules : bool or None, default=None
            None detects existing rules automatically. Explicit False requests fresh fitting; explicit True
            preserves legacy continuation, which only grows the ensemble towards ``num_rules``.
        verbose : bool, default=False
            Print the selected rule and empirical mean loss after each completed iteration. Output is enabled
            when this value or the constructor's ``verbose`` value is true.
        replacement_selector : callable, keyword-only, default=no_replacement
            Function accepting an ensemble snapshot and returning a list of zero-based indices to replace.
            Called only when automatic fitting finds enough rules. The default selects none and updates weights.

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
        if has_origin_rules is not None:
            if not isinstance(has_origin_rules, (bool, np.bool_)):
                raise TypeError('has_origin_rules must be a Boolean or None')
            if replacement_selector is not no_replacement:
                raise ValueError('Use automatic fitting to supply a replacement_selector')
            if not has_origin_rules or not len(self.rules_):
                return self.fit_initial_rules(data, target, verbose)
            return self.fit_initial_rules(data, target, verbose, operation='continue')
        if not callable(replacement_selector):
            raise TypeError('replacement_selector must be callable')
        if not len(self.rules_):
            return self.fit_initial_rules(data, target, verbose)
        base_count = sum(isinstance(rule.q, LinearTerm) for rule in self.rules_)
        first = self.rules_[0].q
        base_count += int(self.include_default_rule and isinstance(first, Conjunction) and not len(first))
        if len(self.rules_) - base_count < self.num_rules:
            return self.fit_initial_rules(data, target, verbose, operation='grow')
        return self.replace_rules(data, target, replacement_selector, verbose)

    def update_weights(self, data, target, verbose=False):
        """Keep all queries and adapt their coefficients using the configured weight updater.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features_in_)
            New observations preserving the existing positional feature schema.
        target : array-like of shape (n_samples,) or (n_samples, 1)
            Finite responses accepted by the configured loss.
        verbose : bool, default=False
            Print the updated empirical mean loss.

        Returns
        -------
        self : GeneralRuleBoostingEstimator
            Updated estimator. The updater controls its learning rate and fixed coefficient prefix.
        """
        if not hasattr(self, 'encoder_') and not len(self.rules_):
            raise RuntimeError('Fit an initial ensemble before updating weights')
        return self.fit_initial_rules(data, target, verbose, operation='weights')

    def replace_rules(self, data, target, replacement_selector=no_replacement, verbose=False):
        """Replace selected learned conjunctions; an empty selection requests coefficient-only adaptation.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features_in_)
            New observations preserving the existing positional feature schema.
        target : array-like of shape (n_samples,) or (n_samples, 1)
            Finite responses accepted by the configured loss.
        replacement_selector : callable, default=no_replacement
            Read-only function of an ensemble snapshot returning unique non-negative integer indices. Base terms
            cannot be selected. Any loss-based selection or preliminary coefficient calculation belongs to it.
        verbose : bool, default=False
            Enable iteration output while regrowing the ensemble.

        Returns
        -------
        self : GeneralRuleBoostingEstimator
            Estimator with the selected rules replaced and prediction caches rebuilt.

        Raises
        ------
        ValueError
            If indices are duplicated, out of range or refer to protected base terms.
        RuntimeError
            If search cannot restore the requested rule count. The pre-replacement state is restored.
        """
        if not callable(replacement_selector):
            raise TypeError('replacement_selector must be callable')
        if not len(self.rules_):
            raise RuntimeError('Fit an initial ensemble before replacing rules')
        if replacement_selector is no_replacement:
            return self.update_weights(data, target, verbose)
        base_count = sum(isinstance(rule.q, LinearTerm) for rule in self.rules_)
        first = self.rules_[0].q
        default_count = int(self.include_default_rule and isinstance(first, Conjunction) and not len(first))
        base_count += default_count
        snapshot = deepcopy(self.rules_)
        snapshot._base_rule_count_ = base_count
        indices = replacement_selector(snapshot)
        if not isinstance(indices, list):
            raise TypeError('replacement_selector must return a list of integer indices; use [] to keep all rules')
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in indices):
            raise TypeError('Replacement indices must be integers')
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(self.rules_) for index in indices):
            raise ValueError('Replacement indices must be unique and within the ensemble')
        if any(index < default_count or isinstance(self.rules_[index].q, LinearTerm) for index in indices):
            raise ValueError('Default and linear terms cannot be replaced')
        if not indices:
            return self.update_weights(data, target, verbose)
        state = self.__dict__.copy()
        original = self.rules_
        selected = set(indices)
        desired = max(self.num_rules, len(original) - base_count)
        started = perf_counter()
        values = as_2d_array(data)
        response = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(values) != len(response) or not len(response) or not np.all(np.isfinite(response)):
            raise ValueError('Features and finite targets must have matching non-zero sample counts')
        loss_before = float(np.mean(loss_function(self.loss)(response, original(values))))
        self.rules_ = AdditiveRuleEnsemble([Rule(rule.q, rule.y, rule.z) for i, rule in enumerate(original) if i not in selected],
                                         n_jobs=original.n_jobs)
        try:
            self.fit_initial_rules(data, target, verbose, operation='replace', target_rules=desired)
            if len(self.rules_) - self._base_rule_count_ < desired:
                raise RuntimeError(f'Replacement could not restore {desired} learned rules: {self.stopping_reason_}')
            self.last_update_.update(removed_indices=list(indices), removed_rules=[str(original[i].q) for i in indices],
                                     previous_weights=np.asarray([rule.y for rule in original]), loss_before=loss_before,
                                     elapsed=perf_counter() - started)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(state)
            raise
        return self

    def fit_initial_rules(self, data, target, verbose=False, *, operation='initial', target_rules=None):
        """Fit a fresh ensemble or execute the shared training loop for an adaptation operation.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Numeric, categorical or mixed feature matrix accepted by ``fit``. Initial fitting may change its schema;
            adaptation preserves the fitted feature names, column order and numeric or categorical column roles.
        target : array-like of shape (n_samples,) or (n_samples, 1)
            Finite response values in the domain of the configured loss.
        verbose : bool, default=False
            Enable the same iteration output as ``fit``.
        operation : {'initial', 'grow', 'continue', 'weights', 'replace'}, keyword-only, default='initial'
            Initial fitting resets the ensemble and ignores learning-rate damping. The other modes share preparation
            and the boosting loop with ``fit``, ``update_weights`` and ``replace_rules``. Use those methods for automatic
            dispatch, coefficient-only adaptation and validated replacement selection respectively.
        target_rules : int or None, keyword-only, default=None
            Non-negative learned-rule count for this operation. None uses ``num_rules``. Replacement supplies the count
            needed to restore the ensemble after removing its selected rules.

        Returns
        -------
        self : GeneralRuleBoostingEstimator
            Fitted estimator. Initial fitting resets iteration and adaptation histories.
        """
        if operation not in {'initial', 'grow', 'continue', 'weights', 'replace'}:
            raise ValueError(f'Unknown fitting operation: {operation}')
        if target_rules is not None:
            if isinstance(target_rules, (bool, np.bool_)) or not isinstance(target_rules, (int, np.integer)) or target_rules < 0:
                raise ValueError('target_rules must be a non-negative integer or None')
        values = as_2d_array(data)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(values) != len(target):
            raise ValueError('Feature and target arrays must contain the same number of samples')
        if not len(target):
            raise ValueError('Training data must contain at least one observation')
        if not np.all(np.isfinite(target)):
            raise ValueError('Target contains NaN or infinite values')
        continuing = operation != 'initial'
        if continuing and hasattr(self, 'n_features_in_') and values.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} columns, received {values.shape[1]}')
        params = {'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'max_col_attr': self.max_col_attr}
        if self.search_params is not None:
            params.update(self.search_params)
        jobs = resolve_numba_jobs(params.get('n_jobs', self.n_jobs))
        runtime_params = params.copy()
        runtime_params.setdefault('n_jobs', jobs)
        requested_names = params.get('feature_names')
        if continuing and requested_names is None and not hasattr(data, 'columns'):
            requested_names = getattr(self, 'feature_names_in_', None)
        feature_names = infer_feature_names(data, values.shape[1], requested_names)
        if continuing and hasattr(self, 'feature_names_in_') and tuple(feature_names) != tuple(self.feature_names_in_):
            raise ValueError('Adaptation must preserve the fitted feature names and column order')
        encoder_kwargs = {key: params[key] for key in _ENCODER_KEYS if key in params and key != 'without'}
        if continuing and hasattr(self, 'encoder_') and 'categorical' not in encoder_kwargs:
            encoder_kwargs['categorical'] = tuple(self.encoder_.categorical_indices_)
        if operation == 'weights' and hasattr(self, 'encoder_'):
            encoder = self.encoder_
            context_matrix = encoder.transform(values)
        else:
            encoder = PropositionEncoder(max_col_attr=params.get('max_col_attr', self.max_col_attr),
                                         feature_names=feature_names, **encoder_kwargs)
            context_matrix = encoder.fit_transform(values, without=params.get('without'))
        if continuing and hasattr(self, 'encoder_') and encoder.categorical_indices_ != self.encoder_.categorical_indices_:
            raise ValueError('Adaptation must preserve numeric and categorical column roles')
        model = copy(self)
        model.search_params_ = params.copy()
        model.n_jobs_ = jobs
        model.encoder_ = encoder
        model.n_features_in_ = values.shape[1]
        model.feature_names_in_ = np.asarray(feature_names)
        model.context_matrix_ = context_matrix
        model.rules_ = self.rules_.copy() if continuing else AdditiveRuleEnsemble(n_jobs=jobs)
        model.rules_.n_jobs = jobs
        model.history = []
        model.history_ = model.history
        model.time = []
        model.time_ = model.time
        model.loss_history_ = []
        model.stopping_reason_ = None
        if not continuing:
            model.prepare_base_terms(values, target, encoder)
            model.update_history_ = []
        else:
            first = model.rules_[0].q if len(model.rules_) else None
            model._default_rule_count_ = int(model.include_default_rule and isinstance(first, Conjunction) and not len(first))
            model._base_rule_count_ = model._default_rule_count_ + sum(isinstance(rule.q, LinearTerm) for rule in model.rules_)
        model.rules_._base_rule_count_ = model._base_rule_count_
        model.initialise_training_cache(values)
        model._basis_ = model.initial_basis(values)
        updater = copy(self.weight_update_method) if self.weight_update_method is not None else KeepWeight()
        updater.loss = self.loss
        updater.reg = float(self.reg)
        model.weight_update_method_ = updater
        if hasattr(updater, 'fixed_prefix') and updater.fixed_prefix is None:
            updater.fixed_prefix = model._default_rule_count_
        if not continuing and hasattr(updater, 'learning_rate'):
            updater.learning_rate = 1.0
        previous_weights = np.asarray([rule.y for rule in model.rules_], dtype=np.float64)
        loss_before = float(np.mean(loss_function(self.loss)(target, model._training_scores_)))
        if not continuing or operation == 'continue':
            model.initialise_base_weights(updater, values, target)
        component_count = 0
        desired = self.num_rules if target_rules is None else target_rules
        while operation == 'weights' or len(model.rules_) - model._base_rule_count_ < desired:
            if operation != 'weights' and component_count >= self.max_components:
                break
            started = perf_counter()
            if operation != 'weights':
                query, objective, mask, reason = model.find_independent_query(values, target, context_matrix, encoder,
                                                                            runtime_params, verbose)
                if query is None:
                    model.stopping_reason_ = reason
                    break
                weight = objective.opt_weight(query) if hasattr(objective, 'opt_weight') else 1.0
                if not np.isfinite(weight):
                    raise FloatingPointError('Initial rule weight is NaN or infinite')
                model.rules_.append(Rule(query, weight))
                mask = np.asarray(mask, dtype=np.float64).reshape(-1, 1)
                model.training_rule_matrix_ = np.ascontiguousarray(np.column_stack((model.training_rule_matrix_, mask)))
                component_count += 1 + len(query)
            weights = model.calculate_weights(updater, values, target, new_rule=continuing and operation != 'weights')
            if len(weights) != len(model.rules_) or not np.all(np.isfinite(weights)):
                raise FloatingPointError('Weight update returned invalid coefficients')
            for rule, updated_weight in zip(model.rules_, weights):
                rule.y = float(updated_weight)
            model._training_scores_ = model.training_rule_matrix_ @ weights
            model.history.append(model.rules_.copy())
            model.loss_history_.append(float(np.mean(loss_function(self.loss)(target, model._training_scores_))))
            model.time.append(perf_counter() - started)
            if self.verbose or verbose:
                if operation != 'weights':
                    print(f'Rule: {model.rules_[-1]}')
                print(f'Loss: {model.loss_history_[-1]:.8g}')
            if operation == 'weights':
                break
        if model.stopping_reason_ is None:
            reached_target = len(model.rules_) - model._base_rule_count_ >= desired
            model.stopping_reason_ = 'maximum_rules' if reached_target else 'component_limit'
            if operation == 'weights':
                model.stopping_reason_ = 'weights_updated'
        model.orth_basis_ = model._basis_.values
        model.build_prediction_plan()
        model.last_update_ = {'operation': operation, 'previous_weights': previous_weights,
                              'weights': np.asarray([rule.y for rule in model.rules_]), 'loss_before': loss_before,
                              'loss_after': float(np.mean(loss_function(self.loss)(target, model._training_scores_))),
                              'removed_indices': [], 'elapsed': float(sum(model.time)), 'stopping_reason': model.stopping_reason_}
        if continuing:
            model.update_history_ = list(getattr(self, 'update_history_', [])) + [model.last_update_]
        self.__dict__.update(model.__dict__)
        return self

    def initial_basis(self, data):
        basis = OrthogonalBasis(len(data), rtol=self.basis_rtol)
        active = list(range(self._default_rule_count_, len(self.rules_)))
        if active:
            basis.rebuild(self.training_rule_matrix_[:, active])
        return basis

    def prepare_base_terms(self, data, target, encoder):
        self._default_rule_count_ = int(bool(self.include_default_rule))
        if self.include_default_rule:
            self.rules_.append(Rule(Conjunction(), initial_constant(self.loss, target)))
        if self.include_linear_terms:
            for term in make_linear_terms(data, encoder):
                self.rules_.append(Rule(term, 0.0))
        self._base_rule_count_ = len(self.rules_)

    def initialise_base_weights(self, updater, data, target):
        linear_indices = [index for index, rule in enumerate(self.rules_) if isinstance(rule.q, LinearTerm)]
        if not linear_indices:
            return
        loss = loss_function(self.loss)
        weights = np.asarray([rule.y for rule in self.rules_], dtype=np.float64)
        scores = self.training_rule_matrix_ @ weights
        for index in linear_indices:
            column = self.training_rule_matrix_[:, index]
            if hasattr(loss, 'derivatives'):
                gradient, hessian = loss.derivatives(target, scores)
            else:
                gradient, hessian = loss.g(target, scores), loss.h(target, scores)
            denominator = self.reg + np.dot(hessian, np.square(column))
            weight = 0.0 if denominator <= 0 else -np.dot(gradient, column) / denominator
            weights[index] = weight
            self.rules_[index].y = float(weight)
            scores += column * weight
        weights = self.calculate_weights(updater, data, target)
        for rule, weight in zip(self.rules_, weights):
            rule.y = float(weight)
        self._training_scores_ = self.training_rule_matrix_ @ weights

    def find_independent_query(self, data, target, context_matrix, encoder, params, verbose):
        forbidden = [self.training_rule_matrix_[:, index].astype(bool) for index, rule in enumerate(self.rules_)
                     if isinstance(rule.q, Conjunction)]
        retry_limit = max(8, min(len(encoder.propositions_) + 1, len(data) + 1))
        reason = 'no_query'
        objective = self.objective_function(data, target, predictions=self._training_scores_, loss=self.loss, reg=self.reg,
                                           rules=self.rules_, orth_basis=self._basis_.values, context_matrix=context_matrix,
                                           propositions=encoder.propositions_, encoder=encoder, feature_names=self.feature_names_in_,
                                           **(self.objective_params or {}))
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

    def initialise_training_cache(self, data):
        if not len(self.rules_):
            self.training_rule_matrix_ = np.empty((len(data), 0), dtype=np.float64)
            self._training_scores_ = np.zeros(len(data), dtype=np.float64)
            return
        self.training_rule_matrix_ = self.rules_.query_matrix(data, self.n_jobs_)
        weights = np.asarray([rule.y for rule in self.rules_], dtype=np.float64)
        self._training_scores_ = self.training_rule_matrix_ @ weights

    def calculate_weights(self, updater, data, target, new_rule=False):
        matrix_method = getattr(type(updater), 'calc_weight_from_matrix', None)
        if matrix_method is not None and matrix_method is not WeightUpdateMethod.calc_weight_from_matrix:
            if new_rule and 'previous_weights' in signature(matrix_method).parameters:
                previous = np.asarray([rule.y for rule in self.rules_], dtype=np.float64)
                previous[-1] = 0.0
                return updater.calc_weight_from_matrix(target, self.rules_, self.training_rule_matrix_, previous_weights=previous)
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
        if not hasattr(self, 'encoder_'):
            raise RuntimeError('Estimator must be fitted before prediction')
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

    def build_prediction_plan(self):
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
        loss = loss_function(self.loss)
        scores = self.decision_function(data)
        return loss.predictions(scores)

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
