"""Query-selection objectives for additive rule boosting."""

from __future__ import annotations

from math import inf

import numpy as np

from fastogb.accelerators import CudaCandidateStatistics, candidate_statistics
from fastogb.array_ops import extent_signature
from fastogb.encoding import PropositionEncoder, as_2d_array, infer_feature_names
from fastogb.kernels import (absolute_prefix_bound, orthogonal_objective_batch, orthogonal_prefix_values,
                                 prefix_objective_batch, squared_prefix_bound)
from fastogb.linalg import orthogonal_binary_norm, project_away
from fastogb.losses import SquaredLoss, loss_function
from fastogb.search import Context, search_methods


class ObjectFunction:
    """Base class shared by query-selection objectives."""

    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, orth_basis=None,
                 context_matrix=None, propositions=None, encoder=None, feature_names=None, hessian_floor=None,
                 **kwargs):
        self.loss = loss_function(loss)
        self.reg = float(reg)
        self.rules = [] if rules is None else rules
        original_data = as_2d_array(data)
        inferred_names = infer_feature_names(data, original_data.shape[1], feature_names)
        original_target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(original_data) != len(original_target):
            raise ValueError('Feature and target arrays must contain the same number of samples')
        if not np.all(np.isfinite(original_target)):
            raise ValueError('Target contains NaN or infinite values')
        scores = np.zeros_like(original_target) if predictions is None else np.asarray(predictions, dtype=np.float64)
        if scores.shape != original_target.shape or not np.all(np.isfinite(scores)):
            raise ValueError('Predictions must be finite and have the same shape as the target')
        if hasattr(self.loss, 'derivatives'):
            gradient, hessian = self.loss.derivatives(original_target, scores)
        else:
            gradient = np.asarray(self.loss.g(original_target, scores), dtype=np.float64)
            hessian = np.asarray(self.loss.h(original_target, scores), dtype=np.float64)
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            raise FloatingPointError('Loss derivatives contain NaN or infinite values')
        if np.any(hessian < 0):
            raise ValueError('Loss Hessian must be non-negative')
        floor = np.finfo(np.float64).tiny if hessian_floor is None else float(hessian_floor)
        if floor <= 0 or not np.isfinite(floor):
            raise ValueError('hessian_floor must be finite and positive')
        hessian = np.maximum(hessian, floor)
        with np.errstate(over='ignore'):
            ratio = np.divide(gradient, hessian)
        self.order = np.argsort(ratio, kind='stable')[::-1]
        self.inverse_order = np.argsort(self.order)
        self.data = original_data[self.order]
        self.target = original_target[self.order]
        self.g = gradient[self.order]
        self.h = hessian[self.order]
        self.n = len(original_target)
        self.encoder = encoder
        self._cuda_statistics = {}
        self.propositions = tuple(propositions or ())
        self.context_matrix = None if context_matrix is None else np.asarray(context_matrix, dtype=bool)[self.order]
        self.feature_names = inferred_names
        if orth_basis is None:
            self.orth_basis = np.empty((self.n, 0), dtype=np.float64)
        else:
            basis = np.asarray(orth_basis, dtype=np.float64)
            if basis.size == 0:
                basis = np.empty((self.n, 0), dtype=np.float64)
            if basis.ndim != 2 or basis.shape[0] != self.n:
                raise ValueError(f'Orthogonal basis must have {self.n} rows, received shape {basis.shape}')
            tolerance = np.sqrt(32 * np.finfo(np.float64).eps)
            if basis.shape[1] and not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=tolerance,
                                                  atol=tolerance):
                raise ValueError('Orthogonal basis columns must be orthonormal')
            self.orth_basis = basis[self.order]

    def __call__(self, extent):
        raise NotImplementedError

    def bound(self, extent):
        raise NotImplementedError

    def query_extent(self, query):
        return np.flatnonzero(np.asarray(query(self.data), dtype=bool))

    def opt_weight(self, query):
        extent = self.query_extent(query)
        if len(extent) == 0:
            return 0.0
        denominator = self.reg + self.h[extent].sum()
        return 0.0 if denominator <= 0 else -self.g[extent].sum() / denominator

    def _reorder(self, order):
        order = np.asarray(order, dtype=np.int64)
        self.order = self.order[order]
        self.inverse_order = np.argsort(self.order)
        self.data = self.data[order]
        self.target = self.target[order]
        self.g = self.g[order]
        self.h = self.h[order]
        self.orth_basis = self.orth_basis[order]
        if self.context_matrix is not None:
            self.context_matrix = self.context_matrix[order]

    def _candidate_sums(self, parent, attributes, indices, values, backend):
        if backend != 'cuda':
            return candidate_statistics(parent, attributes, indices, values, backend=backend)
        key = (id(attributes), id(values))
        if key not in self._cuda_statistics:
            self._cuda_statistics[key] = CudaCandidateStatistics(attributes, values)
        return self._cuda_statistics[key].evaluate(parent, indices)

    def _prefix_exact_batch(self, parent, attributes, indices, regularisation, scale, mode, parallel):
        return prefix_objective_batch(parent, attributes, indices, self.g, self.h, regularisation, scale, mode,
                                      parallel)

    def search(self, method='greedy', verbose=False, forbidden_masks=None, **search_params):
        try:
            search_type = search_methods[method] if isinstance(method, str) else method
        except KeyError as error:
            raise ValueError(f'Unknown search method {method!r}') from error
        context_params, algorithm_params = _split_search_params(search_params)
        if self.context_matrix is None:
            context_params.setdefault('feature_names', self.feature_names)
            encoder = self.encoder or PropositionEncoder(**context_params)
            matrix = encoder.fit_transform(self.data)
            propositions = encoder.propositions_
        else:
            matrix = self.context_matrix
            propositions = self.propositions
        context = Context.from_binary(matrix, propositions, self.data, sort_attributes=True, encoder=self.encoder)
        signatures = set(algorithm_params.pop('forbidden_signatures', ()))
        for mask in forbidden_masks or ():
            ordered_mask = np.asarray(mask, dtype=bool)[self.order]
            signatures.add(extent_signature(ordered_mask))
        search = search_type(context, self, self.bound, verbose=verbose, forbidden_signatures=signatures,
                             **algorithm_params)
        return search.run()


class GradientBoostingObjectiveXGB(ObjectFunction):
    """XGBoost second-order loss-reduction objective for selecting one rule."""

    def __init__(self, *args, length_regularisation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_regularisation = bool(length_regularisation)

    def __call__(self, extent, depth=0):
        if len(extent) == 0:
            return -inf
        gradient_sum = self.g[extent].sum()
        denominator = self._regularisation(depth) + self.h[extent].sum()
        return -inf if denominator <= 0 else gradient_sum ** 2 / (2 * self.n * denominator)

    def bound(self, extent, depth=0):
        return squared_prefix_bound(self.g, self.h, extent, self._regularisation(depth), 2 * self.n)

    def opt_weight(self, query):
        extent = self.query_extent(query)
        denominator = self._regularisation(len(query)) + self.h[extent].sum()
        return 0.0 if not len(extent) or denominator <= 0 else -self.g[extent].sum() / denominator

    def greedy_values(self, parent, attributes, indices, depth, backend):
        _, gradient_sums = self._candidate_sums(parent, attributes, indices, self.g, backend)
        _, hessian_sums = self._candidate_sums(parent, attributes, indices, self.h, backend)
        denominator = self._regularisation(depth) + hessian_sums
        return np.divide(np.square(gradient_sums), 2 * self.n * denominator,
                         out=np.full(len(indices), -inf), where=denominator > 0)

    def exact_batch(self, parent, attributes, indices, depth, parallel=False):
        regularisation = self._regularisation(depth)
        return self._prefix_exact_batch(parent, attributes, indices, regularisation, 2 * self.n, 0, parallel)

    def _regularisation(self, depth):
        return self.reg * (depth + 1) if self.length_regularisation else self.reg


class GradientBoostingObjectiveMWG(ObjectFunction):
    """Maximum absolute gradient-mass objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reorder(np.argsort(self.g, kind='stable')[::-1])
        self.h = np.ones_like(self.g)

    def __call__(self, extent):
        return -inf if len(extent) == 0 else abs(self.g[extent].sum())

    def bound(self, extent):
        if len(extent) == 0:
            return -inf
        gradient = self.g[extent]
        return float(max(np.abs(np.cumsum(gradient)).max(), np.abs(np.cumsum(gradient[::-1])).max()))

    def greedy_values(self, parent, attributes, indices, depth, backend):
        supports, gradient_sums = self._candidate_sums(parent, attributes, indices, self.g, backend)
        return np.where(supports > 0, np.abs(gradient_sums), -inf)

    def exact_batch(self, parent, attributes, indices, depth, parallel=False):
        return self._prefix_exact_batch(parent, attributes, indices, 0.0, 1.0, 1, parallel)


class GradientBoostingObjectiveGPE(ObjectFunction):
    """Gradient projection efficiency objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reorder(np.argsort(self.g, kind='stable')[::-1])
        self.h = np.ones_like(self.g)

    def __call__(self, extent):
        if len(extent) == 0:
            return -inf
        return abs(self.g[extent].sum()) / (2 * self.n * np.sqrt(len(extent) + self.reg))

    def bound(self, extent):
        return absolute_prefix_bound(self.g, extent, self.reg, 2 * self.n)

    def greedy_values(self, parent, attributes, indices, depth, backend):
        supports, gradient_sums = self._candidate_sums(parent, attributes, indices, self.g, backend)
        denominator = 2 * self.n * np.sqrt(supports + self.reg)
        return np.divide(np.abs(gradient_sums), denominator, out=np.full(len(indices), -inf),
                         where=supports > 0)

    def exact_batch(self, parent, attributes, indices, depth, parallel=False):
        return self._prefix_exact_batch(parent, attributes, indices, self.reg, 2 * self.n, 2, parallel)


class OrthogonalBoostingObjective(ObjectFunction):
    """Select a query after removing components spanned by existing rules."""

    def __init__(self, *args, epsilon=1e-10, length_regularisation=False, maximum_query_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)
        self.length_regularisation = bool(length_regularisation)
        self.maximum_query_length = maximum_query_length
        if self.epsilon <= 0:
            raise ValueError('epsilon must be positive')
        self.projected_gradient = np.ascontiguousarray(project_away(self.orth_basis, self.g))
        self.projected_gradient_norm = float(np.linalg.norm(self.projected_gradient))

    def __call__(self, extent, depth=0):
        if self.maximum_query_length is not None and depth > self.maximum_query_length:
            return -inf
        return self._value(extent, self._regularisation(depth))

    def bound(self, extent, depth=0):
        if self.maximum_query_length is not None and depth > self.maximum_query_length:
            return -inf
        return self._safe_bound(extent, self._regularisation(depth))

    def _value(self, extent, regularisation):
        if len(extent) == 0:
            return -inf
        norm = orthogonal_binary_norm(self.orth_basis, extent, regularisation)
        if norm <= self.epsilon and regularisation == 0:
            return -inf
        return abs(self.projected_gradient[extent].sum()) / (norm + self.epsilon)

    def _safe_bound(self, extent, regularisation):
        if len(extent) == 0:
            return -inf
        lower_denominator = np.sqrt(regularisation) + self.epsilon
        local_bound = np.abs(self.projected_gradient[extent]).sum() / lower_denominator
        return float(min(local_bound, self.projected_gradient_norm))

    def prefix_values(self, order, depth=0):
        """Evaluate all ordered prefixes using Yang Algorithm 3."""
        regularisation = self._regularisation(depth)
        return orthogonal_prefix_values(self.projected_gradient, self.orth_basis, order, self.epsilon,
                                        regularisation)

    def exact_batch(self, parent, attributes, indices, depth, parallel=False):
        if self.maximum_query_length is not None and depth > self.maximum_query_length:
            children = np.bitwise_and(attributes[indices], parent)
            invalid = np.full(len(indices), -inf, dtype=np.float64)
            return children, invalid, invalid.copy()
        regularisation = self._regularisation(depth)
        return orthogonal_objective_batch(parent, attributes, indices, self.projected_gradient, self.orth_basis,
                                          regularisation, self.epsilon, self.projected_gradient_norm, parallel)

    def _regularisation(self, depth):
        return self.reg * (depth + 1) if self.length_regularisation else 0.0


def _split_search_params(params):
    context_keys = {
        'categorical', 'feature_names', 'include_missing', 'max_categories', 'max_col_attr',
        'min_category_count', 'without',
    }
    context = {key: value for key, value in params.items() if key in context_keys}
    algorithm = {key: value for key, value in params.items() if key not in context_keys and key != 'discretization'}
    context.pop('without', None)
    return context, algorithm
