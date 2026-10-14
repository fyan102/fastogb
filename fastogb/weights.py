"""Weight-update methods for additive rule ensembles."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from fastogb.losses import LogisticLoss, PoissonLoss, SquaredLoss, loss_function
from fastogb.prediction import evaluate_rule_queries


def rule_matrix(data, rules):
    """Evaluate rule queries once and return a dense sample-by-rule matrix."""
    method = getattr(rules, 'query_matrix', None)
    return method(data) if method is not None else evaluate_rule_queries(data, rules)


class WeightUpdateMethod:
    """Configure coefficient calculation; ``None`` lets the estimator choose its fixed default-rule prefix.

    ``learning_rate`` defaults to one and must be finite and in (0, 1]. ``fixed_prefix`` is a non-negative integer
    or None. Direct calls treat None as zero. Custom methods decide whether to use ``apply_learning_rate``.
    """

    def __init__(self, loss='squared', reg=1.0, learning_rate=1.0, fixed_prefix=None):
        self.loss = loss
        self.reg = float(reg)
        self.learning_rate = float(learning_rate)
        self.fixed_prefix = fixed_prefix
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('learning_rate must be finite and in (0, 1]')
        if fixed_prefix is not None and (isinstance(fixed_prefix, bool) or not isinstance(fixed_prefix, (int, np.integer))
                                         or fixed_prefix < 0):
            raise ValueError('fixed_prefix must be a non-negative integer or None')

    def __setstate__(self, state):
        """Restore updater defaults from versions preceding learning rates and explicit prefix configuration."""
        self.__dict__.update(state)
        if 'learning_rate' not in state:
            self.learning_rate = 1.0
            if state.get('fixed_prefix', 0) == 0:
                self.fixed_prefix = None

    def calc_weight(self, data, target, rules):
        matrix = rule_matrix(data, rules)
        return self.calc_weight_from_matrix(target, rules, matrix)

    def calc_weight_from_matrix(self, target, rules, matrix):
        raise NotImplementedError

    def apply_learning_rate(self, previous, proposed):
        """Apply this updater's coefficient step; initial estimator fitting uses a copied updater with rate one."""
        rate = float(getattr(self, 'learning_rate', 1.0))
        if not np.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError('learning_rate must be finite and in (0, 1]')
        return proposed if rate == 1 else previous + rate * (proposed - previous)


class KeepWeight(WeightUpdateMethod):
    """Keep the weights supplied when rules were added."""

    def calc_weight_from_matrix(self, target, rules, matrix):
        return np.asarray([rule.y for rule in rules], dtype=np.float64)


class FullyCorrective(WeightUpdateMethod):
    """Jointly optimise active weights and apply the configured coefficient step from their previous values.

    ``previous_weights`` in matrix calls is optional and supplies the interpolation origin for newly added rules.
    The initial solver point still comes from the rule coefficients. A rate of one returns the unmodified optimum.
    """

    def __init__(self, loss='squared', reg=1.0, solver='Newton-CG', options=None, learning_rate=1.0, fixed_prefix=None):
        super().__init__(loss, reg, learning_rate, fixed_prefix)
        self.solver = solver
        self.options = options

    def calc_weight_from_matrix(self, target, rules, matrix, previous_weights=None):
        matrix = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if matrix.shape[1] == 0:
            return np.empty(0, dtype=np.float64)
        loss = loss_function(self.loss)
        initial = np.asarray([rule.y for rule in rules], dtype=np.float64)
        previous = initial if previous_weights is None else np.asarray(previous_weights, dtype=np.float64)
        fixed = min(int(self.fixed_prefix or 0), matrix.shape[1])
        if fixed == matrix.shape[1]:
            return initial
        fixed_scores = matrix[:, :fixed] @ initial[:fixed] if fixed else np.zeros(len(target), dtype=np.float64)
        active = matrix[:, fixed:]
        if isinstance(loss, SquaredLoss):
            system = active.T @ active + (self.reg / 2) * np.eye(active.shape[1])
            response = active.T @ (target - fixed_scores)
            try:
                solution = np.linalg.solve(system, response)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(system, response, rcond=None)[0]
            return self.apply_learning_rate(previous, np.concatenate((initial[:fixed], solution)))
        active_initial = initial[fixed:]

        def risk(weights):
            scores = fixed_scores + active @ weights
            return float(np.sum(loss(target, scores)) + self.reg * np.dot(weights, weights) / 2)

        def gradient(weights):
            scores = fixed_scores + active @ weights
            return active.T @ loss.g(target, scores) + self.reg * weights

        def hessian(weights):
            curvature = loss.h(target, fixed_scores + active @ weights)
            return active.T @ (curvature[:, None] * active) + self.reg * np.eye(active.shape[1])

        def hessian_product(weights, direction):
            curvature = loss.h(target, fixed_scores + active @ weights)
            return active.T @ (curvature * (active @ direction)) + self.reg * direction

        method = 'L-BFGS-B' if self.solver in {'GD', 'Line'} else self.solver
        kwargs = {'jac': gradient, 'method': method, 'options': self.options or {'disp': False}}
        if method in {'Newton-CG', 'trust-krylov', 'trust-ncg'}:
            kwargs['hessp'] = hessian_product
        elif method in {'trust-exact', 'dogleg'}:
            kwargs['hess'] = hessian
        result = minimize(risk, active_initial, **kwargs)
        if not np.all(np.isfinite(result.x)):
            raise FloatingPointError(f'Weight optimisation failed: {result.message}')
        stationarity = np.linalg.norm(gradient(result.x), ord=np.inf)
        if not result.success and stationarity > 1e-6 * (1 + np.linalg.norm(result.x, ord=np.inf)):
            raise RuntimeError(f'Weight optimisation failed: {result.message}')
        return self.apply_learning_rate(previous, np.concatenate((initial[:fixed], result.x)))


class LineSearch(WeightUpdateMethod):
    """Optimise only the newest rule weight while retaining earlier weights."""

    def calc_weight_from_matrix(self, target, rules, matrix, previous_weights=None):
        matrix = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        weights = np.asarray([rule.y for rule in rules], dtype=np.float64)
        previous = weights if previous_weights is None else np.asarray(previous_weights, dtype=np.float64)
        if not len(weights):
            return weights
        if len(weights) <= int(self.fixed_prefix or 0):
            return weights
        loss = loss_function(self.loss)
        fixed_scores = matrix[:, :-1] @ weights[:-1]
        newest = matrix[:, -1]

        def risk(value):
            score = fixed_scores + newest * value[0]
            penalty = self.reg * (np.dot(weights[:-1], weights[:-1]) + value[0] ** 2) / 2
            return float(np.sum(loss(target, score)) + penalty)

        def gradient(value):
            score = fixed_scores + newest * value[0]
            return np.asarray([newest @ loss.g(target, score) + self.reg * value[0]])

        result = minimize(risk, weights[-1:], jac=gradient, method='L-BFGS-B')
        if not np.isfinite(result.x[0]):
            raise FloatingPointError(f'Line search failed: {result.message}')
        if not result.success and abs(gradient(result.x)[0]) > 1e-6 * (1 + abs(result.x[0])):
            raise RuntimeError(f'Line search failed: {result.message}')
        proposed = weights.copy()
        proposed[-1] = result.x[0]
        return self.apply_learning_rate(previous, proposed)


def initial_constant(loss, target):
    """Minimise the unregularised empirical loss over a finite constant score."""
    loss = loss_function(loss)
    target = np.asarray(target, dtype=np.float64)
    if not len(target):
        raise ValueError('Cannot fit an intercept to an empty target')
    if isinstance(loss, SquaredLoss):
        return float(np.mean(target))
    if isinstance(loss, LogisticLoss):
        if not np.all(np.isin(target, (-1.0, 1.0))):
            raise ValueError('Logistic loss requires targets encoded as -1 and 1')
        probability = np.mean(target == 1)
        probability = np.clip(probability, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
        return float(np.log(probability / (1 - probability)))
    if isinstance(loss, PoissonLoss):
        loss(target, np.zeros(len(target), dtype=np.float64))
        mean = max(float(np.mean(target)), np.finfo(np.float64).tiny)
        return float(np.log(mean))
    result = minimize_scalar(lambda value: float(np.sum(loss(target, np.full(len(target), value)))))
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f'Intercept optimisation failed: {result.message}')
    return float(result.x)
