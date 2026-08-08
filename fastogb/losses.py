"""Loss functions for additive rule ensembles."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, xlogy

from fastogb.kernels import loss_derivatives


class SquaredLoss:
    name = 'squared'

    def __call__(self, target, scores):
        return np.square(np.asarray(target) - np.asarray(scores))

    def predictions(self, scores):
        return np.asarray(scores)

    def g(self, target, scores):
        return 2 * (np.asarray(scores) - np.asarray(target))

    def h(self, target, scores):
        return np.full(np.asarray(scores).shape, 2.0, dtype=np.float64)

    def derivatives(self, target, scores):
        return loss_derivatives(target, scores, 0)

    def __repr__(self):
        return 'squared_loss'

    def __str__(self):
        return self.name


class LogisticLoss:
    name = 'logistic'

    def __call__(self, target, scores):
        target = _binary_target(target)
        scores = np.asarray(scores, dtype=np.float64)
        return np.logaddexp(0.0, -target * scores)

    def predictions(self, scores):
        return np.where(np.asarray(scores) >= 0, 1, -1)

    def probabilities(self, scores):
        positive = expit(np.asarray(scores, dtype=np.float64))
        return np.column_stack((1 - positive, positive))

    def g(self, target, scores):
        target = _binary_target(target)
        return -target * expit(-target * np.asarray(scores, dtype=np.float64))

    def h(self, target, scores):
        probability = expit(-_binary_target(target) * np.asarray(scores))
        return probability * (1 - probability)

    def derivatives(self, target, scores):
        target = _binary_target(target)
        return loss_derivatives(target, scores, 1)

    def __repr__(self):
        return 'logistic_loss'

    def __str__(self):
        return self.name


class PoissonLoss:
    name = 'poisson'

    def __call__(self, target, scores):
        target = _count_target(target)
        scores = np.asarray(scores, dtype=np.float64)
        mean = np.exp(np.clip(scores, -745, 709))
        return mean - target * scores + xlogy(target, target) - target

    def predictions(self, scores):
        return np.exp(np.clip(np.asarray(scores, dtype=np.float64), -745, 709))

    def g(self, target, scores):
        return self.predictions(scores) - _count_target(target)

    def h(self, target, scores):
        return self.predictions(scores)

    def derivatives(self, target, scores):
        target = _count_target(target)
        return loss_derivatives(target, scores, 2)

    def __repr__(self):
        return 'poisson_loss'

    def __str__(self):
        return self.name


squared_loss = SquaredLoss()
logistic_loss = LogisticLoss()
poisson_loss = PoissonLoss()

loss_functions = {
    'squared': squared_loss,
    'logistic': logistic_loss,
    'poisson': poisson_loss,
}


def loss_function(loss):
    if isinstance(loss, type):
        loss = loss()
    if callable(loss):
        return loss
    try:
        return loss_functions[loss]
    except KeyError as error:
        raise ValueError(f'Unknown loss function {loss!r}') from error


def _binary_target(target):
    target = np.asarray(target, dtype=np.float64)
    if not np.all(np.isin(target, (-1.0, 1.0))):
        raise ValueError('Logistic loss requires targets encoded as -1 and 1')
    return target


def _count_target(target):
    target = np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError('Poisson loss requires finite non-negative targets')
    return target
