"""Model-evaluation helpers for fastogb models returning NumPy values."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss as sklearn_log_loss, r2_score, roc_auc_score


def r2(data, target):
    return lambda model: r2_score(target, model.predict(data))


def accuracy(data, target):
    return lambda model: accuracy_score(target, model.predict(data))


def roc_auc(data, target):
    def metric(model):
        scores = model(data) if callable(model) else model.predict_proba(data)[:, 1]
        return roc_auc_score(target, scores)
    return metric


def log_loss(data, target):
    return lambda model: sklearn_log_loss(target, model.predict_proba(data)[:, 1])


def ensemble_length_vs_perf(model, metric):
    """Evaluate every ensemble prefix, including the empty ensemble."""
    return np.asarray([metric(model[:length]) for length in range(len(model.members) + 1)], dtype=np.float64)
