"""Fused evaluation of several additive rule ensembles."""

from __future__ import annotations

import numpy as np

from fastogb.prediction import build_rule_collection_plan, evaluate_rule_collection


class RuleEnsembleCollection:
    """Evaluate several estimators or ensembles through one compiled prediction plan.

    Call ``refresh`` after any member gains, loses or reweights rules.
    """

    def __init__(self, members, n_jobs=-1):
        self.n_jobs = n_jobs
        self.set_members(members)

    def __len__(self):
        return len(self.members)

    def set_members(self, members):
        """Replace the estimators or ensembles and rebuild the fused plan."""
        self.members = list(members)
        return self.refresh()

    def refresh(self):
        """Rebuild descriptors and cached weights after member models change."""
        self.ensembles = [getattr(member, 'rules_', member) for member in self.members]
        if any(not hasattr(ensemble, '__iter__') for ensemble in self.ensembles):
            raise TypeError('Every collection member must be an estimator or iterable rule ensemble')
        self._evaluation_plan = build_rule_collection_plan(self.ensembles)
        return self

    def decision_function(self, data):
        """Return raw scores with shape ``(n_samples, n_members)``."""
        scores = evaluate_rule_collection(data, self._evaluation_plan, self.n_jobs)
        if scores is not None:
            return scores
        values = _as_2d_data(data)
        columns = []
        for member, ensemble in zip(self.members, self.ensembles):
            method = getattr(member, 'decision_function', None)
            fitted = hasattr(member, 'encoder_')
            columns.append(method(values) if method is not None and fitted else ensemble(values))
        return np.column_stack(columns) if columns else np.empty((len(values), 0), dtype=np.float64)

    def argmax(self, data):
        """Return the member index with the largest raw score for every observation."""
        if not self.members:
            raise ValueError('Cannot choose from an empty rule ensemble collection')
        return np.argmax(self.decision_function(data), axis=1)


def _as_2d_data(data):
    values = data.to_numpy(copy=False) if hasattr(data, 'to_numpy') else np.asarray(data)
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f'Expected one- or two-dimensional feature data, received shape {values.shape}')
    return data
