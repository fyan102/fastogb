"""Rules and additive rule ensembles."""

from __future__ import annotations

import numpy as np

from fastogb.logic import Conjunction
from fastogb.prediction import (build_rule_evaluation_plan, evaluate_rule_ensemble, evaluate_rule_queries,
                                rule_structure_signature)


class Rule:
    """A rule returning ``y`` when its query holds and ``z`` otherwise."""

    def __init__(self, q=None, y=0.0, z=0.0):
        self.q = Conjunction() if q is None else q
        self.y = y
        self.z = z

    def __call__(self, data):
        satisfied = np.asarray(self.q(data), dtype=np.float64)
        return satisfied * self.y + (1 - satisfied) * self.z

    def __repr__(self):
        return f'{self.y:+10.4f} if {self.q}'


class AdditiveRuleEnsemble:
    """An ordered collection of additively combined rules with adaptive compiled evaluation."""

    def __init__(self, members=None, n_jobs=-1):
        self.members = [] if members is None else list(members)
        self.n_jobs = n_jobs
        self._evaluation_plan = None
        self._evaluation_signature = None

    def __repr__(self):
        return '\n'.join(str(rule) for rule in self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return AdditiveRuleEnsemble(self.members[item], self.n_jobs) if isinstance(item, slice) else self.members[item]

    def __iter__(self):
        return iter(self.members)

    def __setstate__(self, state):
        """Restore older pickles that predate compiled prediction plans."""
        self.__dict__.update(state)
        self.n_jobs = state.get('n_jobs', -1)
        self._evaluation_plan = None
        self._evaluation_signature = None

    def __call__(self, data):
        """Return additive scores for one observation or a two-dimensional feature matrix."""
        return evaluate_rule_ensemble(data, self.members, self._get_evaluation_plan(), self.n_jobs)

    def query_matrix(self, data, n_jobs=None):
        """Return one evaluated query column per rule."""
        jobs = self.n_jobs if n_jobs is None else n_jobs
        return evaluate_rule_queries(data, self.members, self._get_evaluation_plan(), jobs)

    def supports_compiled(self, data):
        """Return whether package-native queries can evaluate the supplied data through Numba."""
        if self._get_evaluation_plan() is None:
            return False
        values = data.to_numpy(copy=False) if hasattr(data, 'to_numpy') else np.asarray(data)
        if values.ndim not in {1, 2}:
            return False
        try:
            np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return True

    def append(self, rule):
        self.members.append(rule)
        self._invalidate_evaluation_plan()
        return self

    def size(self):
        return len(self.members) + sum(len(rule.q) for rule in self.members)

    def consolidated(self, inplace=False):
        grouped = {}
        for rule in self.members:
            key = str(rule.q)
            if key not in grouped:
                grouped[key] = Rule(rule.q, rule.y, rule.z)
            else:
                grouped[key].y += rule.y
                grouped[key].z += rule.z
        members = list(grouped.values())
        if inplace:
            self.members = members
            self._invalidate_evaluation_plan()
            return self
        return AdditiveRuleEnsemble(members, self.n_jobs)

    def copy(self):
        return AdditiveRuleEnsemble([Rule(rule.q, rule.y, rule.z) for rule in self.members], self.n_jobs)

    def _get_evaluation_plan(self):
        signature = rule_structure_signature(self.members)
        if signature != self._evaluation_signature:
            self._evaluation_plan = build_rule_evaluation_plan(self.members)
            self._evaluation_signature = signature
        return self._evaluation_plan

    def _invalidate_evaluation_plan(self):
        self._evaluation_plan = None
        self._evaluation_signature = None
