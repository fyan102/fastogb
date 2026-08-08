"""Rules and additive rule ensembles."""

from __future__ import annotations

import numpy as np

from fastogb.logic import Conjunction


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
    """An ordered collection of additively combined rules."""

    def __init__(self, members=None):
        self.members = [] if members is None else list(members)

    def __repr__(self):
        return '\n'.join(str(rule) for rule in self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return AdditiveRuleEnsemble(self.members[item]) if isinstance(item, slice) else self.members[item]

    def __iter__(self):
        return iter(self.members)

    def __call__(self, data):
        scores = np.zeros(len(data), dtype=np.float64)
        for rule in self.members:
            scores += rule(data)
        return scores

    def append(self, rule):
        self.members.append(rule)
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
            return self
        return AdditiveRuleEnsemble(members)

    def copy(self):
        return AdditiveRuleEnsemble([Rule(rule.q, rule.y, rule.z) for rule in self.members])
