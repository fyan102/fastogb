"""Impact-based subgroup discovery over NumPy feature arrays."""

from __future__ import annotations

from math import inf

import numpy as np

from fastogb.encoding import PropositionEncoder, as_2d_array, infer_feature_names
from fastogb.search import Context, search_methods


class Impact:
    """Coverage-weighted deviation of a subgroup target mean from the global mean."""

    def __init__(self, data, target, alpha=1.0, feature_names=None, max_col_attr=10, categorical=None):
        values = as_2d_array(data)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(values) != len(target):
            raise ValueError('Feature and target arrays must contain the same number of samples')
        self.m = len(values)
        self.alpha = float(alpha)
        self.mean = float(np.mean(target))
        self.order = np.argsort(target, kind='stable')[::-1]
        self.data = values[self.order]
        self.target = target[self.order]
        names = infer_feature_names(data, values.shape[1], feature_names)
        self.encoder = PropositionEncoder(max_col_attr=max_col_attr, categorical=categorical, feature_names=names)
        self.matrix = self.encoder.fit_transform(values)[self.order]

    def __call__(self, query):
        extent = np.flatnonzero(np.asarray(query(self.data), dtype=bool))
        return self.value(extent)

    def value(self, extent):
        if len(extent) == 0:
            return -inf
        local_mean = self.target[extent].mean()
        return float((len(extent) / self.m) ** self.alpha * (local_mean - self.mean))

    def bound(self, extent):
        if len(extent) == 0:
            return -inf
        target = self.target[extent]
        lengths = np.arange(1, len(extent) + 1, dtype=np.float64)
        values = (lengths / self.m) ** self.alpha * (np.cumsum(target) / lengths - self.mean)
        return float(values.max())

    def search(self, search='exhaustive', verbose=False, **search_params):
        context = Context.from_binary(self.matrix, self.encoder.propositions_, self.data, sort_attributes=True)
        search_type = search_methods[search] if isinstance(search, str) else search
        return search_type(context, self.value, self.bound, verbose=verbose, **search_params).run()
