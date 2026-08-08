"""Linear design terms used alongside rule indicators."""

from __future__ import annotations

import numpy as np

from fastogb.encoding import as_2d_array


class LinearTerm:
    """A standardised numeric column or a categorical indicator."""

    def __init__(self, column_index, name, location=0.0, scale=1.0, proposition=None):
        self.column_index = int(column_index)
        self.name = name
        self.location = float(location)
        self.scale = float(scale)
        self.proposition = proposition

    def __call__(self, data):
        if self.proposition is not None:
            return np.asarray(self.proposition(data), dtype=np.float64)
        column = np.asarray(as_2d_array(data)[:, self.column_index], dtype=np.float64)
        values = (column - self.location) / self.scale
        return np.where(np.isfinite(values), values, 0.0)

    def __len__(self):
        return 1

    def __repr__(self):
        return f'linear({self.proposition})' if self.proposition is not None else f'linear({self.name})'


def make_linear_terms(data, encoder):
    """Create numeric terms and one-hot terms from a fitted proposition encoder."""
    values = as_2d_array(data)
    terms = []
    categorical = set(encoder.categorical_indices_)
    for index, name in enumerate(encoder.feature_names_in_):
        if index in categorical:
            propositions = [item for item in encoder.propositions_ if item.column_index == index]
            terms.extend(LinearTerm(index, name, proposition=proposition) for proposition in propositions)
            continue
        column = np.asarray(values[:, index], dtype=np.float64)
        finite = column[np.isfinite(column)]
        if not len(finite):
            continue
        location = float(np.mean(finite))
        scale = float(np.std(finite))
        tolerance = np.sqrt(32 * np.finfo(np.float64).eps) * max(np.linalg.norm(finite), 1.0)
        if scale * np.sqrt(len(finite)) > tolerance:
            terms.append(LinearTerm(index, name, location, scale))
    return terms
