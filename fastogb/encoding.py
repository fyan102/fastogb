"""NumPy feature encoding for rule-search contexts."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number

import numpy as np

from fastogb.kernels import proposition_matrix
from fastogb.logic import Constraint, KeyValueProposition


_OPERATION_CODES = {'<': 0, '<=': 1, '>': 2, '>=': 3, '==': 4}


def as_2d_array(data):
    """Return input data as a two-dimensional NumPy array without importing pandas."""
    values = data.to_numpy(copy=False) if hasattr(data, 'to_numpy') else np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f'Expected a two-dimensional feature array, received shape {values.shape}')
    return values


def infer_feature_names(data, n_features, feature_names=None):
    if feature_names is not None:
        names = tuple(feature_names)
    elif hasattr(data, 'columns'):
        names = tuple(data.columns)
    else:
        names = tuple(f'x{i}' for i in range(n_features))
    if len(names) != n_features:
        raise ValueError(f'Expected {n_features} feature names, received {len(names)}')
    return names


class PropositionEncoder:
    """Generate numeric thresholds and one-hot categorical propositions."""

    def __init__(self, max_col_attr=10, categorical=None, feature_names=None, max_categories=None,
                 min_category_count=1, include_missing=True):
        self.max_col_attr = max_col_attr
        self.categorical = categorical
        self.feature_names = feature_names
        self.max_categories = max_categories
        self.min_category_count = min_category_count
        self.include_missing = include_missing

    def fit(self, data, without=None):
        values = as_2d_array(data)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = infer_feature_names(data, values.shape[1], self.feature_names)
        excluded = self._column_indices(without or ())
        self.categorical_indices_ = self._categorical_indices(values)
        self.category_maps_ = {}
        propositions = []
        for index, name in enumerate(self.feature_names_in_):
            if index in excluded:
                continue
            column = values[:, index]
            factory = (
                self._categorical_propositions if index in self.categorical_indices_ else self._numeric_propositions
            )
            propositions.extend(factory(column, index, name))
        self.propositions_ = tuple(propositions)
        self._build_descriptors()
        return self

    def transform(self, data, proposition_indices=None, parallel=None):
        self._check_fitted()
        values = as_2d_array(data)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} columns, received {values.shape[1]}')
        indices = self._proposition_indices(proposition_indices)
        if not len(indices):
            return np.empty((len(values), 0), dtype=bool)
        encoded = self._encode_columns(values, np.unique(self.proposition_columns_[indices]))
        return proposition_matrix(encoded, self.proposition_columns_[indices], self.proposition_operations_[indices],
                                  self.proposition_operands_[indices], parallel=parallel)

    def fit_transform(self, data, without=None):
        return self.fit(data, without=without).transform(data)

    def _check_fitted(self):
        if not hasattr(self, 'propositions_'):
            raise RuntimeError('PropositionEncoder must be fitted before transform')

    def _column_indices(self, columns):
        names = {name: index for index, name in enumerate(self.feature_names_in_)}
        return {names[column] if column in names else int(column) for column in columns}

    def _categorical_indices(self, values):
        if self.categorical is None:
            if values.dtype.kind in 'iuf':
                return set()
            if values.dtype.kind in 'bSU':
                return set(range(values.shape[1]))
            return {index for index in range(values.shape[1]) if not _is_numeric(values[:, index])}
        if isinstance(self.categorical, np.ndarray) and self.categorical.dtype == bool:
            if self.categorical.shape != (values.shape[1],):
                raise ValueError('Categorical mask must contain one entry per feature')
            return set(np.flatnonzero(self.categorical))
        return self._column_indices(self.categorical)

    def _categorical_propositions(self, column, index, name):
        if column.dtype.kind in 'SU':
            unique, first, frequencies = np.unique(column, return_index=True, return_counts=True)
            order = np.argsort(first, kind='stable')
            counts = {unique[position]: int(frequencies[position]) for position in order}
            missing = False
        else:
            counts = {}
            missing = False
            for value in column:
                if _is_missing(value):
                    missing = True
                else:
                    counts[value] = counts.get(value, 0) + 1
        categories = [value for value, count in counts.items() if count >= self.min_category_count]
        if self.max_categories is not None and len(categories) > self.max_categories:
            categories = sorted(categories, key=lambda value: (-counts[value], str(value)))[:self.max_categories]
        self.category_maps_[index] = {value: code for code, value in enumerate(categories)}
        propositions = [KeyValueProposition(name, Constraint.equals(value), index) for value in categories]
        if missing and self.include_missing:
            propositions.append(KeyValueProposition(name, Constraint.missing(), index))
        return propositions

    def _numeric_propositions(self, column, index, name):
        numeric = np.asarray(column, dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        unique = np.unique(finite)
        propositions = []
        if len(unique) > 1:
            max_attributes = self._max_attributes(index, name)
            if max_attributes is not None and 2 * len(unique) > max_attributes:
                count = max(1, max_attributes // 2)
                # Match the legacy qcut search space by retaining every upper bin edge, including the maximum.
                quantiles = np.linspace(0.0, 1.0, count + 1)[1:]
                thresholds = np.unique(np.quantile(finite, quantiles))
                for threshold in thresholds:
                    propositions.append(KeyValueProposition(name, Constraint.less_equals(threshold), index))
                    propositions.append(KeyValueProposition(name, Constraint.greater_equals(threshold), index))
            else:
                for threshold in unique[:-1]:
                    propositions.append(KeyValueProposition(name, Constraint.less_equals(threshold), index))
                for threshold in unique[1:]:
                    propositions.append(KeyValueProposition(name, Constraint.greater_equals(threshold), index))
        if len(finite) != len(numeric) and self.include_missing:
            propositions.append(KeyValueProposition(name, Constraint.missing(), index))
        return propositions

    def _max_attributes(self, index, name):
        if not isinstance(self.max_col_attr, Mapping):
            return self.max_col_attr
        return self.max_col_attr[name] if name in self.max_col_attr else self.max_col_attr.get(index)

    def _build_descriptors(self):
        columns = []
        operations = []
        operands = []
        for proposition in self.propositions_:
            column = proposition.column_index
            operation = proposition.constraint.op
            columns.append(column)
            if operation == 'missing':
                operations.append(6 if column in self.categorical_indices_ else 5)
                operands.append(0.0)
            elif operation == '==' and column in self.categorical_indices_:
                operations.append(4)
                operands.append(float(self.category_maps_[column][proposition.constraint.value]))
            else:
                operations.append(_OPERATION_CODES[operation])
                operands.append(float(proposition.constraint.value))
        self.proposition_columns_ = np.ascontiguousarray(columns, dtype=np.int64)
        self.proposition_operations_ = np.ascontiguousarray(operations, dtype=np.int8)
        self.proposition_operands_ = np.ascontiguousarray(operands, dtype=np.float64)

    def _proposition_indices(self, proposition_indices):
        if proposition_indices is None:
            return np.arange(len(self.propositions_), dtype=np.int64)
        indices = np.asarray(proposition_indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= len(self.propositions_)):
            raise IndexError('Proposition index is outside the fitted encoder')
        return indices

    def _encode_columns(self, values, columns):
        encoded = np.zeros((len(values), self.n_features_in_), dtype=np.float64)
        for column in columns:
            if column not in self.categorical_indices_:
                encoded[:, column] = np.asarray(values[:, column], dtype=np.float64)
                continue
            mapping = self.category_maps_[int(column)]
            for row, value in enumerate(values[:, column]):
                encoded[row, column] = -1.0 if _is_missing(value) else float(mapping.get(value, -2))
        return encoded


def _is_numeric(column):
    for value in column:
        if _is_missing(value):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            return False
    return True


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return False
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
