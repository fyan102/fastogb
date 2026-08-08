"""Propositional logic for fastogb additive ensembles."""

from __future__ import annotations

import operator
from typing import Any, Callable

import numpy as np


class Constraint:
    """A Boolean condition with a readable representation."""

    def __init__(self, cond: Callable, str_repr: Callable | None = None, op: str | None = None, value: Any = None):
        self.cond = cond
        self.str_repr = str_repr or (lambda name: f'{cond}({name})')
        self.op = op
        self.value = value

    def __call__(self, value):
        return self.cond(value)

    def __format__(self, variable):
        return self.str_repr(variable)

    def __repr__(self):
        return f'Constraint({format(self, "x")})'

    @classmethod
    def _comparison(cls, op, value, function):
        return cls(lambda x: function(x, value), lambda name: f'{name}{op}{value}', op, value)

    @classmethod
    def less_equals(cls, value):
        return cls._comparison('<=', value, operator.le)

    @classmethod
    def less(cls, value):
        return cls._comparison('<', value, operator.lt)

    @classmethod
    def greater_equals(cls, value):
        return cls._comparison('>=', value, operator.ge)

    @classmethod
    def greater(cls, value):
        return cls._comparison('>', value, operator.gt)

    @classmethod
    def equals(cls, value):
        return cls._comparison('==', value, operator.eq)

    @classmethod
    def not_equals(cls, value):
        return cls._comparison('!=', value, operator.ne)

    @classmethod
    def missing(cls):
        return cls(_missing_mask, lambda name: f'{name} is missing', 'missing')


def _missing_mask(values):
    values = np.asarray(values)
    if values.dtype.kind in 'fc':
        return np.isnan(values)
    return np.frompyfunc(_is_missing, 1, 1)(values).astype(bool)


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


_operator_factory = {
    '==': Constraint.equals,
    '!=': Constraint.not_equals,
    '>': Constraint.greater,
    '<': Constraint.less,
    '>=': Constraint.greater_equals,
    '<=': Constraint.less_equals,
}


def constraint_from_op_string(op, value):
    return _operator_factory[op](value)


class KeyValueProposition:
    """A constraint on one named input column."""

    def __init__(self, key, constraint, column_index=None):
        self.key = key
        self.column_index = key if column_index is None and isinstance(key, int) else column_index
        self.constraint = constraint
        self.repr = format(constraint, key)

    def __call__(self, data):
        values = self._column(data)
        if self.constraint.op in {'<', '<=', '>', '>='}:
            values = np.asarray(values, dtype=np.float64)
        return self.constraint(values)

    def _column(self, data):
        if isinstance(data, np.ndarray):
            if self.column_index is None:
                if data.dtype.names and self.key in data.dtype.names:
                    return data[self.key]
                raise ValueError(f'Column index is unavailable for proposition {self!r}')
            return data[..., self.column_index]
        return data[self.key]

    @property
    def is_exclusive(self):
        return self.constraint.op in {'==', 'missing'}

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        return isinstance(other, KeyValueProposition) and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __le__(self, other):
        return str(self) <= str(other)


class TabulatedProposition:
    """A proposition corresponding to one column of a binary table."""

    def __init__(self, table, column_index):
        self.table = np.asarray(table)
        self.column_index = column_index
        self.key = column_index
        self.repr = f'c{column_index}'

    def __call__(self, rows):
        rows = np.asarray(rows)
        if rows.ndim >= 2:
            return rows[..., self.column_index]
        return self.table[rows, self.column_index]

    def __repr__(self):
        return self.repr


class Conjunction:
    """A conjunction of elementary propositions."""

    def __init__(self, props=()):
        self.props = sorted(props, key=str)
        self.repr = ' & '.join(map(str, self.props)) if self.props else 'True'

    def __call__(self, data):
        if not self.props:
            values = np.asarray(data)
            return True if values.ndim <= 1 else np.ones(len(values), dtype=bool)
        result = np.asarray(self.props[0](data), dtype=bool)
        for proposition in self.props[1:]:
            result = np.logical_and(result, proposition(data))
        return result

    def __repr__(self):
        return self.repr

    def __getitem__(self, item):
        return self.props[item]

    def __len__(self):
        return len(self.props)

    def __eq__(self, other):
        return isinstance(other, Conjunction) and self.props == other.props

    def __hash__(self):
        return hash(tuple(self.props))
