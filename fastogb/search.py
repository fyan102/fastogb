"""Search contexts and query-search algorithms."""

from itertools import combinations

import numpy as np

from fastogb.array_ops import intersect_sorted
from fastogb.encoding import PropositionEncoder, as_2d_array
from fastogb.kernels import complete_packed_closure, find_small_packed_critical_index, pack_masks
from fastogb.logic import KeyValueProposition, TabulatedProposition
from fastogb.searches import CoreQueryTreeSearch, GreedySearch, OrthogonalBeamSearch
from fastogb.searches.common import accepts_depth


class Context:
    """A binary relation between samples and elementary propositions."""

    @classmethod
    def from_tab(cls, table, sort_attributes=False):
        matrix = np.asarray(table, dtype=bool)
        if matrix.ndim != 2:
            raise ValueError(f'Expected a two-dimensional binary table, received shape {matrix.shape}')
        attributes = [TabulatedProposition(matrix, index) for index in range(matrix.shape[1])]
        return cls.from_binary(matrix, attributes, np.arange(matrix.shape[0]), sort_attributes)

    @classmethod
    def from_data(cls, data, without=None, max_col_attr=10, sort_attributes=True, categorical=None,
                  feature_names=None, max_categories=None, min_category_count=1, include_missing=True, **kwargs):
        encoder = PropositionEncoder(max_col_attr=max_col_attr, categorical=categorical, feature_names=feature_names,
                                     max_categories=max_categories, min_category_count=min_category_count,
                                     include_missing=include_missing)
        matrix = encoder.fit_transform(data, without=without)
        return cls.from_binary(matrix, encoder.propositions_, as_2d_array(data), sort_attributes, encoder)

    @classmethod
    def from_binary(cls, matrix, attributes=None, objects=None, sort_attributes=True, encoder=None):
        matrix = np.asarray(matrix, dtype=bool)
        if matrix.ndim != 2:
            raise ValueError(f'Expected a two-dimensional binary matrix, received shape {matrix.shape}')
        context = cls.__new__(cls)
        context._initialise(matrix, attributes, objects, sort_attributes, encoder)
        return context

    def __init__(self, attributes, objects, sort_attributes=True):
        objects = list(objects)
        matrix = np.empty((len(objects), len(attributes)), dtype=bool)
        for column, attribute in enumerate(attributes):
            matrix[:, column] = [bool(attribute(obj)) for obj in objects]
        self._initialise(matrix, attributes, objects, sort_attributes, None)

    def _initialise(self, matrix, attributes, objects, sort_attributes, encoder):
        self.matrix = np.ascontiguousarray(matrix, dtype=bool)
        self.m, self.n = self.matrix.shape
        self.attributes = list(attributes or [TabulatedProposition(self.matrix, i) for i in range(self.n)])
        if len(self.attributes) != self.n:
            raise ValueError(f'Expected {self.n} attributes, received {len(self.attributes)}')
        self.objects = np.arange(self.m) if objects is None else objects
        self.encoder = encoder
        if sort_attributes and self.n:
            order = np.argsort(self.matrix.sum(axis=0), kind='stable')
            self.matrix = np.ascontiguousarray(self.matrix[:, order])
            self.attributes = [self.attributes[index] for index in order]
        self.extents = [np.flatnonzero(self.matrix[:, index]).astype(np.int64) for index in range(self.n)]
        self.bit_extents = np.ascontiguousarray(self.matrix.T)
        self.packed_extents = pack_masks(self.bit_extents)

    def extension(self, intent):
        if not intent:
            return np.arange(self.m, dtype=np.int64)
        result = self.extents[intent[0]]
        for index in intent[1:]:
            result = intersect_sorted(result, self.extents[index])
        return result

    def greedy_simplification(self, intent, extent):
        uncovered = set(range(self.m)).difference(map(int, extent))
        available = list(range(len(intent)))
        covering = [set(range(self.m)).difference(map(int, self.extents[index])) for index in intent]
        result = []
        while uncovered and available:
            best = max(available, key=lambda index: len(covering[index].intersection(uncovered)))
            newly_covered = covering[best].intersection(uncovered)
            if not newly_covered:
                break
            result.append(intent[best])
            uncovered.difference_update(newly_covered)
            available.remove(best)
        return result

    def find_small_crit_index(self, gen_index, packed_extension, closure):
        return find_small_packed_critical_index(gen_index, packed_extension, closure, self.packed_extents)

    def complete_closure(self, gen_index, packed_extension, closure):
        return complete_packed_closure(gen_index, packed_extension, closure, self.packed_extents)

    def incompatible(self, generator, augmentation):
        proposition = self.attributes[augmentation]
        if not isinstance(proposition, KeyValueProposition) or not proposition.is_exclusive:
            return False
        for index in generator:
            other = self.attributes[index]
            if (isinstance(other, KeyValueProposition) and other.is_exclusive
                    and other.column_index == proposition.column_index and other != proposition):
                return True
        return False

def validate_bound(context, objective, bound, max_attributes=16, atol=1e-12):
    """Exhaustively validate a search bound on a small context."""
    if context.n > max_attributes:
        raise ValueError(f'Bound validation is limited to {max_attributes} attributes, received {context.n}')
    objective_with_depth = accepts_depth(objective)
    bound_with_depth = accepts_depth(bound)
    intents = [intent for depth in range(context.n + 1) for intent in combinations(range(context.n), depth)]
    extents = {intent: context.extension(list(intent)) for intent in intents}
    values = {intent: objective(extent, len(intent)) if objective_with_depth else objective(extent)
              for intent, extent in extents.items()}
    for parent in intents:
        parent_set = set(parent)
        parent_bound = bound(extents[parent], len(parent)) if bound_with_depth else bound(extents[parent])
        for child, value in values.items():
            if parent_set.issubset(child) and value > parent_bound + atol:
                raise AssertionError(
                    f'Invalid bound for intent {parent}: {parent_bound} is below {value} at descendant {child}'
                )
    return True


search_methods = {'exhaustive': CoreQueryTreeSearch, 'greedy': GreedySearch, 'ogb': OrthogonalBeamSearch}

__all__ = ['Context', 'CoreQueryTreeSearch', 'GreedySearch', 'OrthogonalBeamSearch', 'search_methods',
           'validate_bound']
