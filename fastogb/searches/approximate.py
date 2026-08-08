"""Orthogonal beam and greedy query searches."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from math import inf

import numpy as np

from fastogb.array_ops import extent_signature
from fastogb.kernels import is_subset, mask_intersection_extent
from fastogb.logic import Conjunction, KeyValueProposition
from fastogb.parallel import numba_threads, resolve_numba_jobs
from fastogb.searches.common import accepts_depth, accepts_keyword


class OrthogonalBeamSearch:
    """Yang Algorithm 2 over encoded numeric thresholds and categorical propositions."""

    def __init__(self, ctx, obj, bnd=None, beam_width=1, max_depth=None, verbose=False, forbidden_signatures=None,
                 n_jobs=1, parallel_min_families=4, **kwargs):
        if not hasattr(obj, 'prefix_values'):
            raise TypeError('Orthogonal search requires an objective that implements prefix_values')
        if beam_width is not None and beam_width != inf and int(beam_width) < 1:
            raise ValueError('beam_width must be positive, infinite or None')
        self.ctx = ctx
        self.f = obj
        self.beam_width = None if beam_width is None or beam_width == inf else int(beam_width)
        self.max_depth = max_depth
        self.verbose = verbose
        self.n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
        if self.n_jobs < 1:
            raise ValueError('n_jobs must be a positive integer or -1')
        self.parallel_min_families = int(parallel_min_families)
        self.forbidden_signatures = set(forbidden_signatures or ())
        self.gradient_order = np.argsort(self.f.projected_gradient, kind='stable')[::-1]
        self.families, self.singletons = self._candidate_groups()

    def run(self):
        manager = ThreadPoolExecutor(max_workers=self.n_jobs) if self.n_jobs > 1 else nullcontext(None)
        with manager as executor:
            return self._run(executor)

    def _run(self, executor):
        full_mask = np.ones(self.ctx.m, dtype=bool)
        full_extent = np.arange(self.ctx.m, dtype=np.int64)
        full_value = self.f(full_extent, 0)
        beam = [(full_value, tuple(), full_mask, full_extent)]
        best = beam[0] if self._allowed(full_mask) else None
        depth = 0
        inspected = 1
        while beam and (self.max_depth is None or depth < self.max_depth):
            candidates = {}
            best_value = -inf if best is None else best[0]
            for _, intent, parent, parent_extent in beam:
                if self._prefix_bound(parent, depth + 1) <= best_value:
                    continue
                for candidate in self._refinements(intent, parent, parent_extent, depth + 1, executor):
                    inspected += 1
                    value, child_intent, child_mask, child_extent = candidate
                    signature = extent_signature(child_mask)
                    previous = candidates.get(signature)
                    if previous is None or self._prefer(candidate, previous):
                        candidates[signature] = candidate
                    if self._allowed(child_mask) and (best is None or self._prefer(candidate, best)):
                        best = candidate
                        best_value = value
            beam = self._limit_beam(list(candidates.values()))
            depth += 1
        if best is None:
            return None
        _, intent, _, extent = best
        generator = self.ctx.greedy_simplification(list(intent), extent)
        if self.verbose:
            print(f'Found orthogonal query after inspecting {inspected} candidates: {generator}')
        return Conjunction(self.ctx.attributes[index] for index in generator)

    def _prefix_bound(self, parent, depth):
        order = self.gradient_order[parent[self.gradient_order]]
        if not len(order):
            return -inf
        forward = self.f.prefix_values(order, depth)
        reverse = self.f.prefix_values(order[::-1], depth)
        return float(max(np.max(forward), np.max(reverse)))

    def _refinements(self, intent, parent, parent_extent, depth, executor):
        arguments = [(family, intent, parent, len(parent_extent), depth) for family in self.families]
        if executor is not None and len(arguments) >= self.parallel_min_families:
            batches = executor.map(self._family_refinements, arguments)
        else:
            batches = map(self._family_refinements, arguments)
        for batch in batches:
            yield from batch
        selected = set(intent)
        for index in self.singletons:
            if index in selected or self.ctx.incompatible(intent, index):
                continue
            child_mask, child_extent = mask_intersection_extent(parent, self.ctx.bit_extents[index])
            if not len(child_extent) or len(child_extent) == len(parent_extent):
                continue
            value = float(self.f(child_extent, depth))
            yield value, tuple((*intent, int(index))), child_mask, child_extent

    def _family_refinements(self, arguments):
        (indices, order), intent, parent, parent_length, depth = arguments
        filtered_order = order[parent[order]]
        if not len(filtered_order):
            return []
        values = self.f.prefix_values(filtered_order, depth)
        selected = set(intent)
        refinements = []
        for index in indices:
            if index in selected or self.ctx.incompatible(intent, index):
                continue
            child_mask, child_extent = mask_intersection_extent(parent, self.ctx.bit_extents[index])
            if not len(child_extent) or len(child_extent) == parent_length:
                continue
            value = float(values[len(child_extent) - 1])
            refinements.append((value, tuple((*intent, int(index))), child_mask, child_extent))
        return refinements

    def _candidate_groups(self):
        groups = {}
        singletons = []
        for index, proposition in enumerate(self.ctx.attributes):
            if not isinstance(proposition, KeyValueProposition) or proposition.column_index is None:
                singletons.append(index)
                continue
            operation = proposition.constraint.op
            if operation not in {'<', '<=', '>', '>='}:
                singletons.append(index)
                continue
            direction = 1 if operation in {'<', '<='} else -1
            groups.setdefault((proposition.column_index, direction), []).append(index)
        families = []
        for (column_index, direction), indices in groups.items():
            column = np.asarray(self.f.data[:, column_index], dtype=np.float64)
            finite = np.flatnonzero(np.isfinite(column))
            order = finite[np.argsort(column[finite], kind='stable')]
            if direction < 0:
                order = order[::-1]
            families.append((np.asarray(indices, dtype=np.int64), np.ascontiguousarray(order)))
        return families, np.asarray(singletons, dtype=np.int64)

    def _allowed(self, mask):
        return extent_signature(mask) not in self.forbidden_signatures

    def _limit_beam(self, candidates):
        candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return candidates if self.beam_width is None else candidates[:self.beam_width]

    @staticmethod
    def _prefer(candidate, incumbent):
        return (candidate[0], -len(candidate[1]), tuple(-index for index in candidate[1])) > (
            incumbent[0], -len(incumbent[1]), tuple(-index for index in incumbent[1])
        )


class GreedySearch:
    """Greedily add the proposition yielding the largest immediate improvement."""

    def __init__(self, ctx, obj, bnd=None, verbose=False, forbidden_signatures=None, backend='auto', n_jobs=1,
                 parallel_min_candidates=8, parallel_min_work=1_000_000, **kwargs):
        self.ctx = ctx
        self.f = obj
        self.verbose = verbose
        self.with_depth = accepts_depth(obj)
        self.forbidden_signatures = set(forbidden_signatures or ())
        self.backend = backend
        self.n_jobs = resolve_numba_jobs(n_jobs)
        self.parallel_min_candidates = int(parallel_min_candidates)
        self.parallel_min_work = int(parallel_min_work)
        self.parallel_values = hasattr(self.f, 'greedy_values') and accepts_keyword(self.f.greedy_values, 'parallel')
        if self.parallel_min_candidates < 1 or self.parallel_min_work < 1:
            raise ValueError('Parallel search thresholds must be positive')

    def _objective(self, extent, depth):
        return self.f(extent, depth) if self.with_depth else self.f(extent)

    def run(self):
        with numba_threads(self.n_jobs):
            return self._run()

    def _run(self):
        intent = []
        extent = self.ctx.extension([])
        bit_extent = np.ones(self.ctx.m, dtype=bool)
        value = self._objective(extent, 0)
        while True:
            indices = np.asarray([index for index in range(self.ctx.n)
                                  if index not in intent and not self.ctx.incompatible(intent, index)], dtype=np.int64)
            best_index, best_extent, best_bit_extent, best_value = self._best_candidate(
                bit_extent, indices, len(intent) + 1
            )
            best_index = best_index if best_value > value else None
            if best_index is None:
                break
            value = best_value
            intent.append(best_index)
            extent = best_extent
            bit_extent = best_bit_extent
        mask = np.zeros(self.ctx.m, dtype=bool)
        mask[extent] = True
        if extent_signature(mask) in self.forbidden_signatures:
            return None
        return Conjunction(self.ctx.attributes[index] for index in intent)

    def _best_candidate(self, parent, indices, depth):
        if not len(indices):
            return None, None, None, -inf
        indices = np.asarray([index for index in indices if not is_subset(parent, self.ctx.bit_extents[index])],
                             dtype=np.int64)
        if not len(indices):
            return None, None, None, -inf
        if hasattr(self.f, 'greedy_values'):
            enough_work = len(parent) * len(indices) >= self.parallel_min_work
            parallel = (self.n_jobs > 1 and len(indices) >= self.parallel_min_candidates and enough_work
                        and self.backend != 'cuda')
            arguments = (parent, self.ctx.bit_extents, indices, depth, self.backend)
            values = self.f.greedy_values(*arguments, parallel=parallel) if self.parallel_values else (
                self.f.greedy_values(*arguments)
            )
            position = int(np.argmax(values))
            index = int(indices[position])
            bit_extent = np.logical_and(parent, self.ctx.bit_extents[index])
            return index, np.flatnonzero(bit_extent), bit_extent, float(values[position])
        best = (None, None, None, -inf)
        for index in indices:
            bit_extent = np.logical_and(parent, self.ctx.bit_extents[index])
            extent = np.flatnonzero(bit_extent)
            candidate = self._objective(extent, depth)
            if candidate > best[3]:
                best = (int(index), extent, bit_extent, candidate)
        return best
