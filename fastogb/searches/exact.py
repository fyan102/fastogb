"""Exact branch-and-bound search over the core-query prefix tree."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from math import inf

import numpy as np

from fastogb.kernels import (full_packed_extent, packed_extent_signature, packed_intersection_extent)
from fastogb.logic import Conjunction
from fastogb.parallel import numba_threads, resolve_numba_jobs
from fastogb.searches.boundaries import Node, make_boundary
from fastogb.searches.common import accepts_depth


class CoreQueryTreeSearch:
    """Branch-and-bound search over the prefix tree of core queries."""

    traversal_orders = {'breadthfirst', 'bestboundfirst', 'bestvaluefirst', 'depthfirst'}

    def __init__(self, ctx, obj, bnd, order='bestboundfirst', apx=1.0, max_depth=10, verbose=False, n_jobs=1,
                 parallel_min_candidates=8, parallel_min_work=1_000_000, forbidden_signatures=None, backend='auto',
                 **kwargs):
        if order not in self.traversal_orders:
            raise ValueError(f'Unknown traversal order {order!r}')
        if apx <= 0:
            raise ValueError('Approximation factor must be positive')
        if backend == 'cuda':
            raise ValueError('CUDA batch evaluation is currently supported by greedy search only')
        self.ctx = ctx
        self.f = obj
        self.g = bnd
        self.order = order
        self.apx = apx
        self.max_depth = max_depth
        self.verbose = verbose
        self.n_jobs = resolve_numba_jobs(n_jobs)
        self.parallel_min_candidates = int(parallel_min_candidates)
        self.parallel_min_work = int(parallel_min_work)
        if self.parallel_min_candidates < 1 or self.parallel_min_work < 1:
            raise ValueError('Parallel search thresholds must be positive')
        self.forbidden_signatures = set(forbidden_signatures or ())
        self.with_depth = accepts_depth(obj)
        self._reset_stats()

    def _reset_stats(self):
        self.popped = 0
        self.created = 0
        self.avg_created_length = 0.0
        self.pruned = 0

    def _objective(self, extent, depth):
        return self.f(extent, depth) if self.with_depth else self.f(extent)

    def _bound(self, extent, depth):
        return self.g(extent, depth) if self.with_depth else self.g(extent)

    def traversal(self):
        use_executor = self.n_jobs > 1 and not hasattr(self.f, 'exact_batch')
        manager = ThreadPoolExecutor(max_workers=self.n_jobs) if use_executor else nullcontext(None)
        thread_manager = numba_threads(self.n_jobs) if hasattr(self.f, 'exact_batch') else nullcontext()
        with thread_manager, manager as executor:
            yield from self._traversal(executor)

    def _traversal(self, executor):
        self._reset_stats()
        boundary = make_boundary(self.order)
        full = self.ctx.extension([])
        packed_full = full_packed_extent(self.ctx.m)
        root = Node([], np.zeros(self.ctx.n, dtype=bool), full, packed_full, -1, self.ctx.n,
                    self._objective(full, 0), inf, self._allowed(packed_full))
        best = root if root.allowed else None
        yield root
        boundary.push(([(index, self.ctx.n, inf) for index in range(self.ctx.n)], root))
        self.created = 1

        while boundary:
            operations, current = boundary.pop()
            self.popped += 1
            best_value = -inf if best is None else best.val
            candidates = []
            for augmentation, inherited_crit, inherited_bound in operations:
                if not self._skip(current, augmentation, inherited_crit, inherited_bound, best_value):
                    candidates.append((current, augmentation, inherited_crit))
            if hasattr(self.f, 'exact_batch'):
                evaluations = self._evaluate_batch(candidates)
            elif executor is not None and len(candidates) >= self.parallel_min_candidates:
                evaluations = executor.map(self._evaluate, candidates)
            else:
                evaluations = map(self._evaluate, candidates)
            children = []
            for augmentation, inherited_crit, extension, packed_extension, generator, value, bound in evaluations:
                best_value = -inf if best is None else best.val
                if bound * self.apx < best_value and value <= best_value:
                    self.pruned += 1
                    continue
                child = self._make_child(current, generator, extension, packed_extension, augmentation,
                                         inherited_crit, value, bound)
                if child.allowed and (best is None or child.val > best.val):
                    best = child
                children.append(child)
                yield child
                if best is not None and best.val >= self.apx * current.val_bound and self.order == 'bestboundfirst':
                    return
            best_value = -inf if best is None else best.val
            active = [(child.gen_index, child.crit_idx, child.val_bound) for child in children
                      if child.val_bound * self.apx > best_value]
            for child in children:
                if child.valid and (not self.max_depth or len(child.generator) < self.max_depth):
                    boundary.push((active, child))

    def _allowed(self, packed_extension):
        return packed_extent_signature(packed_extension, self.ctx.m) not in self.forbidden_signatures

    def _skip(self, current, augmentation, inherited_crit, inherited_bound, best_value):
        return (augmentation <= current.gen_index or inherited_crit < current.gen_index
                or inherited_bound * self.apx <= best_value or current.closure[augmentation]
                or self.ctx.incompatible(current.generator, augmentation))

    def _evaluate(self, candidate):
        current, augmentation, inherited_crit = candidate
        packed_extension, extension = packed_intersection_extent(
            current.packed_extension, self.ctx.packed_extents[augmentation], self.ctx.m
        )
        generator = [*current.generator, augmentation]
        value = self._objective(extension, len(generator))
        bound = self._bound(extension, len(generator))
        return augmentation, inherited_crit, extension, packed_extension, generator, value, bound

    def _evaluate_batch(self, candidates):
        if not candidates:
            return ()
        current = candidates[0][0]
        indices = np.asarray([candidate[1] for candidate in candidates], dtype=np.int64)
        depth = len(current.generator) + 1
        enough_work = self.ctx.m * len(candidates) >= self.parallel_min_work
        parallel = self.n_jobs > 1 and len(candidates) >= self.parallel_min_candidates and enough_work
        children, values, bounds = self.f.exact_batch(
            current.packed_extension, self.ctx.packed_extents, indices, depth, parallel
        )
        evaluations = []
        for output, (_, augmentation, inherited_crit) in enumerate(candidates):
            packed_extension = children[output]
            generator = [*current.generator, augmentation]
            evaluations.append((augmentation, inherited_crit, None, packed_extension, generator,
                                float(values[output]), float(bounds[output])))
        return evaluations

    def _make_child(self, current, generator, extension, packed_extension, augmentation, inherited_crit, value, bound):
        self.created += 1
        self.avg_created_length += (len(generator) - self.avg_created_length) / self.created
        closure = current.closure.copy()
        closure[augmentation] = True
        if inherited_crit < augmentation and not current.closure[inherited_crit]:
            crit_index = inherited_crit
        else:
            crit_index = self.ctx.find_small_crit_index(augmentation, packed_extension, closure)
        if crit_index > augmentation:
            crit_index = self.ctx.complete_closure(augmentation, packed_extension, closure)
        else:
            closure[crit_index] = True
        allowed = self._allowed(packed_extension)
        return Node(generator, closure, extension, packed_extension, augmentation, crit_index, value, bound, allowed,
                    self.ctx.m)

    def run(self):
        optimum = None
        inspected = 0
        for node in self.traversal():
            inspected += 1
            if node.allowed and (optimum is None or node.val > optimum.val):
                optimum = node
        if optimum is None:
            return None
        if not optimum.valid:
            self.ctx.complete_closure(optimum.gen_index, optimum.packed_extension, optimum.closure)
        closure = np.flatnonzero(optimum.closure).tolist()
        generator = self.ctx.greedy_simplification(closure, optimum.extension)
        if self.verbose:
            print(f'Found optimum after inspecting {inspected} nodes: {generator}')
        return Conjunction(self.ctx.attributes[index] for index in generator)
