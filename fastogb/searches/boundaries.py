"""Boundary containers and nodes used by exact core-query search."""

from collections import deque
from heapq import heappop, heappush
from itertools import count

import numpy as np

from fastogb.kernels import packed_to_extent


class Node:
    """A node and incoming edge in a core-query search tree."""

    def __init__(self, generator, closure, extension, packed_extension, gen_index, crit_index, value, bound,
                 allowed=True, row_count=None):
        self.generator = generator
        self.closure = closure
        self._extension = extension
        self.packed_extension = packed_extension
        self.row_count = row_count
        self.gen_index = gen_index
        self.crit_idx = crit_index
        self.val = value
        self.val_bound = bound
        self.valid = crit_index > gen_index
        self.allowed = allowed

    @property
    def extension(self):
        if self._extension is None:
            self._extension = packed_to_extent(self.packed_extension, self.row_count)
        return self._extension

    def __repr__(self):
        closure = np.flatnonzero(self.closure)
        return f'N({self.generator}, {closure}, {self.val:.5g}, {self.val_bound:.5g}, {self.extension})'


class QueueBoundary:
    def __init__(self, lifo=False):
        self.items = deque()
        self.lifo = lifo

    def __bool__(self):
        return bool(self.items)

    def __len__(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop() if self.lifo else self.items.popleft()


class PriorityBoundary:
    def __init__(self, priority):
        self.heap = []
        self.priority = priority
        self.sequence = count()

    def __bool__(self):
        return bool(self.heap)

    def __len__(self):
        return len(self.heap)

    def push(self, item):
        _, node = item
        heappush(self.heap, (*self.priority(node), next(self.sequence), item))

    def pop(self):
        return heappop(self.heap)[-1]


def make_boundary(order):
    if order == 'breadthfirst':
        return QueueBoundary()
    if order == 'depthfirst':
        return QueueBoundary(lifo=True)
    priority = (lambda node: (-node.val_bound, -node.val)) if order == 'bestboundfirst' else (
        lambda node: (-node.val, -node.val_bound)
    )
    return PriorityBoundary(priority)
