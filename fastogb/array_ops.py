"""Array operations used by fastogb search implementations."""

from __future__ import annotations

import numpy as np
from numba import njit

def _intersect_kernel(left, right):
    result = np.empty(min(len(left), len(right)), dtype=np.int64)
    left_index = 0
    right_index = 0
    result_index = 0
    while left_index < len(left) and right_index < len(right):
        left_value = left[left_index]
        right_value = right[right_index]
        if left_value < right_value:
            left_index += 1
        elif right_value < left_value:
            right_index += 1
        else:
            result[result_index] = left_value
            result_index += 1
            left_index += 1
            right_index += 1
    return result[:result_index]


_compiled_intersect = njit(cache=True, nogil=True)(_intersect_kernel)


def intersect_sorted(left, right):
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    return _compiled_intersect(left, right)
def extent_signature(mask):
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder='little').tobytes()
