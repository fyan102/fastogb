"""Mandatory Numba CPU kernels and optional CUDA kernels for fastogb."""

from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, prange

try:
    from numba import cuda
except (ImportError, SystemError):
    cuda = None


def _candidate_statistics_kernel(parent, attributes, indices, values):
    supports = np.zeros(len(indices), dtype=np.int64)
    sums = np.zeros(len(indices), dtype=np.float64)
    for output_index in prange(len(indices)):
        attribute = indices[output_index]
        support = 0
        total = 0.0
        for row in range(len(parent)):
            if parent[row] and attributes[attribute, row]:
                support += 1
                total += values[row]
        supports[output_index] = support
        sums[output_index] = total
    return supports, sums


_compiled_statistics = njit(cache=True, nogil=True)(_candidate_statistics_kernel)
_compiled_parallel_statistics = njit(cache=True, nogil=True, parallel=True)(_candidate_statistics_kernel)

if cuda is not None:
    @cuda.jit
    def _cuda_statistics_kernel(parent, attributes, indices, values, supports, sums):
        output_index = cuda.blockIdx.x
        if output_index >= len(indices):
            return
        attribute = indices[output_index]
        thread = cuda.threadIdx.x
        support = cuda.shared.array(256, dtype=int64)
        total = cuda.shared.array(256, dtype=float64)
        local_support = 0
        local_total = 0.0
        for row in range(thread, len(parent), cuda.blockDim.x):
            if parent[row] and attributes[attribute, row]:
                local_support += 1
                local_total += values[row]
        support[thread] = local_support
        total[thread] = local_total
        cuda.syncthreads()
        width = cuda.blockDim.x // 2
        while width:
            if thread < width:
                support[thread] += support[thread + width]
                total[thread] += total[thread + width]
            cuda.syncthreads()
            width //= 2
        if thread == 0:
            supports[output_index] = support[0]
            sums[output_index] = total[0]


def candidate_statistics(parent, attributes, indices, values, backend='auto'):
    """Compute support and value sums for candidate intersections."""
    parent, attributes, indices, values = _normalise_inputs(parent, attributes, indices, values)
    if backend == 'cuda':
        return CudaCandidateStatistics(attributes, values).evaluate(parent, indices)
    if backend not in {'auto', 'numba'}:
        raise ValueError(f'Unknown acceleration backend {backend!r}')
    work = len(parent) * len(indices)
    kernel = _compiled_parallel_statistics if work >= 100_000 and len(indices) >= 8 else _compiled_statistics
    return kernel(parent, attributes, indices, values)


class CudaCandidateStatistics:
    """Keep invariant candidate data on an NVIDIA GPU across repeated evaluations."""

    def __init__(self, attributes, values):
        if not cuda_available():
            raise RuntimeError('CUDA is unavailable; an NVIDIA GPU and the CUDA-enabled Numba extra are required')
        attributes = np.ascontiguousarray(attributes, dtype=np.bool_)
        values = np.ascontiguousarray(values, dtype=np.float64)
        if attributes.ndim != 2 or attributes.shape[1] != len(values):
            raise ValueError('Attributes must have shape (number of candidates, number of samples)')
        self.attributes = cuda.to_device(attributes)
        self.values = cuda.to_device(values)
        self.n_samples = len(values)

    def evaluate(self, parent, indices):
        parent = np.ascontiguousarray(parent, dtype=np.bool_)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        if parent.shape != (self.n_samples,):
            raise ValueError(f'Expected a parent mask of length {self.n_samples}')
        device_parent = cuda.to_device(parent)
        device_indices = cuda.to_device(indices)
        supports = cuda.device_array(len(indices), dtype=np.int64)
        sums = cuda.device_array(len(indices), dtype=np.float64)
        threads = 256
        if len(indices):
            _cuda_statistics_kernel[len(indices), threads](device_parent, self.attributes, device_indices,
                                                           self.values, supports, sums)
        return supports.copy_to_host(), sums.copy_to_host()


def cuda_available():
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def numba_available():
    return True


def _normalise_inputs(parent, attributes, indices, values):
    parent = np.ascontiguousarray(parent, dtype=np.bool_)
    attributes = np.ascontiguousarray(attributes, dtype=np.bool_)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    if parent.ndim != 1 or values.shape != parent.shape:
        raise ValueError('Parent and values must be one-dimensional arrays with matching lengths')
    if attributes.ndim != 2 or attributes.shape[1] != len(parent):
        raise ValueError('Attributes must have shape (number of candidates, number of samples)')
    if np.any(indices < 0) or np.any(indices >= len(attributes)):
        raise IndexError('Candidate index is outside the attribute matrix')
    return parent, attributes, indices, values
