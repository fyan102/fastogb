"""Numerically stable linear algebra for rule bases."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr

from fastogb.kernels import orthogonal_extent_norm


class OrthogonalBasis:
    """Incrementally maintain an orthonormal basis with dependence detection."""

    def __init__(self, size, rtol=None):
        self.size = int(size)
        self.rtol = rtol
        self.values = np.empty((self.size, 0), dtype=np.float64)

    def append(self, vector):
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.shape != (self.size,):
            raise ValueError(f'Expected vector shape {(self.size,)}, received {vector.shape}')
        if not np.all(np.isfinite(vector)):
            raise ValueError('Basis candidate contains NaN or infinite values')
        scale = np.linalg.norm(vector)
        tolerance = self._tolerance(scale)
        if scale <= tolerance:
            return False
        residual = project_away(self.values, vector)
        residual = project_away(self.values, residual)
        residual_norm = np.linalg.norm(residual)
        if residual_norm <= tolerance:
            return False
        self.values = np.column_stack((self.values, residual / residual_norm))
        return True

    def rebuild(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.size:
            raise ValueError(f'Expected a matrix with {self.size} rows, received shape {matrix.shape}')
        if matrix.shape[1] == 0:
            self.values = np.empty((self.size, 0), dtype=np.float64)
            return np.empty(0, dtype=np.int64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError('Basis matrix contains NaN or infinite values')
        basis, triangular, pivots = qr(matrix, mode='economic', pivoting=True)
        diagonal = np.abs(np.diag(triangular))
        scale = diagonal[0] if len(diagonal) else 0.0
        rank = int(np.count_nonzero(diagonal > self._tolerance(scale)))
        self.values = basis[:, :rank]
        return np.sort(np.asarray(pivots[:rank], dtype=np.int64))

    def _tolerance(self, scale):
        if self.rtol is not None:
            return float(self.rtol) * max(float(scale), 1.0)
        return np.sqrt(32 * np.finfo(np.float64).eps) * max(float(scale), 1.0)


def project_away(basis, vector):
    basis = np.asarray(basis, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != len(vector):
        raise ValueError('Basis row count must match the vector length')
    return vector.copy() if basis.shape[1] == 0 else vector - basis @ (basis.T @ vector)


def orthonormalize(matrix, rtol=None):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f'Expected a two-dimensional matrix, received shape {matrix.shape}')
    basis = OrthogonalBasis(matrix.shape[0], rtol=rtol)
    basis.rebuild(matrix)
    return basis.values


def orthogonal_binary_norm(basis, extent, regularisation=0.0):
    return orthogonal_extent_norm(basis, extent, regularisation)


def orthogonal_binary_prefix_norms(basis, extent, regularisation=0.0):
    basis = np.asarray(basis, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.int64)
    lengths = np.arange(1, len(extent) + 1, dtype=np.float64)
    if basis.shape[1] == 0:
        return np.sqrt(lengths + regularisation)
    projections = np.cumsum(basis[extent], axis=0)
    projection_norms = np.einsum('ij,ij->i', projections, projections)
    return np.sqrt(_safe_residual_squared(lengths, projection_norms) + regularisation)


def _safe_residual_squared(total_norm, projection_norm):
    total_norm = np.asarray(total_norm, dtype=np.float64)
    projection_norm = np.asarray(projection_norm, dtype=np.float64)
    residual = total_norm - projection_norm
    scale = np.maximum(np.maximum(np.abs(total_norm), np.abs(projection_norm)), 1.0)
    tolerance = np.sqrt(32 * np.finfo(np.float64).eps) * scale
    if np.any(residual < -tolerance):
        minimum = float(np.min(residual))
        raise FloatingPointError(f'Orthogonal projection exceeds vector norm by {-minimum:g}')
    return np.where(residual <= tolerance, 0.0, residual)
