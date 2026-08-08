"""Small NumPy datasets used in examples and tests."""

from __future__ import annotations

import numpy as np


def noisy_parity(n, d=3, variance=0.25, random_seed=None):
    """Generate noisy hypercube observations labelled by vertex parity."""
    if n < 0 or d < 1 or variance < 0:
        raise ValueError('n and variance must be non-negative and d must be positive')
    rng = np.random.default_rng(random_seed)
    centres = rng.choice(np.array([-1.0, 1.0]), size=(n, d))
    target = np.prod(centres, axis=1).astype(np.int64)
    data = centres + rng.normal(scale=np.sqrt(variance), size=(n, d))
    return data, target


def alternating_block_model(a, b, k):
    """Generate alternating positive and negative one-dimensional blocks."""
    if a < 1 or b < 1 or k < 1:
        raise ValueError('a, b and k must be positive integers')
    size = (a + b) * k - b
    values = np.arange(size)
    target = np.where(values % (a + b) < a, 1, -1)
    return values.reshape(-1, 1), target
