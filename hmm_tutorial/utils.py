from __future__ import annotations

import numpy as np


def random_convex_combination(n: int) -> np.ndarray:
    """Return n coefficients which sum to one"""
    coefs = np.zeros(shape=n)
    coefs[0] = np.random.uniform(low=0.0, high=1.0)
    for i in range(1, n - 1):
        high = 1 - coefs.sum()
        coefs[i] = np.random.uniform(low=0, high=high)
    coefs[-1] = 1 - coefs.sum()
    np.random.shuffle(coefs)
    return coefs


def random_transition_matrix(n_states: int) -> np.ndarray:
    """Return a matrix where every row sums to one"""
    tm = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        tm[i, :] = random_convex_combination(n_states)
    return tm
