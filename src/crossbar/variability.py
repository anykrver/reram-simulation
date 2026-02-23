"""Conductance/weight variability (e.g. Gaussian, cycle-to-cycle)."""

from typing import Optional

import numpy as np


def add_conductance_variability(
    G: np.ndarray,
    std: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to conductance (cycle-to-cycle or device-to-device).

    Args:
        G: Nominal conductance matrix.
        std: Standard deviation (fraction or absolute, applied as additive).
        seed: Random seed for reproducibility.

    Returns:
        G_noisy: G + noise, clipped to non-negative.
    """
    G = np.asarray(G, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, G.shape)
    G_noisy = np.maximum(G + noise, 0.0)
    return G_noisy.astype(G.dtype)


def add_variability_percent(
    G: np.ndarray,
    cv: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Multiplicative variability: G * (1 + N(0, cv^2)).

    Args:
        G: Nominal conductance.
        cv: Coefficient of variation (std/mean).
        seed: Random seed.

    Returns:
        G_noisy: Non-negative conductance.
    """
    G = np.asarray(G, dtype=np.float64)
    rng = np.random.default_rng(seed)
    factor = 1.0 + rng.normal(0, cv, G.shape)
    G_noisy = np.maximum(G * factor, 0.0)
    return G_noisy.astype(G.dtype)
