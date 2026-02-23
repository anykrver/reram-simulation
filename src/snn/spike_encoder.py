"""Spike encoding: rate (Poisson) or temporal."""

from typing import Optional

import numpy as np


class PoissonEncoder:
    """Encode continuous values to spike trains via Poisson process."""

    def __init__(
        self,
        max_rate: float = 100.0,
        timestep_ms: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            max_rate: Maximum firing rate (Hz).
            timestep_ms: Timestep duration in ms.
            seed: Random seed for reproducibility.
        """
        self.max_rate = max_rate
        self.timestep_ms = timestep_ms
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def encode(
        self,
        x: np.ndarray,
        num_timesteps: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode x (values in [0, 1]) to spikes (num_timesteps, ...).

        Args:
            x: Input values, shape (...,). Normalized to [0,1] for rate.
            num_timesteps: Number of time steps.
            seed: Override seed for this call.

        Returns:
            spikes: Binary array shape (num_timesteps,) + x.shape.
        """
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0.0, 1.0)
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        rate = x * self.max_rate * (self.timestep_ms / 1000.0)
        shape = (num_timesteps,) + x.shape
        spikes = rng.random(shape) < np.broadcast_to(rate, shape)
        return spikes.astype(np.float32)
