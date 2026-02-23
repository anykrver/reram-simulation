"""LIF / threshold neurons: membrane, fire, reset."""

from typing import Optional

import numpy as np


class LIFNeuron:
    """Leaky integrate-and-fire neuron."""

    def __init__(
        self,
        threshold: float = 1.0,
        leak: float = 0.99,
        reset: float = 0.0,
        dtype: type = np.float32,
    ):
        """
        Args:
            threshold: Spike threshold (V).
            leak: Membrane leak factor per timestep (V *= leak).
            reset: Membrane potential after spike.
            dtype: Compute dtype.
        """
        self.threshold = threshold
        self.leak = leak
        self.reset = reset
        self.dtype = dtype

    def step(
        self,
        current: np.ndarray,
        membrane: Optional[np.ndarray] = None,
    ):
        """
        One timestep: integrate current, leak, threshold, reset.

        Args:
            current: Input current (e.g. from crossbar), shape (n_neurons,).
            membrane: Previous membrane state; if None, zeros.

        Returns:
            spikes: Binary spikes shape (n_neurons,).
            membrane_new: New membrane state (n_neurons,).
        """
        current = np.asarray(current, dtype=self.dtype)
        n = current.size
        if membrane is None:
            membrane = np.zeros(n, dtype=self.dtype)
        else:
            membrane = np.asarray(membrane, dtype=self.dtype)
        membrane = membrane * self.leak + current
        spikes = (membrane >= self.threshold).astype(self.dtype)
        membrane = np.where(spikes > 0, self.reset, membrane)
        return spikes, membrane
