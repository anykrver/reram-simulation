"""Ideal ReRAM crossbar: I = V × G (Ohm's law, matrix-vector)."""

import numpy as np


class IdealCrossbar:
    """Pure matrix multiplication: currents I = V @ G (voltages × conductance)."""

    def __init__(self, rows: int, cols: int, dtype: type = np.float32):
        """
        Args:
            rows: Number of wordlines (voltage rows).
            cols: Number of bitlines (current columns).
            dtype: NumPy dtype for computation.
        """
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self._G = None

    def set_conductance(self, G: np.ndarray) -> None:
        """Set conductance matrix (rows x cols). Clipped to non-negative."""
        G = np.asarray(G, dtype=self.dtype)
        if G.shape != (self.rows, self.cols):
            raise ValueError(f"G shape {G.shape} != ({self.rows}, {self.cols})")
        self._G = np.maximum(G, 0.0)

    def run(self, V: np.ndarray) -> np.ndarray:
        """
        Compute currents I = V @ G (matrix-vector or batch of vectors).

        Args:
            V: Voltages, shape (rows,) or (batch, rows).

        Returns:
            I: Currents, shape (cols,) or (batch, cols).
        """
        if self._G is None:
            raise RuntimeError("Conductance not set; call set_conductance first.")
        V = np.asarray(V, dtype=self.dtype)
        if V.ndim == 1:
            V = V.reshape(1, -1)
        if V.shape[1] != self.rows:
            raise ValueError(f"V cols {V.shape[1]} != crossbar rows {self.rows}")
        I = V @ self._G
        return I.squeeze(0) if I.shape[0] == 1 else I
