"""Crossbar controller: load weights, apply V, read I."""

from typing import Callable, Optional

import numpy as np


class CrossbarController:
    """FSM: load weights, run V, sample I."""

    def __init__(
        self,
        set_conductance: Callable[[np.ndarray], None],
        run: Callable[[np.ndarray], np.ndarray],
        rows: int,
        cols: int,
    ):
        self.set_conductance = set_conductance
        self.run = run
        self.rows = rows
        self.cols = cols
        self._G: Optional[np.ndarray] = None

    def load_weights(self, G: np.ndarray) -> None:
        G = np.asarray(G)
        if G.shape != (self.rows, self.cols):
            raise ValueError("G shape mismatch")
        self._G = G.copy()
        self.set_conductance(G)

    def execute(self, V: np.ndarray) -> np.ndarray:
        if self._G is None:
            raise RuntimeError("Weights not loaded")
        return self.run(V)
