"""Energy: E = V^2 * G * t."""

from typing import Optional
import numpy as np


class EnergyEstimator:
    def __init__(self, voltage: float = 1.0, timestep_s: float = 1e-6):
        self.voltage = voltage
        self.timestep_s = timestep_s

    def energy_crossbar(self, V: np.ndarray, G: np.ndarray, t_s: Optional[float] = None) -> float:
        V = np.asarray(V, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        t = t_s if t_s is not None else self.timestep_s
        if V.ndim == 1:
            V = V[:, np.newaxis]
        V_eff = np.broadcast_to(V, G.shape)
        return float(np.sum(V_eff ** 2 * G) * t)

    def latency_cycles(self, rows: int, cols: int, cycles_per_op: int = 1) -> int:
        return rows * cols * cycles_per_op
