"""Non-ideal ReRAM crossbar: ideal + IR drop, quantization, variability."""

from typing import Optional

import numpy as np

from .ideal_crossbar import IdealCrossbar
from .ir_drop_model import apply_ir_drop
from .quantization import quantize_weights, quantize_activations
from .variability import add_conductance_variability


class NonIdealCrossbar:
    """
    Crossbar with optional noise, IR drop, variability, quantization.
    Wraps ideal I = V @ G and applies non-ideal effects.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        noise_std: float = 0.0,
        ir_drop_row_r: float = 0.0,
        ir_drop_col_r: float = 0.0,
        variability_std: float = 0.0,
        quantize_bits: Optional[int] = None,
        seed: Optional[int] = None,
        dtype: type = np.float32,
    ):
        """
        Args:
            rows: Wordlines.
            cols: Bitlines.
            noise_std: Additive conductance noise (0 = off).
            ir_drop_row_r: Row resistance for IR drop (0 = off).
            ir_drop_col_r: Column resistance for IR drop (0 = off).
            variability_std: Conductance variability (0 = off).
            quantize_bits: Weight/activation bits (None = off).
            seed: RNG seed.
            dtype: Compute dtype.
        """
        self.rows = rows
        self.cols = cols
        self.noise_std = noise_std
        self.ir_drop_row_r = ir_drop_row_r
        self.ir_drop_col_r = ir_drop_col_r
        self.variability_std = variability_std
        self.quantize_bits = quantize_bits
        self.seed = seed
        self.dtype = dtype
        self._ideal = IdealCrossbar(rows, cols, dtype)
        self._G_nominal = None

    def set_conductance(self, G: np.ndarray) -> None:
        """Set nominal conductance; apply variability and optional quantization."""
        G = np.asarray(G, dtype=self.dtype)
        if G.shape != (self.rows, self.cols):
            raise ValueError(f"G shape {G.shape} != ({self.rows}, {self.cols})")
        self._G_nominal = np.maximum(G, 0.0).copy()
        G_eff = self._G_nominal.copy()
        if self.variability_std > 0:
            G_eff = add_conductance_variability(
                G_eff, std=self.variability_std, seed=self.seed
            )
        if self.quantize_bits is not None:
            G_eff = quantize_weights(G_eff, bits=self.quantize_bits)
        self._ideal.set_conductance(G_eff)

    def run(self, V: np.ndarray) -> np.ndarray:
        """
        Run with optional IR drop and activation quantization.
        Currents computed from effective V and stored G.
        """
        if self._G_nominal is None:
            raise RuntimeError("Conductance not set.")
        V = np.asarray(V, dtype=self.dtype)
        G = self._ideal._G
        if self.quantize_bits is not None:
            V = quantize_activations(V, bits=self.quantize_bits)
        if self.ir_drop_row_r > 0 or self.ir_drop_col_r > 0:
            V_1d = V.ravel()
            V_2d = V_1d.reshape(-1, 1) if V.ndim == 1 else V
            V_eff = apply_ir_drop(
                V_2d,
                G,
                row_resistance=self.ir_drop_row_r,
                col_resistance=self.ir_drop_col_r,
            )
            I = (V_eff * G).sum(axis=0)
            out = I.astype(self.dtype)
            return out if V.ndim == 1 else out.reshape(1, -1)
        if self.noise_std > 0:
            rng = np.random.default_rng(self.seed)
            G_noisy = np.maximum(G + rng.normal(0, self.noise_std, G.shape), 0.0)
            self._ideal.set_conductance(G_noisy)
        I = self._ideal.run(V)
        return I
