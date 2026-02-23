"""Weight and activation quantization (e.g. int8 / fixed-point)."""

from typing import Optional

import numpy as np


def quantize_weights(
    W: np.ndarray,
    bits: int = 8,
    symmetric: bool = True,
) -> np.ndarray:
    """
    Quantize weights to signed fixed-point.

    Args:
        W: Full-precision weights.
        bits: Number of bits (excluding sign if symmetric).
        symmetric: If True, range is [-2^(bits-1), 2^(bits-1)-1].

    Returns:
        Quantized float representation (dequantized for use in crossbar).
    """
    W = np.asarray(W)
    n_levels = 2 ** bits
    if symmetric:
        scale = np.abs(W).max()
        if scale == 0:
            return W
        scale = scale / (n_levels // 2 - 1)
        q = np.round(W / scale).astype(np.int32)
        q = np.clip(q, -n_levels // 2, n_levels // 2 - 1)
        return (q * scale).astype(W.dtype)
    else:
        w_min, w_max = W.min(), W.max()
        scale = (w_max - w_min) / (n_levels - 1) if w_max > w_min else 1.0
        q = np.round((W - w_min) / scale).astype(np.int32)
        q = np.clip(q, 0, n_levels - 1)
        return (q * scale + w_min).astype(W.dtype)


def quantize_activations(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantize activations (e.g. voltages) to unsigned fixed-point [0, 2^bits - 1].

    Args:
        x: Activations (non-negative for ReRAM).
        bits: Number of bits.

    Returns:
        Dequantized float in same range.
    """
    x = np.asarray(x)
    x = np.maximum(x, 0.0)
    n_levels = 2 ** bits
    scale = x.max() / (n_levels - 1) if x.max() > 0 else 1.0
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, 0, n_levels - 1)
    return (q * scale).astype(x.dtype)
