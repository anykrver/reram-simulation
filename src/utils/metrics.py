"""Accuracy, energy, latency aggregation."""

from typing import List, Optional

import numpy as np


def compute_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    top_k: int = 1,
) -> float:
    """
    Classification accuracy (top-k).

    Args:
        predictions: Logits or class scores (N, C).
        targets: Integer labels (N,).
        top_k: Count prediction correct if target in top-k.

    Returns:
        Accuracy in [0, 1].
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets, dtype=np.intp)
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if targets.ndim != 1:
        targets = targets.ravel()
    N = targets.size
    if N == 0:
        return 0.0
    top = np.argsort(-predictions, axis=1)[:, :top_k]
    correct = np.any(top == targets[:, np.newaxis], axis=1)
    return float(np.mean(correct))


def aggregate_energy(
    energy_list: List[float],
    timesteps: Optional[int] = None,
) -> dict:
    """
    Aggregate energy over runs.

    Args:
        energy_list: Per-step or per-run energy (J).
        timesteps: If set, reshape as (runs, timesteps) and sum per run.

    Returns:
        Dict with total, mean, std.
    """
    arr = np.asarray(energy_list)
    total = float(np.sum(arr))
    return {
        "total_J": total,
        "mean_J": float(np.mean(arr)),
        "std_J": float(np.std(arr)) if arr.size > 1 else 0.0,
        "count": int(arr.size),
    }
