"""Weight save / load helpers for crossbar conductance matrices."""

from pathlib import Path

import numpy as np


def save_weights(G: np.ndarray, path: str) -> None:
    """
    Save conductance matrix G to a .npy file.

    Parameters
    ----------
    G    : Numpy array (any shape).
    path : Destination file path (`.npy` extension added if missing).
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), G)


def load_weights(path: str) -> np.ndarray:
    """
    Load conductance matrix G from a .npy file.

    Parameters
    ----------
    path : Source file path.

    Returns
    -------
    G : Numpy float32 array.
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    return np.load(str(path)).astype(np.float32)
