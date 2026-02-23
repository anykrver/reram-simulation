"""Weight save / load helpers for crossbar conductance matrices."""

from pathlib import Path

import numpy as np


def save_weights(G: np.ndarray | list[np.ndarray], path: str) -> None:
    """
    Save conductance matrix G (or list of matrices) to a .npy file.

    Parameters
    ----------
    G    : Numpy array or list of arrays.
    path : Destination file path.
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Force object-type array for list of layers to avoid "inhomogeneous shape" errors
    if isinstance(G, list):
        G_to_save = np.array(G, dtype=object)
    else:
        G_to_save = np.asarray(G)
        
    np.save(str(path), G_to_save)


def load_weights(path: str) -> np.ndarray | list[np.ndarray]:
    """
    Load conductance matrix G from a .npy file.

    Parameters
    ----------
    path : Source file path.

    Returns
    -------
    G : Numpy float32 array or list of float32 arrays.
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    data = np.load(str(path), allow_pickle=True)
    if data.dtype == object:
        # It's a list/object array of layers
        return [layer.astype(np.float32) for layer in data]
    return data.astype(np.float32)
