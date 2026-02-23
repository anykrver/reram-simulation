"""IR drop model: voltage drop along rows and columns (resistive interconnect)."""

import numpy as np


def apply_ir_drop(
    V: np.ndarray,
    G: np.ndarray,
    row_resistance: float = 0.1,
    col_resistance: float = 0.1,
    num_steps: int = 5,
) -> np.ndarray:
    """
    Apply iterative IR drop to get effective voltages at each cell.

    Simplified model: voltage at (i,j) reduced by current flowing through
    row/column resistance. Returns effective V matrix same shape as implied by V, G.

    Args:
        V: Applied voltages (rows,) or (rows,).
        G: Conductance matrix (rows, cols).
        row_resistance: Resistance per row segment (Ohms).
        col_resistance: Resistance per column segment (Ohms).
        num_steps: Iterations for convergence.

    Returns:
        V_eff: Effective voltages (rows, cols) for energy/current computation.
    """
    V = np.asarray(V, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, np.newaxis]
    rows, cols = G.shape
    V_eff = np.broadcast_to(V, (rows, cols)).copy()
    for _ in range(num_steps):
        I_col = (V_eff * G).sum(axis=0)
        I_row = (V_eff * G).sum(axis=1)
        drop_col = np.cumsum(np.insert(I_col * col_resistance, 0, 0))[:-1]
        drop_row = np.cumsum(np.insert(I_row * row_resistance, 0, 0))[:-1]
        V_eff = np.maximum(
            V - drop_row[:, np.newaxis] - drop_col[np.newaxis, :],
            0.0,
        )
    return V_eff.astype(G.dtype)
