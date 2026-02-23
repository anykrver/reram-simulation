"""Heatmaps, firing raster, power curves (for dashboard/notebooks)."""

from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def plot_heatmap(
    data: np.ndarray,
    title: str = "Heatmap",
    xlabel: str = "Columns",
    ylabel: str = "Rows",
    figsize: tuple = (6, 5),
):
    """
    Plot 2D heatmap (e.g. conductance matrix).

    Args:
        data: 2D array.
        title: Plot title.
        xlabel: X axis label.
        ylabel: Y axis label.
        figsize: Figure size.

    Returns:
        matplotlib Figure or None if matplotlib not available.
    """
    if not _HAS_MPL:
        return None
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


def plot_firing_raster(
    spikes: np.ndarray,
    title: str = "Firing raster",
    xlabel: str = "Time",
    ylabel: str = "Neuron",
    figsize: tuple = (8, 4),
):
    """
    Raster plot: time vs neuron index, dot where spike.

    Args:
        spikes: (timesteps, neurons) binary.
        title: Plot title.
        xlabel: X axis label.
        ylabel: Y axis label.
        figsize: Figure size.

    Returns:
        matplotlib Figure or None.
    """
    if not _HAS_MPL:
        return None
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        spikes = spikes.reshape(-1, spikes.size)
    fig, ax = plt.subplots(figsize=figsize)
    t_axis = np.arange(spikes.shape[0])
    n_axis = np.arange(spikes.shape[1])
    t_grid, n_grid = np.meshgrid(t_axis, n_axis, indexing="ij")
    mask = spikes > 0
    ax.scatter(t_grid[mask], n_grid[mask], c="k", s=1, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


def plot_power_curve(
    time_ms: np.ndarray,
    power_mW: np.ndarray,
    title: str = "Power",
    figsize: tuple = (6, 3),
):
    """
    Power vs time curve.

    Args:
        time_ms: Time points (ms).
        power_mW: Power (mW).
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure or None.
    """
    if not _HAS_MPL:
        return None
    time_ms = np.asarray(time_ms).ravel()
    power_mW = np.asarray(power_mW).ravel()
    if power_mW.size != time_ms.size:
        power_mW = np.resize(power_mW, time_ms.size)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_ms, power_mW)
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Power (mW)")
    plt.tight_layout()
    return fig
