import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(x: Sequence[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_learning_curve(
    rewards: Sequence[float],
    title: str,
    out_path: str | os.PathLike,
    window: int = 100,
    xlabel: str = "Episode",
    ylabel: str = "Return",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rewards, alpha=0.25, label="raw")
    ma = _moving_average(rewards, window)
    if len(ma) > 0:
        ax.plot(np.arange(len(ma)) + window - 1, ma, label=f"MA-{window}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, out_path)


def plot_multiple_curves(
    curves: dict[str, Sequence[float]],
    title: str,
    out_path: str | os.PathLike,
    window: int = 100,
    xlabel: str = "Episode",
    ylabel: str = "Return",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals in curves.items():
        ma = _moving_average(vals, window)
        if len(ma) == 0:
            ax.plot(vals, label=name)
        else:
            ax.plot(np.arange(len(ma)) + window - 1, ma, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, out_path)


def save_fig(fig: plt.Figure, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_value_grid(
    V: np.ndarray, shape: tuple[int, int], title: str, out_path: str | os.PathLike
) -> None:
    """Plot a state-value function laid out on a 2D grid (FrozenLake / CliffWalking)."""
    grid = V.reshape(shape)
    fig, ax = plt.subplots(figsize=(shape[1] * 0.7 + 1, shape[0] * 0.7 + 1))
    im = ax.imshow(grid, cmap="viridis")
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() * 0.6 else "black",
                    fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    save_fig(fig, out_path)
