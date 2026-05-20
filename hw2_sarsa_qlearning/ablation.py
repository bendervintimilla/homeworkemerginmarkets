"""Hyperparameter ablation: 3x3 grid of alpha x epsilon_end on CliffWalking.

Justifies the alpha=0.5 / eps_end=0.05 choice used in the main run. For each
(alpha, eps_end) cell we run SARSA and Q-Learning with 3 seeds and 5,000
episodes (half the main run, enough to see the asymptote). We report mean
final-window return per cell.

The hypothesis is that:
  - SARSA prefers smaller alpha (more conservative, less volatile near cliff)
  - Q-Learning is more robust to alpha but blows up at large alpha + low eps
  - eps_end < 0.05 hurts both because it kills late-stage exploration too fast
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .algorithms import sarsa, q_learning


ALPHAS = [0.05, 0.1, 0.5]
EPS_ENDS = [0.01, 0.05, 0.10]
SEEDS = [42, 123, 456]
NUM_EPISODES = 5_000


def _resolve(base, versions):
    for v in versions:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"Cannot find {base}")


def run_cell(algo_fn, env_id: str, alpha: float, eps_end: float,
             seeds: list[int], num_eps: int) -> dict:
    """Train `algo_fn` over `seeds` and return (mean, std) of final return."""
    finals = []
    for seed in seeds:
        env = gym.make(env_id)
        res = algo_fn(env, num_episodes=num_eps, alpha=alpha,
                     gamma=0.99, eps_start=1.0, eps_end=eps_end,
                     decay_frac=0.8, seed=seed)
        env.close()
        final = float(res.returns[-200:].mean())
        finals.append(final)
    return {
        "alpha": alpha,
        "eps_end": eps_end,
        "mean": float(np.mean(finals)),
        "std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "per_seed": finals,
    }


def run_ablation(algo_name: str, algo_fn, env_id: str) -> np.ndarray:
    """Return a (len(ALPHAS), len(EPS_ENDS)) grid of mean final returns."""
    grid_mean = np.zeros((len(ALPHAS), len(EPS_ENDS)))
    grid_std = np.zeros((len(ALPHAS), len(EPS_ENDS)))
    print(f"\n{'-' * 70}")
    print(f"  {algo_name}")
    print(f"{'-' * 70}")
    print(f"{'':>6s} | " + " | ".join(f"eps_end={e:.2f}" for e in EPS_ENDS))
    for i, alpha in enumerate(ALPHAS):
        cells = []
        for j, eps_end in enumerate(EPS_ENDS):
            t0 = time.time()
            cell = run_cell(algo_fn, env_id, alpha, eps_end, SEEDS, NUM_EPISODES)
            elapsed = time.time() - t0
            grid_mean[i, j] = cell["mean"]
            grid_std[i, j] = cell["std"]
            cells.append(f"{cell['mean']:+8.2f}")
            print(f"  α={alpha:.2f}  eps_end={eps_end:.2f}  → {cell['mean']:+7.2f} ± {cell['std']:5.2f}  ({elapsed:.1f}s)")
    return grid_mean, grid_std


def plot_heatmap(grid_mean: np.ndarray, algo_name: str, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(grid_mean, cmap="RdYlGn", aspect="equal")

    for i in range(grid_mean.shape[0]):
        for j in range(grid_mean.shape[1]):
            ax.text(j, i, f"{grid_mean[i, j]:+.1f}",
                    ha="center", va="center",
                    color="black", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(EPS_ENDS)))
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_xticklabels([f"ε_end={e:.2f}" for e in EPS_ENDS])
    ax.set_yticklabels([f"α={a:.2f}" for a in ALPHAS])
    ax.set_xlabel("Final exploration rate")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"{algo_name} — final return on CliffWalking\n"
                 f"3 seeds × 5,000 eps per cell  ·  greener = better",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="mean final return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    CLIFF = _resolve("CliffWalking", ["v0", "v1"])
    out_assets = Path("hw2_sarsa_qlearning/assets")
    out_assets.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()
    for algo_name, algo_fn in [("sarsa", sarsa), ("q_learning", q_learning)]:
        grid_mean, grid_std = run_ablation(algo_name, algo_fn, CLIFF)
        results[algo_name] = {
            "alphas": ALPHAS,
            "eps_ends": EPS_ENDS,
            "grid_mean": grid_mean.tolist(),
            "grid_std": grid_std.tolist(),
        }
        plot_heatmap(grid_mean, algo_name.upper(),
                     out_assets / f"ablation_{algo_name}.png")

    out_json = Path("hw2_sarsa_qlearning/results/ablation.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
