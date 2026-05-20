"""Master plots from the 4-algorithms × 10-seeds run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "sarsa": "#4cc9f0",          # cyan
    "q_learning": "#ff7a3d",     # orange
    "expected_sarsa": "#5ee6a3", # green
    "double_q": "#c084fc",       # purple
}
LABELS = {
    "sarsa": "SARSA (on-policy)",
    "q_learning": "Q-Learning (off-policy)",
    "expected_sarsa": "Expected SARSA",
    "double_q": "Double Q-Learning",
}


def smooth(y, k):
    k = max(1, min(k, len(y) // 4))
    return np.convolve(y, np.ones(k) / k, mode="valid")


def bootstrap_ci_band(stack: np.ndarray, n_boot: int = 1000,
                      alpha: float = 0.05,
                      rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap 95% CI band over the seed axis of a (seeds, time) curve.

    With n=10 seeds and visibly non-Gaussian per-seed returns (Q-Learning has
    fat lower tail from occasional cliff falls), the standard mean ± std band
    is misleading. The percentile bootstrap resamples seeds with replacement
    1000 times and takes the 2.5 / 97.5 percentiles of the resampled means at
    each time step. This is the textbook recipe (Efron & Tibshirani 1993)
    and the only legitimate option below n ≈ 30.
    """
    n_seeds, n_steps = stack.shape
    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty((n_boot, n_steps), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = stack[idx].mean(axis=0)
    lo = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    mean = stack.mean(axis=0)
    return mean, lo, hi


def plot_learning_curves(runs, env_label, window, out_path, title, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
        returns = runs[f"{env_label}_{algo}_returns"]  # (seeds, episodes)
        stack = np.stack([smooth(r, window) for r in returns])
        m, lo, hi = bootstrap_ci_band(stack, n_boot=1000, alpha=0.05,
                                       rng_seed=hash(algo) & 0xFFFF)
        x = np.arange(len(m)) + window - 1
        ax.plot(x, m, color=COLORS[algo], linewidth=2.2, label=LABELS[algo])
        ax.fill_between(x, lo, hi, color=COLORS[algo], alpha=0.18,
                        label="_nolegend_")
    ax.set_title(title + "  ·  shaded = 95% bootstrap CI (1000 resamples)",
                 fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Training return ({window}-ep moving avg)")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def plot_final_bars(runs, env_label, last_k, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    algos = ["sarsa", "q_learning", "expected_sarsa", "double_q"]
    means = []
    stds = []
    for algo in algos:
        returns = runs[f"{env_label}_{algo}_returns"]
        per_seed = returns[:, -last_k:].mean(axis=1)
        means.append(per_seed.mean())
        stds.append(per_seed.std())
    bars = ax.bar(range(4), means, yerr=stds, capsize=8,
                  color=[COLORS[a] for a in algos],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + (s + 1.5 if m > 0 else -s - 4),
                f"{m:+.2f}\n±{s:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABELS[a] for a in algos], rotation=10)
    ax.set_ylabel(f"Mean return over last {last_k} training episodes\n(across 10 seeds)")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def plot_figure_6_4(out_path):
    data = np.load("hw2_sarsa_qlearning/results/figure_6_4.npz")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sarsa_mean = data["sarsa_mean"]
    ql_mean = data["ql_mean"]
    # Smooth lightly to match the book's visual style
    sm = smooth(sarsa_mean, 10)
    qm = smooth(ql_mean, 10)
    x = np.arange(len(sm)) + 10 - 1
    ax.plot(x, sm, color=COLORS["sarsa"], linewidth=2.5, label="SARSA (on-policy)")
    ax.plot(x, qm, color=COLORS["q_learning"], linewidth=2.5, label="Q-Learning (off-policy)")
    ax.axhline(-17, color=COLORS["sarsa"], linestyle=":", linewidth=1, alpha=0.5,
               label="S&B reference: SARSA → −17")
    ax.axhline(-25, color=COLORS["q_learning"], linestyle=":", linewidth=1, alpha=0.5,
               label="S&B reference: Q-Learn → −25")
    ax.set_title(
        f"Reproduction of Sutton & Barto Figure 6.4 — CliffWalking, "
        f"α=0.5, γ=1.0, ε=0.1, {int(data['n_seeds'])} seeds averaged",
        fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sum of rewards during episode (smoothed)")
    ax.set_ylim(-100, 0)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def main():
    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    out_assets = Path("hw2_sarsa_qlearning/assets")
    out_assets.mkdir(exist_ok=True, parents=True)

    plot_learning_curves(
        runs, "taxi", window=300,
        out_path=out_assets / "taxi_4algos.png",
        title="Taxi-v3 — 4 TD-control algorithms, 10 seeds, ±1 std",
        ylim=(-800, 30),
    )
    plot_learning_curves(
        runs, "cliff", window=50,
        out_path=out_assets / "cliff_4algos.png",
        title="CliffWalking — 4 TD-control algorithms, 10 seeds, ±1 std",
        ylim=(-200, 5),
    )
    plot_final_bars(
        runs, "taxi", last_k=200,
        out_path=out_assets / "taxi_final_bars.png",
        title="Taxi-v3 — final training return by algorithm",
    )
    plot_final_bars(
        runs, "cliff", last_k=200,
        out_path=out_assets / "cliff_final_bars.png",
        title="CliffWalking — final training return by algorithm",
    )

    fig_6_4_npz = Path("hw2_sarsa_qlearning/results/figure_6_4.npz")
    if fig_6_4_npz.exists():
        plot_figure_6_4(out_assets / "figure_6_4_reproduction.png")


if __name__ == "__main__":
    main()
