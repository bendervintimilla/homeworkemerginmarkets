"""Plot n-step SARSA learning curves and final-bar comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {1: "#4cc9f0", 4: "#f9c74f", 16: "#f25c54"}
LABELS = {1: "n=1 (regular SARSA)", 4: "n=4", 16: "n=16 (near Monte Carlo)"}


def smooth(y, k):
    k = max(1, min(k, len(y) // 4))
    return np.convolve(y, np.ones(k) / k, mode="valid")


def bootstrap_ci(stack, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_seeds, n_steps = stack.shape
    boot = np.empty((n_boot, n_steps), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot[b] = stack[idx].mean(axis=0)
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return stack.mean(axis=0), lo, hi


def main():
    data = np.load("hw2_sarsa_qlearning/results/n_step_sarsa.npz")
    n_values = data["n_values"]
    out_assets = Path("hw2_sarsa_qlearning/assets")
    out_assets.mkdir(parents=True, exist_ok=True)

    # --- Learning curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for n in n_values:
        rets = data[f"n{n}_returns"]  # (seeds, episodes)
        stack = np.stack([smooth(r, 50) for r in rets])
        mean, lo, hi = bootstrap_ci(stack, seed=int(n))
        x = np.arange(len(mean)) + 49
        ax.plot(x, mean, color=COLORS[int(n)], linewidth=2.2, label=LABELS[int(n)])
        ax.fill_between(x, lo, hi, color=COLORS[int(n)], alpha=0.15)
    ax.set_title("n-step SARSA on CliffWalking — bias-variance trade-off\n"
                 "5 seeds, α=0.5, γ=0.99, ε decays 1.0→0.05 over 80% of training\n"
                 "Shaded = 95% bootstrap CI",
                 fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Training return (50-ep moving avg)")
    ax.set_ylim(-200, 0)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_assets / "n_step_sarsa_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

    # --- Final bars: mean ± std ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, stds = [], []
    for n in n_values:
        finals = data[f"n{n}_returns"][:, -200:].mean(axis=1)
        means.append(finals.mean())
        stds.append(finals.std(ddof=1))
    bars = ax.bar(range(len(n_values)), means, yerr=stds, capsize=8,
                  color=[COLORS[int(n)] for n in n_values],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m - s - 4,
                f"{m:+.2f}\n±{s:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f"n={int(n)}" for n in n_values])
    ax.set_ylabel("Final training return (last 200 eps, mean ± std, 5 seeds)")
    ax.set_title("n-step SARSA: as n grows, variance grows", fontsize=11)
    ax.set_ylim(-55, 0)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out2 = out_assets / "n_step_sarsa_bars.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
