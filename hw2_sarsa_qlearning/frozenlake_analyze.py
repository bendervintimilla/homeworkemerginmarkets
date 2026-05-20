"""Post-hoc analysis of FrozenLake runs. Computes bias correctly for Double Q
(uses (Q1+Q2)/2, not the sum) and generates a plot showing the asymmetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "sarsa": "#4cc9f0",
    "q_learning": "#ff7a3d",
    "expected_sarsa": "#5ee6a3",
    "double_q": "#c084fc",
}
LABELS = {
    "sarsa": "SARSA",
    "q_learning": "Q-Learning",
    "expected_sarsa": "Expected SARSA",
    "double_q": "Double Q-Learning",
}


def main():
    data = np.load("hw2_sarsa_qlearning/results/frozenlake.npz")
    V_star = data["V_star"]
    n_states = len(V_star)

    summary = {}
    print(f"FrozenLake-v1 (slippery)  ·  V*(start) = {V_star[0]:.4f}")
    print(f"{'algo':<18s} {'success':>10s} {'mean_bias':>11s} {'|mean_bias|':>12s} {'max_bias':>10s}")
    for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
        Qs = data[f"{algo}_Q"]   # (seeds, S, A)
        rets = data[f"{algo}_returns"]
        success = float(rets[:, -1000:].mean())
        success_std = float(rets[:, -1000:].mean(axis=1).std(ddof=1))

        # CRITICAL: Double Q stores Q1+Q2 (per the algorithms_extended.Q property).
        # The unbiased value estimator is the AVERAGE, not the sum.
        if algo == "double_q":
            Qs_eval = Qs / 2.0
        else:
            Qs_eval = Qs

        bias = Qs_eval.max(axis=2).mean(axis=0) - V_star
        mean_bias = float(bias.mean())
        mean_abs_bias = float(np.abs(bias).mean())
        max_bias = float(bias.max())
        print(f"{LABELS[algo]:<18s} "
              f"{success:>7.3f}±{success_std:.3f} "
              f"{mean_bias:>+11.4f} {mean_abs_bias:>12.4f} {max_bias:>+10.4f}")
        summary[algo] = {
            "success_rate_mean": success,
            "success_rate_std": success_std,
            "mean_bias": mean_bias,
            "mean_abs_bias": mean_abs_bias,
            "max_bias": max_bias,
            "bias_per_state": bias.tolist(),
        }

    # --- Plot: bias per state on the 4x4 grid for each algo ---------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 6),
                              gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    biases = {a: np.array(summary[a]["bias_per_state"]) for a in LABELS}
    vmax = max(np.abs(b).max() for b in biases.values())
    ims = []
    for ax, algo in zip(axes.flat, ["sarsa", "q_learning", "expected_sarsa", "double_q"]):
        bias = biases[algo]
        im = ax.imshow(bias.reshape(4, 4), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ims.append(im)
        ax.set_title(f"{LABELS[algo]}  ·  mean bias = {summary[algo]['mean_bias']:+.3f}",
                     fontsize=11)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        # Mark holes (3,5,7,11,12) and goal (15) from 4x4 default map
        holes = {5, 7, 11, 12}
        goal = 15
        start = 0
        for s in range(16):
            r, c = s // 4, s % 4
            if s in holes:
                ax.text(c, r, "H", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
            elif s == goal:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="black")
            elif s == start:
                ax.text(c, r, "S", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="black")
    fig.suptitle(
        "Overestimation bias on FrozenLake-v1 (slippery)\n"
        "max_a Q − V*  ·  red = overestimate, blue = underestimate",
        fontsize=12, y=1.02)
    fig.colorbar(ims[0], ax=list(axes.flat), shrink=0.7, label="bias")
    out = Path("hw2_sarsa_qlearning/assets/frozenlake_bias.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # --- Bar plot: success rate vs V* threshold --------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = ["sarsa", "q_learning", "expected_sarsa", "double_q"]
    means = [summary[a]["success_rate_mean"] for a in algos]
    stds = [summary[a]["success_rate_std"] for a in algos]
    bars = ax.bar(range(4), means, yerr=stds, capsize=8,
                  color=[COLORS[a] for a in algos],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axhline(V_star[0], color="red", linestyle="--", linewidth=1.5,
               label=f"V*(start) = {V_star[0]:.3f}  (optimal)")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.02,
                f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABELS[a] for a in algos], rotation=10)
    ax.set_ylabel("Final success rate (avg last 1000 eps, 10 seeds)")
    ax.set_title("FrozenLake-v1 (slippery): all algos converge near V*\nexcept Double Q under-trained at this α/episode count",
                 fontsize=11)
    ax.set_ylim(0, 0.65)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out2 = Path("hw2_sarsa_qlearning/assets/frozenlake_success.png")
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Saved {out2}")

    Path("hw2_sarsa_qlearning/results/frozenlake_analysis.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
