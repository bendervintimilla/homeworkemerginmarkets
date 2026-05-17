"""Bias and regret analysis against the optimal V*.

For each algorithm and each state s, compute:

  bias(s)   = max_a Q_algo(s, a)  -  V*(s)
            <0 → underestimation (SARSA's exploratory cost)
            >0 → overestimation (Q-Learning's max bias)
            ~0 → unbiased (Double Q-Learning expected here)

  regret(s) = V*(s) - V_algo_greedy(s)
            where V_algo_greedy(s) = Q*(s, argmax_a Q_algo(s, a))
            measures how suboptimal the greedy policy from this algo is.

Generates:
  - results/bias_analysis.json   - per-algo per-state bias statistics
  - assets/bias_per_state.png    - heatmap of bias on CliffWalking 4×12 grid
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    vi = np.load("hw2_sarsa_qlearning/results/value_iteration.npz")

    out_assets = Path("hw2_sarsa_qlearning/assets")
    out_assets.mkdir(exist_ok=True, parents=True)

    summary = {}

    for env_label, env_key in [("cliff", "cliff"), ("taxi", "taxi")]:
        V_star = vi[f"{env_key}_V_star"]
        Q_star = vi[f"{env_key}_Q_star"]
        summary[env_label] = {}

        print(f"\n=== {env_label.upper()} (|S|={len(V_star)}) ===")
        print(f"{'algorithm':<18} {'mean_bias':>11} {'mean|bias|':>11} {'mean_regret':>11}")

        bias_per_algo = {}
        for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
            Qs = runs[f"{env_label}_{algo}_Q"]  # (seeds, S, A)
            # Double Q stores Q1+Q2 (sum). The unbiased VALUE estimator is
            # the average (Q1+Q2)/2 — argmax is invariant either way.
            if algo == "double_q":
                Qs = Qs / 2.0
            bias = Qs.max(axis=2).mean(axis=0) - V_star
            greedy_a = Qs.mean(axis=0).argmax(axis=1)
            V_algo = Q_star[np.arange(len(V_star)), greedy_a]
            regret = V_star - V_algo

            mean_bias = float(bias.mean())
            mean_abs_bias = float(np.abs(bias).mean())
            mean_regret = float(regret.mean())
            summary[env_label][algo] = {
                "mean_bias": mean_bias,
                "mean_abs_bias": mean_abs_bias,
                "mean_regret": mean_regret,
                "bias_per_state": bias.tolist(),
            }
            bias_per_algo[algo] = bias
            print(f"{algo:<18} {mean_bias:>11.4f} {mean_abs_bias:>11.4f} {mean_regret:>11.4f}")

        if env_label == "cliff":
            fig, axes = plt.subplots(2, 2, figsize=(13, 6.5),
                                     gridspec_kw={"hspace": 0.45, "wspace": 0.20})
            vmax = max(np.abs(b).max() for b in bias_per_algo.values())
            ims = []
            for ax, (algo, bias) in zip(axes.flat, bias_per_algo.items()):
                im = ax.imshow(bias.reshape(4, 12), cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax)
                ims.append(im)
                ax.set_title(f"{algo}  ·  mean bias = {bias.mean():+.3f}")
                ax.set_xticks(range(12))
                ax.set_yticks(range(4))
                ax.tick_params(labelsize=8)
                for c in range(1, 11):
                    ax.text(c, 3, "C", ha="center", va="center",
                            color="black", fontsize=9, fontweight="bold")
                ax.text(0, 3, "S", ha="center", va="center", fontsize=9, fontweight="bold")
                ax.text(11, 3, "G", ha="center", va="center", fontsize=9, fontweight="bold")
            fig.suptitle(
                "Overestimation bias on CliffWalking (max_a Q − V*)\n"
                "Red = overestimates V*  •  Blue = underestimates V*",
                fontsize=13, y=1.02)
            fig.colorbar(ims[0], ax=list(axes.flat), shrink=0.7, label="bias")
            out = out_assets / "bias_per_state.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\nSaved {out}")

    out_json = Path("hw2_sarsa_qlearning/results/bias_analysis.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
