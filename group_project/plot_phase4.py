"""Phase 4 plot: stochastic vs deterministic expert collection."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def main():
    data = json.loads((RUNS / "stoch_expert_walker2d" / "results.json").read_text())
    dense = json.loads((RUNS / "expert_eval_dense_walker2d.json").read_text())

    a, b = data["variants"]  # deterministic, stochastic
    exp_m = dense["mean"]
    exp_sem = dense["sem"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart of reward
    labels = ["BC trained on\ndeterministic demos", "BC trained on\nstochastic demos"]
    means = [a["bc_eval_R"], b["bc_eval_R"]]
    stds = [a["bc_eval_R_std"], b["bc_eval_R_std"]]
    colors = ["C0", "C1"]

    bars = ax1.bar(labels, means, yerr=stds, color=colors, alpha=0.7,
                   edgecolor="black", linewidth=0.8, capsize=8)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + s + 60,
                 f"{m:.0f} ± {s:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax1.axhline(exp_m, color="black", linestyle="--", linewidth=1.5,
                label=f"PPO expert ({exp_m:.0f}, n=100)")
    ax1.fill_between([-0.5, 1.5],
                     exp_m - 1.96 * exp_sem, exp_m + 1.96 * exp_sem,
                     color="black", alpha=0.10)
    delta = b["bc_eval_R"] - a["bc_eval_R"]
    ax1.annotate("", xy=(1, b["bc_eval_R"]), xytext=(0, a["bc_eval_R"]),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax1.text(0.5, (a["bc_eval_R"] + b["bc_eval_R"]) / 2,
             f"Δ = +{delta:.0f}\n({delta/a['bc_eval_R']*100:+.1f}%)",
             ha="center", fontsize=11, color="red", fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.9))
    ax1.set_ylabel("BC eval reward (20 episodes)")
    ax1.set_title("BC final reward: stochastic > deterministic")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0, max(means) + max(stds) + 300)
    ax1.set_xlim(-0.5, 1.5)

    # Right: comparison of dataset characteristics
    metrics = {
        "Pairs collected\n(more = better?)": [a["n_pairs"], b["n_pairs"]],
        "Expert R during collection\n(stochastic falls earlier)": [
            a["expert_during_collection_R"], b["expert_during_collection_R"]],
        "State-coverage proxy\n(trace cov of obs)": [
            a["state_coverage"]["trace_cov"],
            b["state_coverage"]["trace_cov"]],
    }
    x = np.arange(len(metrics))
    width = 0.35
    det_vals = [v[0] for v in metrics.values()]
    sto_vals = [v[1] for v in metrics.values()]
    # Normalize each metric to its max for comparability
    det_norm = [d / max(d, s) for d, s in zip(det_vals, sto_vals)]
    sto_norm = [s / max(d, s) for d, s in zip(det_vals, sto_vals)]

    bars1 = ax2.bar(x - width/2, det_norm, width, color="C0", alpha=0.7,
                    edgecolor="black", linewidth=0.8, label="deterministic")
    bars2 = ax2.bar(x + width/2, sto_norm, width, color="C1", alpha=0.7,
                    edgecolor="black", linewidth=0.8, label="stochastic")
    # Annotate with raw values
    for bar, val in zip(bars1, det_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:,.0f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, sto_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:,.0f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax2.set_ylabel("Normalized to max per metric")
    ax2.set_title("Why does stochastic win? — collection diagnostics")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.25)

    fig.suptitle("Phase 4 — Noise at data-collection time beats more data",
                 fontsize=12)
    fig.tight_layout()
    out = ASSETS / "phase4_stochastic_vs_deterministic.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
