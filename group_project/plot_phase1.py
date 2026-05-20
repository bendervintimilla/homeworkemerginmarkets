"""Quick plot of Phase 1 BC scaling law."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def main():
    data = json.loads((RUNS / "bc_scaling_walker2d" / "results.json").read_text())
    dense = json.loads((RUNS / "expert_eval_dense_walker2d.json").read_text())
    Ns = np.array(data["N"])
    means = np.array(data["mean"])
    stds = np.array(data["std"])
    # Use the dense (100 ep) expert reference instead of the noisy 20-ep one.
    exp_m = dense["mean"]
    exp_s = dense["std"]
    exp_sem = dense["sem"]
    fracs = means / exp_m * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute reward with expert band
    ax1.errorbar(Ns, means, yerr=stds, fmt="o-", color="C0", capsize=4,
                 linewidth=2, markersize=8, label="BC")
    ax1.axhline(exp_m, color="black", linestyle="--", linewidth=1.5,
                label=f"PPO expert: {exp_m:.0f} ± {exp_s:.0f} (n=100)")
    # 95% CI on the expert MEAN (mean ± 1.96 SEM), not std (which is per-episode).
    ax1.fill_between([Ns.min() * 0.8, Ns.max() * 1.2],
                     exp_m - 1.96 * exp_sem, exp_m + 1.96 * exp_sem,
                     color="black", alpha=0.15, label="Expert 95% CI")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of expert demonstrations (log)")
    ax1.set_ylabel("Eval reward (mean ± std, 20 episodes)")
    ax1.set_title("Phase 1 — BC scaling law on Walker2D-v5")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(Ns.min() * 0.8, Ns.max() * 1.2)

    # Right: % expert recovered
    bars = ax2.bar(range(len(Ns)), fracs, color="C0", alpha=0.7,
                   edgecolor="black", linewidth=0.8)
    for bar, frac in zip(bars, fracs):
        ax2.text(bar.get_x() + bar.get_width() / 2, frac + 1.5,
                 f"{frac:.1f}%", ha="center", fontsize=9)
    ax2.axhline(100, color="black", linestyle="--", linewidth=1.5,
                label="Expert reference (100%)")
    ax2.axhline(90, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_xticks(range(len(Ns)))
    ax2.set_xticklabels([f"N={n}" for n in Ns])
    ax2.set_ylabel("% expert reward recovered")
    ax2.set_title("BC vs PPO expert at each N")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylim(0, max(fracs.max() + 10, 130))

    fig.suptitle("Walker2D-v5 — BC saturates immediately; BC can EXCEED expert via MSE smoothing",
                 fontsize=12)
    fig.tight_layout()
    out = ASSETS / "phase1_bc_scaling.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
