"""Master efficiency-frontier plot combining all three phases.

A single figure that puts BC scaling law, DAgger, and the PPO learning curve
on the same axes (or two complementary subplots) so the trade-off is visually
obvious in one image.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def main():
    scaling = json.loads((RUNS / "bc_scaling_walker2d" / "results.json").read_text())
    dagger = json.loads((RUNS / "dagger_walker2d" / "results.json").read_text())
    eff = json.loads((RUNS / "data_efficiency_walker2d.json").read_text())

    Ns = np.array(scaling["N"])
    bc_mean = np.array(scaling["mean"])
    bc_std = np.array(scaling["std"])
    expert_mean = scaling["expert_mean"]
    expert_std = scaling["expert_std"]

    dag_it = np.array(dagger["iteration"])
    dag_mean = np.array(dagger["mean_reward"])
    dag_std = np.array(dagger["std_reward"])
    dag_size = np.array(dagger["dataset_size"])

    ppo_t = np.array(eff["ppo_timesteps"])
    ppo_r = np.array(eff["ppo_smoothed_reward"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # === Subplot 1: BC scaling law ===
    ax1.errorbar(Ns, bc_mean, yerr=bc_std, fmt="o-", color="C0", capsize=4,
                 linewidth=2, label="BC")
    ax1.axhline(expert_mean, color="black", linestyle="--", linewidth=1,
                label=f"PPO expert ({expert_mean:.0f})")
    ax1.fill_between([Ns.min(), Ns.max()],
                     expert_mean - expert_std, expert_mean + expert_std,
                     color="black", alpha=0.08)
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of expert demonstrations (log)")
    ax1.set_ylabel("Eval reward (mean ± std, 20 episodes)")
    ax1.set_title("Phase 1 — BC scaling law")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Annotate knee.
    fracs = bc_mean / expert_mean
    if (fracs >= 0.9).any():
        first_90 = int(np.argmax(fracs >= 0.9))
        ax1.annotate(f">90% expert\nN={Ns[first_90]}",
                     xy=(Ns[first_90], bc_mean[first_90]),
                     xytext=(Ns[first_90] * 0.4, bc_mean[first_90] - 800),
                     fontsize=9, ha="center",
                     arrowprops=dict(arrowstyle="->", color="C0", alpha=0.7))

    # === Subplot 2: DAgger + PPO curve overlaid ===
    # DAgger on left y, PPO learning curve as background
    ax2.plot(ppo_t / 1000, ppo_r, color="C3", alpha=0.5, linewidth=1.5,
             label="PPO online (env steps)")
    ax2.axhline(expert_mean, color="black", linestyle="--", linewidth=1,
                label=f"Expert ({expert_mean:.0f})")
    ax2.set_xlabel("PPO env steps (× 1k)  |  DAgger iter dataset size (norm.)")
    ax2.set_ylabel("Eval reward")

    # Plot DAgger on a SECOND x-axis (top) showing iteration index.
    ax2b = ax2.twiny()
    ax2b.errorbar(dag_it, dag_mean, yerr=dag_std, fmt="s-", color="C2",
                  capsize=4, linewidth=2, label="DAgger (iterations)")
    ax2b.set_xlabel("DAgger iteration", color="C2")
    ax2b.tick_params(axis="x", labelcolor="C2")

    # Merge legends.
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right")
    ax2.set_title("Phases 2–3 — DAgger vs PPO online")
    ax2.grid(alpha=0.3)

    fig.suptitle("Walker2D-v5 — Offline imitation vs online interaction",
                 fontsize=13)
    fig.tight_layout()
    out = ASSETS / "efficiency_frontier.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")

    # === Additional plot: % expert recovered ===
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, fracs * 100, "o-", color="C0", linewidth=2, label="BC")
    if len(dag_it) > 1:
        dag_frac = dag_mean / expert_mean * 100
        # Place DAgger iterations on the x-axis as "equivalent demos"
        # using cumulative dataset size in trajectories (≈860 steps/traj).
        dag_n_equiv = dag_size / 860.0
        ax.plot(dag_n_equiv, dag_frac, "s-", color="C2", linewidth=2,
                label="DAgger (cumulative)")
    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="Expert (100%)")
    ax.axhline(90, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Effective dataset size (expert-trajectory equivalents)")
    ax.set_ylabel("% expert reward recovered")
    ax.set_title("Walker2D-v5 — % expert recovered vs demo budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = ASSETS / "percent_expert_recovered.png"
    fig2.savefig(out2, dpi=120)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
