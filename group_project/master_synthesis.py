"""Master synthesis plot — all approaches on one axis."""

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
    dense = json.loads((RUNS / "expert_eval_dense_walker2d.json").read_text())
    stoch = json.loads((RUNS / "stoch_expert_walker2d" / "results.json").read_text())
    dagger = json.loads((RUNS / "dagger_walker2d" / "results.json").read_text())
    eff = json.loads((RUNS / "data_efficiency_walker2d.json").read_text())

    exp_m = dense["mean"]
    exp_sem = dense["sem"]
    exp_std = dense["std"]

    stoch_m = stoch["variants"][1]["bc_eval_R"]
    stoch_s = stoch["variants"][1]["bc_eval_R_std"]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # PPO online learning curve as the background "online budget" axis
    ppo_t = np.array(eff["ppo_timesteps"])
    ppo_r = np.array(eff["ppo_smoothed_reward"])
    ax.plot(ppo_t, ppo_r, color="gray", alpha=0.5, linewidth=2, label="PPO online (1.5M env-steps)")

    # Reference bands
    ax.axhline(exp_m, color="black", linestyle="--", linewidth=1.5,
               label=f"PPO expert: {exp_m:.0f} (n=100)")
    ax.fill_between([0, ppo_t.max()],
                    exp_m - 1.96 * exp_sem, exp_m + 1.96 * exp_sem,
                    color="black", alpha=0.10)

    # BC scaling — at "effective env-step equivalents" via PPO matching curve.
    # For visual placement, use the demo count * 850 pairs/demo as "samples".
    Ns = np.array(scaling["N"])
    bc_m = np.array(scaling["mean"])
    bc_s = np.array(scaling["std"])
    bc_x = Ns * 850  # offline (s, a) pair count as x-axis (commensurable with env-steps)
    ax.errorbar(bc_x, bc_m, yerr=bc_s, fmt="o-", color="C0", capsize=4,
                linewidth=2, markersize=8, label="BC deterministic (N demos)")

    # Annotate N labels
    for x, y, n in zip(bc_x, bc_m, Ns):
        ax.annotate(f"N={n}", (x, y), xytext=(5, -10), textcoords="offset points",
                    fontsize=8, color="C0")

    # BC stochastic — single point at the same "data budget" as N=50
    sto_x = 50 * 850 * 0.5  # half the data because episodes are half as long
    ax.errorbar([sto_x], [stoch_m], yerr=[stoch_s], fmt="*", color="C1",
                markersize=22, capsize=4, linewidth=2,
                label=f"BC stochastic, N=50: {stoch_m:.0f}")
    ax.annotate("WINNER\n(noise > more data)", (sto_x, stoch_m),
                xytext=(20, 30), textcoords="offset points", fontsize=10,
                color="C1", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="C1"))

    # DAgger trajectory — x = cumulative dataset size
    dag_it = np.array(dagger["iteration"])
    dag_m = np.array(dagger["mean_reward"])
    dag_s = np.array(dagger["std_reward"])
    dag_x = np.array(dagger["dataset_size"])  # in (s,a) pairs
    ax.errorbar(dag_x, dag_m, yerr=dag_s, fmt="s-", color="C3",
                capsize=4, linewidth=2, markersize=7, alpha=0.85,
                label=f"DAgger β=0 (5 iter; mean = {dag_m[1:].mean():.0f})")

    ax.set_xscale("log")
    ax.set_xlabel("Effective data budget — offline (s,a) pairs OR online env-steps (log)")
    ax.set_ylabel("Eval reward (mean ± std, 20 episodes)")
    ax.set_title("Walker2D-v5 — Data efficiency frontier across offline and online learning")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1000, ppo_t.max() * 1.1)
    ax.set_ylim(min(bc_m.min(), dag_m.min()) - 300, max(bc_m.max(), exp_m) + 400)

    fig.tight_layout()
    out = ASSETS / "master_synthesis.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
