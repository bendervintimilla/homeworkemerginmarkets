"""Generate plots for the group_project REPORT.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def plot_learning_curve():
    data = np.load(RUNS / "ppo_walker2d" / "eval" / "evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]
    mean = results.mean(axis=1)
    std = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timesteps, mean, color="C0", linewidth=2, label="PPO eval reward")
    ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color="C0")
    ax.axhline(2500, color="gray", linestyle="--", linewidth=1, label="Walker2D solved threshold")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward (eval, mean ± std)")
    ax.set_title("PPO on Walker2D-v5 — 1.5M steps, 4 envs")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "ppo_walker2d_learning_curve.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def plot_bc_vs_expert():
    data = np.load(RUNS / "eval_walker2d.npz")
    exp = data["expert_returns"]
    bc = data["bc_returns"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = [1, 2]
    parts = ax.violinplot([exp, bc], positions=positions, showmeans=True, showmedians=False)
    for body, color in zip(parts["bodies"], ["C0", "C1"]):
        body.set_facecolor(color)
        body.set_alpha(0.4)
    ax.scatter([1] * len(exp), exp, color="C0", alpha=0.6, label=f"PPO expert: {exp.mean():.0f} ± {exp.std():.0f}")
    ax.scatter([2] * len(bc), bc, color="C1", alpha=0.6, label=f"BC: {bc.mean():.0f} ± {bc.std():.0f}")
    ax.set_xticks(positions, ["PPO expert", "Behavior Cloning"])
    ax.set_ylabel("Episode reward (20 episodes)")
    ax.set_title(f"PPO expert vs BC — gap {(exp.mean() - bc.mean()) / exp.mean() * 100:.1f}%")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = ASSETS / "bc_vs_expert.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_learning_curve()
    plot_bc_vs_expert()
