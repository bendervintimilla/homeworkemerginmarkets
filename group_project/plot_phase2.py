"""Phase 2 plot: DAgger iterations, vanilla β=0 vs β=0.3 mix."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def main():
    base = json.loads((RUNS / "dagger_walker2d" / "results.json").read_text())
    dense = json.loads((RUNS / "expert_eval_dense_walker2d.json").read_text())
    stoch = json.loads((RUNS / "stoch_expert_walker2d" / "results.json").read_text())

    # Optional β=0.3 ablation
    beta_path = RUNS / "dagger_walker2d_beta0.3" / "results.json"
    beta = json.loads(beta_path.read_text()) if beta_path.exists() else None

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Reference bands
    exp_m = dense["mean"]
    exp_sem = dense["sem"]
    ax.axhline(exp_m, color="black", linestyle="--", linewidth=1.5,
               label=f"PPO expert: {exp_m:.0f} (n=100)")
    ax.fill_between([-0.5, max(base["iteration"]) + 0.5],
                    exp_m - 1.96 * exp_sem, exp_m + 1.96 * exp_sem,
                    color="black", alpha=0.10, label="Expert 95% CI")
    stoch_R = stoch["variants"][1]["bc_eval_R"]
    ax.axhline(stoch_R, color="C1", linestyle=":", linewidth=2,
               label=f"BC stochastic (Phase 4): {stoch_R:.0f}")

    # Vanilla DAgger
    it = np.array(base["iteration"])
    m = np.array(base["mean_reward"])
    s = np.array(base["std_reward"])
    ax.errorbar(it, m, yerr=s, fmt="o-", color="C0", capsize=5, linewidth=2,
                markersize=8, label="DAgger (β=0, vanilla)")

    # Optional β=0.3
    if beta is not None:
        it2 = np.array(beta["iteration"])
        m2 = np.array(beta["mean_reward"])
        s2 = np.array(beta["std_reward"])
        ax.errorbar(it2 + 0.06, m2, yerr=s2, fmt="s-", color="C2", capsize=5,
                    linewidth=2, markersize=8, label="DAgger (β=0.3)")

    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Eval reward (mean ± std, 20 episodes)")
    ax.set_title("Phase 2 — DAgger does not improve over BC baseline on Walker2D-v5")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xticks(it)
    ax.set_xlim(-0.3, max(it) + 0.5)

    fig.tight_layout()
    out = ASSETS / "phase2_dagger.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
