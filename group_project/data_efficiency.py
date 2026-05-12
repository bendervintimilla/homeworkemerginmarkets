"""Phase 3 — Demo-equivalence: how many PPO env steps does one expert demo replace?

Premise: we already have the PPO learning curve from EvalCallback's
evaluations.npz (eval every 10k steps for 1.5M total steps). For each BC
scaling-law point N (expert demos), find the PPO env-step count at which PPO
crosses the same reward level. That number is the "online cost equivalent" of
one expert demo at that operating point.

Why this matters: it tells you, given a fixed compute or data budget, whether
to spend it on (a) more expert demos for BC, or (b) more PPO env steps. The
two are commensurable through this curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    """Simple centered moving average to reduce eval noise."""
    if len(y) < k:
        return y
    pad = k // 2
    y_padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(k) / k
    return np.convolve(y_padded, kernel, mode="valid")


def find_crossing(timesteps: np.ndarray, smoothed_reward: np.ndarray, target: float) -> float | None:
    """Return the env-step count at which smoothed reward first crosses `target`,
    using linear interpolation. None if never crosses."""
    above = smoothed_reward >= target
    if not above.any():
        return None
    idx = int(np.argmax(above))
    if idx == 0:
        return float(timesteps[0])
    # Linear interpolation between idx-1 and idx.
    x0, x1 = timesteps[idx - 1], timesteps[idx]
    y0, y1 = smoothed_reward[idx - 1], smoothed_reward[idx]
    if y1 == y0:
        return float(x1)
    frac = (target - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def main():
    # 1. Load PPO learning curve.
    ppo_eval = np.load(RUNS / "ppo_walker2d" / "eval" / "evaluations.npz")
    timesteps = ppo_eval["timesteps"]
    results = ppo_eval["results"]  # (n_evals, n_episodes_per_eval)
    mean_per_eval = results.mean(axis=1)
    smoothed = smooth(mean_per_eval, k=5)

    # 2. Load BC scaling results.
    scaling = json.loads((RUNS / "bc_scaling_walker2d" / "results.json").read_text())
    Ns = np.array(scaling["N"])
    bc_means = np.array(scaling["mean"])
    expert_mean = scaling["expert_mean"]

    # 3. For each BC operating point, find the PPO step count that matches.
    equivalences = []
    print(f"\n[EFFICIENCY] Expert reference: {expert_mean:.1f}")
    print(f"[EFFICIENCY] PPO peak smoothed: {smoothed.max():.1f}")
    print(f"\n  {'N demos':>8}  {'BC reward':>10}  {'PPO steps to match':>22}  {'Steps/demo':>12}")
    print("  " + "-" * 60)
    for N, bc_r in zip(Ns, bc_means):
        # Each demo is ~860 steps (the avg trajectory length).
        # Total demo "interaction" is N * 860 steps from the EXPERT's data, but
        # they're offline — no env exploration. The PPO equivalence is in
        # online env steps regardless.
        ppo_steps = find_crossing(timesteps, smoothed, target=bc_r)
        if ppo_steps is None:
            equivalences.append({"N": int(N), "bc_reward": float(bc_r),
                                 "ppo_steps_to_match": None,
                                 "steps_per_demo": None})
            print(f"  {N:>8}  {bc_r:>10.1f}  {'never':>22}  {'-':>12}")
            continue
        steps_per_demo = ppo_steps / N
        equivalences.append({
            "N": int(N),
            "bc_reward": float(bc_r),
            "ppo_steps_to_match": float(ppo_steps),
            "steps_per_demo": float(steps_per_demo),
        })
        print(f"  {N:>8}  {bc_r:>10.1f}  {ppo_steps:>22,.0f}  {steps_per_demo:>12,.0f}")

    out_dir = RUNS
    out_path = out_dir / "data_efficiency_walker2d.json"
    out_path.write_text(json.dumps({
        "expert_mean": expert_mean,
        "ppo_timesteps": timesteps.tolist(),
        "ppo_smoothed_reward": smoothed.tolist(),
        "equivalences": equivalences,
    }, indent=2))
    print(f"\n[EFFICIENCY] Saved {out_path}")


if __name__ == "__main__":
    main()
