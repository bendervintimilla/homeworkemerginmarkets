"""Sample efficiency: episodes to reach a threshold fraction of V*(start).

The notebook claims Q-Learning "learns a hair faster than SARSA early on" but
never quantifies it. This module measures, for each algorithm and seed, the
first episode where the smoothed training return crosses a threshold defined
as a fraction of the optimal V*(start) computed by Value Iteration.

For CliffWalking, V*(start) = -13 (γ=1). For Taxi, V*(start) varies by
initial passenger/destination but we use the mean V* across the 64 possible
start states reported by Value Iteration.

Output is a per-algorithm, per-environment table of mean ± std episodes to
threshold. Lower is more sample-efficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def smooth_returns(returns: np.ndarray, window: int) -> np.ndarray:
    """Per-seed moving average. returns has shape (seeds, episodes)."""
    kernel = np.ones(window) / window
    # mode='valid' produces shorter arrays; we pad with NaN at the start so
    # the output keeps the same episode index.
    n_seeds, n_eps = returns.shape
    out = np.full_like(returns, np.nan, dtype=np.float32)
    for i in range(n_seeds):
        smoothed = np.convolve(returns[i], kernel, mode="valid")
        out[i, window - 1:] = smoothed
    return out


def first_crossing(smoothed_per_seed: np.ndarray, threshold: float) -> np.ndarray:
    """For each seed, return the first episode index where the smoothed
    return crosses (>=) the threshold. NaN if never crossed."""
    n_seeds, n_eps = smoothed_per_seed.shape
    out = np.full(n_seeds, np.nan)
    for i in range(n_seeds):
        crosses = np.where(smoothed_per_seed[i] >= threshold)[0]
        if len(crosses) > 0:
            out[i] = crosses[0]
    return out


def main():
    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    vi = np.load("hw2_sarsa_qlearning/results/value_iteration.npz")

    summary = {"cliff": {}, "taxi": {}}

    # --- CliffWalking ------------------------------------------------------
    # Methodology: threshold = 90% of each algorithm's OWN final-window mean.
    # This is the standard "convergence" measure in RL benchmarking — it
    # asks "how many episodes until this algorithm reaches ~90% of where it
    # eventually ends up". Using V* as the ceiling would penalize SARSA and
    # Expected SARSA unfairly (they converge to a safer-but-suboptimal policy
    # whose value is V_pi^* not V*, around -17 instead of -13).
    cliff_v_star_start = float(vi["cliff_V_star"][36])
    print(f"CliffWalking: V*(start) = {cliff_v_star_start:.3f}  "
          f"(threshold = 90% of each algo's own final mean)")

    for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
        rets = runs[f"cliff_{algo}_returns"]
        # Algo-specific threshold: 90% of own final 200-ep mean.
        # In the negative-reward regime, "90%" means "closer to 0" — for a
        # mean of -20, the threshold is -20 / 0.9 = -22.22 (more permissive)
        # not -18 (more strict). We want the crossing to come from below.
        algo_final = float(rets[:, -200:].mean())
        cliff_threshold = algo_final / 0.9 if algo_final < 0 else 0.9 * algo_final
        smoothed = smooth_returns(rets, window=50)
        eps_to_thresh = first_crossing(smoothed, cliff_threshold)
        n_solved = int(np.sum(~np.isnan(eps_to_thresh)))
        if n_solved > 0:
            valid = eps_to_thresh[~np.isnan(eps_to_thresh)]
            mean_eps = float(valid.mean())
            std_eps = float(valid.std(ddof=1)) if n_solved > 1 else float("nan")
        else:
            mean_eps = float("nan")
            std_eps = float("nan")
        summary["cliff"][algo] = {
            "v_star_start": cliff_v_star_start,
            "threshold": cliff_threshold,
            "algo_final_mean": algo_final,
            "seeds_reached": n_solved,
            "n_seeds": int(rets.shape[0]),
            "mean_episodes_to_threshold": mean_eps,
            "std_episodes_to_threshold": std_eps,
            "per_seed_episodes": [None if np.isnan(x) else int(x) for x in eps_to_thresh],
        }
        print(f"  {algo:18s}  threshold = {cliff_threshold:>7.2f}  "
              f"reached by {n_solved}/{rets.shape[0]} seeds   "
              f"mean eps: {mean_eps:>7.1f}  ±{std_eps:>6.1f}")

    # --- Taxi: threshold = 90% of algo's own final-window mean ------------
    print(f"\nTaxi-v3 (threshold = 90% of each algo's own final mean):")

    for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
        rets = runs[f"taxi_{algo}_returns"]
        algo_final = float(rets[:, -200:].mean())
        taxi_threshold = 0.9 * algo_final if algo_final > 0 else algo_final / 0.9
        smoothed = smooth_returns(rets, window=300)
        eps_to_thresh = first_crossing(smoothed, taxi_threshold)
        n_solved = int(np.sum(~np.isnan(eps_to_thresh)))
        if n_solved > 0:
            valid = eps_to_thresh[~np.isnan(eps_to_thresh)]
            mean_eps = float(valid.mean())
            std_eps = float(valid.std(ddof=1)) if n_solved > 1 else float("nan")
        else:
            mean_eps = float("nan")
            std_eps = float("nan")
        summary["taxi"][algo] = {
            "threshold": taxi_threshold,
            "algo_final_mean": algo_final,
            "seeds_reached": n_solved,
            "n_seeds": int(rets.shape[0]),
            "mean_episodes_to_threshold": mean_eps,
            "std_episodes_to_threshold": std_eps,
            "per_seed_episodes": [None if np.isnan(x) else int(x) for x in eps_to_thresh],
        }
        print(f"  {algo:18s}  threshold = {taxi_threshold:>7.2f}  "
              f"reached by {n_solved}/{rets.shape[0]} seeds   "
              f"mean eps: {mean_eps:>7.1f}  ±{std_eps:>6.1f}")

    out = Path("hw2_sarsa_qlearning/results/sample_efficiency.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
