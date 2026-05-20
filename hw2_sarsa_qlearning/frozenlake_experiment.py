"""FrozenLake-v1 (slippery=True): the textbook environment for showing the
overestimation bias of Q-Learning vs the bias-correcting power of Double Q.

CliffWalking is deterministic — Q-Learning's overestimation only matters
because of ε-greedy exploration sampling. FrozenLake-v1 with is_slippery=True
has TRUE stochastic transitions (chosen action succeeds with p=1/3, slips
to perpendicular with p=2/3). This is where the max bias of Q-Learning
becomes a real bug.

We run all four algorithms on FrozenLake-v1 (4×4 map) and compare against
Value Iteration ground truth. Expected findings:
  - Q-Learning overestimates V* substantially (max bias)
  - Double Q-Learning corrects it
  - SARSA underestimates because exploration penalty is real (and slippery)
  - Expected SARSA is closer to ground truth than SARSA
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .algorithms import sarsa, q_learning
from .algorithms_extended import expected_sarsa, double_q_learning
from .value_iteration import value_iteration


ALGOS = {
    "sarsa": sarsa,
    "q_learning": q_learning,
    "expected_sarsa": expected_sarsa,
    "double_q": double_q_learning,
}


def main():
    ENV_ID = "FrozenLake-v1"
    SEEDS = [42, 123, 456, 789, 1000, 1234, 2024, 3141, 5678, 9999]
    NUM_EPISODES = 30_000  # Slippery FrozenLake needs more episodes to converge
    ALPHA = 0.1            # Lower alpha is needed for stochastic envs
    GAMMA = 0.99

    # Ground truth via value iteration.
    env = gym.make(ENV_ID, is_slippery=True)
    V_star, policy_star, Q_star = value_iteration(env, gamma=GAMMA)
    env.close()
    print(f"FrozenLake-v1 (slippery): V*(start=0) = {V_star[0]:.4f}")
    print(f"  optimal success prob ≈ {V_star[0]:.4f} (1 = solve every time)")

    payload = {
        "env_id": ENV_ID,
        "is_slippery": True,
        "seeds": np.array(SEEDS),
        "num_episodes": NUM_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "V_star": V_star,
        "Q_star": Q_star,
    }

    t0 = time.time()
    for algo_name, algo_fn in ALGOS.items():
        print(f"\n--- {algo_name} (slippery FrozenLake) ---")
        all_Q = []
        all_returns = []
        for seed in SEEDS:
            env = gym.make(ENV_ID, is_slippery=True)
            tic = time.time()
            res = algo_fn(env, num_episodes=NUM_EPISODES, alpha=ALPHA,
                          gamma=GAMMA, eps_start=1.0, eps_end=0.05,
                          decay_frac=0.8, seed=seed)
            env.close()
            all_Q.append(res.Q)
            all_returns.append(res.returns)
            final = float(res.returns[-1000:].mean())
            print(f"  seed={seed}  final success rate = {final:.3f}  "
                  f"({time.time() - tic:.1f}s)")
        Qs = np.stack(all_Q)
        rets = np.stack(all_returns)
        final_per_seed = rets[:, -1000:].mean(axis=1)
        payload[f"{algo_name}_Q"] = Qs
        payload[f"{algo_name}_returns"] = rets
        print(f"  → final success rate: {final_per_seed.mean():.3f} "
              f"± {final_per_seed.std(ddof=1):.3f}")

        # Bias against V* (only for non-terminal, non-hole cells)
        bias = Qs.max(axis=2).mean(axis=0) - V_star  # (S,)
        mean_bias = float(bias.mean())
        mean_abs_bias = float(np.abs(bias).mean())
        print(f"  → mean bias vs V*: {mean_bias:+.4f}  "
              f"(|bias| = {mean_abs_bias:.4f})")

    out_npz = Path("hw2_sarsa_qlearning/results/frozenlake.npz")
    np.savez_compressed(out_npz, **payload)
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
