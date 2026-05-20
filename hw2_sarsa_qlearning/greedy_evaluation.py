"""Greedy-policy evaluation + random-policy baseline.

The PDF explicitly requires:
  1. "Compare the performance of the learned policy to a baseline
      (e.g., random actions)."
  2. "Test the learned policy in the environment and measure
      performance metrics such as total reward and number of steps
      to reach the goal."

This module produces both, for SARSA / Q-Learning / Expected SARSA /
Double Q-Learning, on Taxi-v3 and CliffWalking-v1. Each algorithm's
greedy policy is rolled out for 200 episodes per seed across the 10
seeds used in training. We report mean reward AND mean steps.

The random baseline is also evaluated for 200 episodes × 10 seeds for
a fair comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


def _resolve(base, versions):
    for v in versions:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"Cannot find {base}")


def random_rollout(env_id: str, n_episodes: int, max_steps: int,
                   seed: int) -> tuple[float, float, float]:
    """Random policy baseline. Returns (mean_reward, mean_steps, mean_success)."""
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    rewards = []
    lengths = []
    successes = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_r, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            a = int(rng.integers(n_actions))
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
            steps += 1
        rewards.append(ep_r)
        lengths.append(steps)
        successes.append(1 if done else 0)
    env.close()
    return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(successes))


def greedy_rollout(env_id: str, Q: np.ndarray, n_episodes: int, max_steps: int,
                   seed: int) -> tuple[float, float, float]:
    """Greedy policy rollout. Returns (mean_reward, mean_steps, mean_success).
    Ties in Q are broken uniformly at random to avoid degenerate loops."""
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    rewards = []
    lengths = []
    successes = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_r, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            qs = Q[s]
            cands = np.flatnonzero(qs == qs.max())
            a = int(rng.choice(cands))
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
            steps += 1
        rewards.append(ep_r)
        lengths.append(steps)
        successes.append(1 if done else 0)
    env.close()
    return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(successes))


def main():
    TAXI = _resolve("Taxi", ["v3", "v4"])
    CLIFF = _resolve("CliffWalking", ["v0", "v1"])

    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    n_seeds = int(len(runs["seeds"]))

    summary = {"taxi": {}, "cliff": {}}
    t0 = time.time()

    # -------------------------------------------------------------------
    # Random baseline
    # -------------------------------------------------------------------
    print("=" * 70)
    print("RANDOM POLICY BASELINE")
    print("=" * 70)
    for env_label, env_id, max_steps in [
        ("taxi", TAXI, 200),
        ("cliff", CLIFF, 500),
    ]:
        all_r, all_l, all_s = [], [], []
        for seed in runs["seeds"]:
            r, l, s = random_rollout(env_id, n_episodes=200,
                                      max_steps=max_steps, seed=int(seed))
            all_r.append(r)
            all_l.append(l)
            all_s.append(s)
        summary[env_label]["random"] = {
            "reward_mean": float(np.mean(all_r)),
            "reward_std": float(np.std(all_r, ddof=1)),
            "steps_mean": float(np.mean(all_l)),
            "steps_std": float(np.std(all_l, ddof=1)),
            "success_rate_mean": float(np.mean(all_s)),
            "success_rate_std": float(np.std(all_s, ddof=1)),
        }
        s = summary[env_label]["random"]
        print(f"  {env_label:<8s}  reward={s['reward_mean']:+8.2f} ±{s['reward_std']:5.2f}  "
              f"steps={s['steps_mean']:6.1f} ±{s['steps_std']:5.1f}  "
              f"success={s['success_rate_mean']*100:5.1f}%")

    # -------------------------------------------------------------------
    # Greedy rollouts of each trained algorithm
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    print("GREEDY ROLLOUTS OF TRAINED POLICIES")
    print("=" * 70)
    for env_label, env_id, max_steps in [
        ("taxi", TAXI, 200),
        ("cliff", CLIFF, 200),
    ]:
        print(f"\n[{env_label.upper()}]")
        for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
            Q_all = runs[f"{env_label}_{algo}_Q"]  # (seeds, S, A)
            all_r, all_l, all_s = [], [], []
            for i, seed in enumerate(runs["seeds"]):
                Q = Q_all[i]
                # For double_q, Q is already Q1+Q2 (sum); argmax-equivalent to avg
                r, l, s = greedy_rollout(env_id, Q, n_episodes=200,
                                          max_steps=max_steps, seed=int(seed) + 100000)
                all_r.append(r)
                all_l.append(l)
                all_s.append(s)
            summary[env_label][algo] = {
                "reward_mean": float(np.mean(all_r)),
                "reward_std": float(np.std(all_r, ddof=1)),
                "steps_mean": float(np.mean(all_l)),
                "steps_std": float(np.std(all_l, ddof=1)),
                "success_rate_mean": float(np.mean(all_s)),
                "success_rate_std": float(np.std(all_s, ddof=1)),
            }
            s = summary[env_label][algo]
            print(f"  {algo:<18s}  reward={s['reward_mean']:+8.2f} ±{s['reward_std']:5.2f}  "
                  f"steps={s['steps_mean']:6.1f} ±{s['steps_std']:5.1f}  "
                  f"success={s['success_rate_mean']*100:5.1f}%")

    out = Path("hw2_sarsa_qlearning/results/greedy_evaluation.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
