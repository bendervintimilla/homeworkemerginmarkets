"""Reproduce Sutton & Barto Figure 6.4 verbatim.

Spec from the book (page 132 of 2nd edition):
  - Environment: CliffWalking
  - Algorithms: SARSA and Q-Learning
  - α = 0.5
  - γ = 1.0 (undiscounted, episodic)
  - ε = 0.1 (fixed)
  - 500 episodes
  - Curves averaged over 100 independent runs

Expected result (from the book):
  - SARSA   asymptotes around  -17 reward per episode
  - Q-Learn asymptotes around  -25 reward per episode

The narrative: Q-Learning learns the optimal greedy policy (cliff edge), but
its training reward is worse because while exploring with ε=0.1 it
occasionally falls into the cliff. SARSA learns a safer policy further from
the cliff, gives up some optimality, gains stability.
"""

from __future__ import annotations

import argparse
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


def _epsilon_greedy(Q, s, eps, n_actions, rng):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    qs = Q[s]
    candidates = np.flatnonzero(qs == qs.max())
    return int(rng.choice(candidates))


def sarsa_fig64(env, num_episodes, alpha, gamma, eps, max_steps, seed):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        a = _epsilon_greedy(Q, s, eps, n_actions, rng)
        ep_ret, steps = 0.0, 0
        while steps < max_steps:
            s_next, r, done, trunc, _ = env.step(a)
            a_next = _epsilon_greedy(Q, s_next, eps, n_actions, rng)
            target = r + gamma * Q[s_next, a_next] * (0.0 if done else 1.0)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s_next, a_next
            ep_ret += r
            steps += 1
            if done or trunc:
                break
        returns[ep] = ep_ret
    return returns


def ql_fig64(env, num_episodes, alpha, gamma, eps, max_steps, seed):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_ret, steps = 0.0, 0
        while steps < max_steps:
            a = _epsilon_greedy(Q, s, eps, n_actions, rng)
            s_next, r, done, trunc, _ = env.step(a)
            target = r + gamma * Q[s_next].max() * (0.0 if done else 1.0)
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            ep_ret += r
            steps += 1
            if done or trunc:
                break
        returns[ep] = ep_ret
    return returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--out", default="hw2_sarsa_qlearning/results/figure_6_4.npz")
    args = p.parse_args()

    CLIFF = _resolve("CliffWalking", ["v0", "v1"])
    print(f"[F6.4] Env: {CLIFF}")
    print(f"[F6.4] {args.seeds} seeds × {args.episodes} episodes")
    print(f"[F6.4] alpha={args.alpha}, gamma={args.gamma}, eps={args.eps} (fixed)")

    sarsa_runs = np.zeros((args.seeds, args.episodes))
    ql_runs = np.zeros((args.seeds, args.episodes))

    t0 = time.time()
    for i, seed in enumerate(range(args.seeds)):
        env = gym.make(CLIFF)
        sarsa_runs[i] = sarsa_fig64(env, args.episodes, args.alpha, args.gamma,
                                     args.eps, args.max_steps, seed)
        env.close()
        env = gym.make(CLIFF)
        ql_runs[i] = ql_fig64(env, args.episodes, args.alpha, args.gamma,
                              args.eps, args.max_steps, seed + 10_000)
        env.close()
        if (i + 1) % 10 == 0:
            print(f"  seed {i+1}/{args.seeds}  elapsed {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"\n[F6.4] Done in {elapsed:.1f}s")

    sarsa_mean = sarsa_runs.mean(axis=0)
    ql_mean = ql_runs.mean(axis=0)
    print(f"\n[F6.4] Asymptote (mean over last 50 episodes, averaged over seeds):")
    print(f"  SARSA      : {sarsa_runs[:, -50:].mean():.2f}  (S&B says ≈ -17)")
    print(f"  Q-Learning : {ql_runs[:, -50:].mean():.2f}  (S&B says ≈ -25)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, sarsa_runs=sarsa_runs, ql_runs=ql_runs,
                        sarsa_mean=sarsa_mean, ql_mean=ql_mean,
                        n_seeds=args.seeds, n_episodes=args.episodes,
                        alpha=args.alpha, gamma=args.gamma, eps=args.eps)
    print(f"\n[F6.4] Saved {out}")


if __name__ == "__main__":
    main()
