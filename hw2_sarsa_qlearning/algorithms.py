"""SARSA (on-policy TD control) and Q-Learning (off-policy TD control).

Both follow the same skeleton: epsilon-greedy behavior policy, single TD step
update, decaying epsilon. The only difference is the bootstrap target:

    SARSA      :  target = r + gamma * Q(s', a')        with a' ~ epsilon-greedy
    Q-Learning :  target = r + gamma * max_a Q(s', a)   (greedy w.r.t. current Q)

This is the textbook "horse vs. racehorse" comparison from S&B 6.4-6.5: SARSA
is conservative (learns the value of the policy it actually follows, including
exploration), Q-Learning is greedy (learns the optimal Q regardless of how it
explores). Cliff Walking is the canonical example where they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import gymnasium as gym


@dataclass
class TrainingResult:
    Q: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray


def _epsilon_greedy(Q: np.ndarray, s: int, eps: float, n_actions: int,
                    rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    qs = Q[s]
    candidates = np.flatnonzero(qs == qs.max())
    return int(rng.choice(candidates))


def _eps_schedule(ep: int, total: int, eps_start: float, eps_end: float,
                  decay_frac: float) -> float:
    decay_eps = max(1, int(total * decay_frac))
    frac = min(1.0, ep / decay_eps)
    return eps_start + frac * (eps_end - eps_start)


def sarsa(
    env: gym.Env,
    num_episodes: int = 5_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.8,
    max_steps: int = 1_000,
    seed: int = 0,
) -> TrainingResult:
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))

    returns = np.zeros(num_episodes, dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        eps = _eps_schedule(ep, num_episodes, eps_start, eps_end, decay_frac)
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        a = _epsilon_greedy(Q, s, eps, n_actions, rng)

        ep_ret, steps = 0.0, 0
        while steps < max_steps:
            s_next, r, done, truncated, _ = env.step(a)
            a_next = _epsilon_greedy(Q, s_next, eps, n_actions, rng)
            # SARSA target uses the action ACTUALLY chosen by the policy.
            target = r + gamma * Q[s_next, a_next] * (0.0 if done else 1.0)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s_next, a_next
            ep_ret += r
            steps += 1
            if done or truncated:
                break
        returns[ep] = ep_ret
        lengths[ep] = steps

    return TrainingResult(Q, returns, lengths)


def q_learning(
    env: gym.Env,
    num_episodes: int = 5_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.8,
    max_steps: int = 1_000,
    seed: int = 0,
) -> TrainingResult:
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))

    returns = np.zeros(num_episodes, dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        eps = _eps_schedule(ep, num_episodes, eps_start, eps_end, decay_frac)
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_ret, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            a = _epsilon_greedy(Q, s, eps, n_actions, rng)
            s_next, r, done, truncated, _ = env.step(a)
            # Q-learning target = greedy max, regardless of behavior policy.
            best_next = 0.0 if done else Q[s_next].max()
            target = r + gamma * best_next
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            ep_ret += r
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps

    return TrainingResult(Q, returns, lengths)


def evaluate_greedy(env: gym.Env, Q: np.ndarray, episodes: int = 200,
                    max_steps: int = 1_000, seed: int = 9999) -> tuple[float, float]:
    """Greedy rollout. Breaks ties uniformly at random to avoid degenerate loops
    when several actions still share the same (e.g. zero) Q-value."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes)
    for i in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_r, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            qs = Q[s]
            cand = np.flatnonzero(qs == qs.max())
            a = int(rng.choice(cand))
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
            steps += 1
        returns[i] = ep_r
        lengths[i] = steps
    return float(returns.mean()), float(lengths.mean())
