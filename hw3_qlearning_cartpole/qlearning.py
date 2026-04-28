"""Tabular Q-Learning on a discretized CartPole state."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .discretizer import CartPoleDiscretizer


@dataclass
class TrainingResult:
    Q: np.ndarray
    returns: np.ndarray
    discretizer: CartPoleDiscretizer


def epsilon_greedy(Q: np.ndarray, s: int, eps: float, n_actions: int,
                   rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    qs = Q[s]
    cand = np.flatnonzero(qs == qs.max())
    return int(rng.choice(cand))


def train(
    env: gym.Env,
    discretizer: CartPoleDiscretizer,
    num_episodes: int = 5_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.6,
    seed: int = 0,
) -> TrainingResult:
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((discretizer.n_states, n_actions), dtype=np.float64)
    returns = np.zeros(num_episodes, dtype=np.float32)

    decay_eps = max(1, int(num_episodes * decay_frac))
    for ep in range(num_episodes):
        eps = eps_start + min(1.0, ep / decay_eps) * (eps_end - eps_start)
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        s = discretizer(obs)
        ep_ret, done, truncated = 0.0, False, False
        while not (done or truncated):
            a = epsilon_greedy(Q, s, eps, n_actions, rng)
            obs_next, r, done, truncated, _ = env.step(a)
            s_next = discretizer(obs_next)
            best_next = 0.0 if done else Q[s_next].max()
            target = r + gamma * best_next
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            ep_ret += r
        returns[ep] = ep_ret

    return TrainingResult(Q, returns, discretizer)


def evaluate(env: gym.Env, Q: np.ndarray, discretizer: CartPoleDiscretizer,
             episodes: int = 50, seed: int = 9999) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rets = np.zeros(episodes)
    lens = np.zeros(episodes)
    for i in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        s = discretizer(obs)
        ep_r, steps, done, truncated = 0.0, 0, False, False
        while not (done or truncated):
            qs = Q[s]
            cand = np.flatnonzero(qs == qs.max())
            a = int(rng.choice(cand))
            obs, r, done, truncated, _ = env.step(a)
            s = discretizer(obs)
            ep_r += r
            steps += 1
        rets[i] = ep_r
        lens[i] = steps
    return float(rets.mean()), float(lens.mean())
