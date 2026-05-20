"""Generic Monte Carlo first-visit / every-visit control for tabular envs.

Works on Taxi, FrozenLake, Volcano (any env with Discrete obs and action spaces).

The HW1 spec requests both first-visit and every-visit MC; the function
exposes a `mode` flag so the notebook can demonstrate both. The default is
first-visit because it's what Sutton & Barto recommend for episodic envs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    cand = np.flatnonzero(qs == qs.max())
    return int(rng.choice(cand))


def mc_control(
    env: gym.Env,
    num_episodes: int = 50_000,
    gamma: float = 0.99,
    alpha: float | None = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.8,
    max_steps: int = 1_000,
    mode: Literal["first_visit", "every_visit"] = "first_visit",
    seed: int = 0,
) -> TrainingResult:
    """Generic on-policy MC control.

    If `alpha is None`, uses incremental sample-mean (Q += (G-Q)/N(s,a)).
    Otherwise uses constant-alpha update Q += alpha * (G - Q), which is what
    the HW1 spec writes: Q(s,a) <- Q(s,a) + alpha * [G_t - Q(s,a)].
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)

    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    counts = np.zeros((n_states, n_actions), dtype=np.int64)

    decay_eps = max(1, int(num_episodes * epsilon_decay_frac))
    returns = np.zeros(num_episodes, dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        frac = min(1.0, ep / decay_eps)
        eps = epsilon_start + frac * (epsilon_end - epsilon_start)

        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        episode = []
        ep_ret, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            a = _epsilon_greedy(Q, s, eps, n_actions, rng)
            s_next, r, done, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s_next
            ep_ret += r
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps

        # Compute discounted returns G_t back-to-front in a single pass.
        Gs = np.zeros(len(episode))
        running = 0.0
        for t in range(len(episode) - 1, -1, -1):
            running = gamma * running + episode[t][2]
            Gs[t] = running

        # Apply MC update.
        if mode == "first_visit":
            seen: set[tuple[int, int]] = set()
            for t, (s_t, a_t, _) in enumerate(episode):
                key = (s_t, a_t)
                if key in seen:
                    continue
                seen.add(key)
                counts[s_t, a_t] += 1
                if alpha is None:
                    Q[s_t, a_t] += (Gs[t] - Q[s_t, a_t]) / counts[s_t, a_t]
                else:
                    Q[s_t, a_t] += alpha * (Gs[t] - Q[s_t, a_t])
        else:  # every_visit
            for t, (s_t, a_t, _) in enumerate(episode):
                counts[s_t, a_t] += 1
                if alpha is None:
                    Q[s_t, a_t] += (Gs[t] - Q[s_t, a_t]) / counts[s_t, a_t]
                else:
                    Q[s_t, a_t] += alpha * (Gs[t] - Q[s_t, a_t])

    return TrainingResult(Q, returns, lengths)


def evaluate_greedy(env: gym.Env, Q: np.ndarray, episodes: int = 500,
                    max_steps: int = 1_000, seed: int = 9999) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rets = np.zeros(episodes)
    lens = np.zeros(episodes)
    for i in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_r, steps, done, truncated = 0.0, 0, False, False
        while not (done or truncated) and steps < max_steps:
            qs = Q[s]
            cand = np.flatnonzero(qs == qs.max())
            a = int(rng.choice(cand))
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
            steps += 1
        rets[i] = ep_r
        lens[i] = steps
    return float(rets.mean()), float(lens.mean())
