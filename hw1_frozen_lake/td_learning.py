"""TD(0) policy evaluation and TD-control (Q-learning style) on FrozenLake.

This file is intentionally kept distinct from HW2 so HW1 covers ONLY the basic
TD(0) update rule, which is the bridge between MC and full bootstrapping
methods covered in Lecture 4.

    TD(0) value update:   V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))

For control we use Q-learning's update (off-policy TD control) so the student
sees how TD(0) generalizes from prediction to control.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from .monte_carlo import epsilon_greedy


def td0_evaluation(
    env: gym.Env,
    policy: np.ndarray,
    num_episodes: int = 20_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: int = 0,
) -> np.ndarray:
    """Evaluate V^pi using TD(0). policy[s] -> action."""
    n_states = env.observation_space.n
    rng = np.random.default_rng(seed)
    V = np.zeros(n_states, dtype=np.float64)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done, truncated = False, False
        while not (done or truncated):
            action = int(policy[state])
            next_state, reward, done, truncated, _ = env.step(action)
            target = reward + gamma * V[next_state] * (0.0 if done else 1.0)
            V[state] += alpha * (target - V[state])
            state = next_state
    return V


def td_control(
    env: gym.Env,
    num_episodes: int = 50_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int | None = None,
    seed: int = 0,
):
    """Q-learning style TD control. Returns Q, returns, lengths."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = int(num_episodes * 0.8)

    returns_history = np.zeros(num_episodes, dtype=np.float32)
    lengths_history = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, epsilon_decay_episodes))
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        state, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done, truncated = False, False
        ep_return = 0.0
        steps = 0
        while not (done or truncated):
            action = epsilon_greedy(Q, state, epsilon, n_actions, rng)
            next_state, reward, done, truncated, _ = env.step(action)
            # Q-learning: bootstrap on max over next actions (off-policy).
            best_next = 0.0 if done else Q[next_state].max()
            target = reward + gamma * best_next
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
            ep_return += reward
            steps += 1
        returns_history[ep] = ep_return
        lengths_history[ep] = steps

    return Q, returns_history, lengths_history
