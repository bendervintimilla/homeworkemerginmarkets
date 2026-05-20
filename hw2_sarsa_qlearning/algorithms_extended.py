"""Extended TD-control algorithms covering Sutton & Barto Ch. 6.

In addition to the basic SARSA and Q-Learning (in `algorithms.py`), this
module adds the two natural next steps from the textbook:

- **Expected SARSA** (S&B §6.6): instead of bootstrapping on a single sampled
  action `Q(s', a')` (SARSA) or the max `max_a Q(s', a)` (Q-Learning), it uses
  the *expectation* over actions under the current ε-greedy policy:

        Q(s, a) ← Q(s, a) + α[r + γ · E_π[Q(s', a')] − Q(s, a)]

  Lower variance than SARSA (no single-action sample), unbiased under the
  current policy. Reduces to Q-Learning when the policy is greedy (ε=0).

- **Double Q-Learning** (S&B §6.7): the canonical max-bias issue in Q-Learning
  is that the same Q-values are used for both *selecting* the next action and
  *evaluating* it. Double Q-Learning maintains two independent estimators Q1,
  Q2 and decouples selection from evaluation:

        a* = argmax_a Q1(s', a)
        Q1(s, a) ← Q1(s, a) + α[r + γ · Q2(s', a*) − Q1(s, a)]

  (and symmetrically for Q2, flipping a coin at each step). Reduces the
  systematic overestimation that Q-Learning's `max` introduces in noisy
  environments.

Both algorithms reuse the ε-greedy and ε-schedule helpers from the basic
`algorithms` module so the only difference is the bootstrap target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from .algorithms import TrainingResult, _epsilon_greedy, _eps_schedule


# ----------------------------- Expected SARSA -----------------------------

def expected_sarsa(
    env: gym.Env,
    num_episodes: int = 10_000,
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
        while steps < max_steps:
            a = _epsilon_greedy(Q, s, eps, n_actions, rng)
            s_next, r, done, truncated, _ = env.step(a)
            # Expected SARSA: bootstrap on the EXPECTED next-state value under
            # the ε-greedy policy. Probability of each action:
            #   ε / n_actions  for non-greedy actions
            #   ε / n_actions + (1 - ε)  for the greedy action
            qs = Q[s_next]
            max_q = qs.max()
            greedy_mask = (qs == max_q)
            n_greedy = greedy_mask.sum()
            probs = np.full(n_actions, eps / n_actions)
            probs[greedy_mask] += (1 - eps) / n_greedy
            expected_q = float((probs * qs).sum())
            target = r + gamma * expected_q * (0.0 if done else 1.0)
            Q[s, a] += alpha * (target - Q[s, a])      # <-- EXPECTED SARSA UPDATE
            s = s_next
            ep_ret += r
            steps += 1
            if done or truncated:
                break
        returns[ep] = ep_ret
        lengths[ep] = steps
    return TrainingResult(Q, returns, lengths)


# ----------------------------- Double Q-Learning -----------------------------

@dataclass
class DoubleQResult:
    """Double Q-Learning returns two Q-tables. Greedy action is argmax of their
    SUM (per S&B §6.7). We also expose the average so existing helpers that
    expect a single Q matrix can be used by reading `Q`."""
    Q1: np.ndarray
    Q2: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray

    @property
    def Q(self) -> np.ndarray:
        # The behaviour policy and evaluation policy use Q1 + Q2.
        return self.Q1 + self.Q2


def double_q_learning(
    env: gym.Env,
    num_episodes: int = 10_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.8,
    max_steps: int = 1_000,
    seed: int = 0,
) -> DoubleQResult:
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    returns = np.zeros(num_episodes, dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        eps = _eps_schedule(ep, num_episodes, eps_start, eps_end, decay_frac)
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_ret, steps = 0.0, 0
        while steps < max_steps:
            # Behaviour policy is ε-greedy w.r.t. Q1 + Q2 (per S&B §6.7).
            qs_sum = Q1[s] + Q2[s]
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                candidates = np.flatnonzero(qs_sum == qs_sum.max())
                a = int(rng.choice(candidates))

            s_next, r, done, truncated, _ = env.step(a)

            # Flip a coin: update Q1 using Q2 as the evaluator, or vice versa.
            if rng.random() < 0.5:
                # Update Q1. Selection uses Q1 (argmax), evaluation uses Q2.
                a_star = int(np.argmax(Q1[s_next]))
                target = r + gamma * Q2[s_next, a_star] * (0.0 if done else 1.0)
                Q1[s, a] += alpha * (target - Q1[s, a])
            else:
                a_star = int(np.argmax(Q2[s_next]))
                target = r + gamma * Q1[s_next, a_star] * (0.0 if done else 1.0)
                Q2[s, a] += alpha * (target - Q2[s, a])

            s = s_next
            ep_ret += r
            steps += 1
            if done or truncated:
                break
        returns[ep] = ep_ret
        lengths[ep] = steps
    return DoubleQResult(Q1, Q2, returns, lengths)
