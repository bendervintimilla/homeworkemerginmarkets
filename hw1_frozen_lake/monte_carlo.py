"""First-visit Monte Carlo control with epsilon-greedy policy on FrozenLake.

Implements the algorithm from Sutton & Barto Ch. 5.4 ("On-policy first-visit
MC control for epsilon-soft policies"):

    1. Generate an episode following pi (epsilon-soft).
    2. For each (s, a) appearing in the episode, compute G = sum gamma^k r_{t+k+1}
       starting from its FIRST visit.
    3. Update Q(s, a) = average of all returns observed for that (s, a).
    4. Improve the policy: greedy w.r.t. Q with exploration prob epsilon.

We keep an explicit "Returns" counter (incremental average) so memory is O(|S||A|)
instead of storing every observed return.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float, n_actions: int,
                   rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    # Tie-breaking: random among argmax actions to avoid getting stuck on the
    # first action when Q is still zero everywhere.
    qs = Q[state]
    max_q = qs.max()
    candidates = np.flatnonzero(qs == max_q)
    return int(rng.choice(candidates))


def mc_control(
    env: gym.Env,
    num_episodes: int = 50_000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int | None = None,
    seed: int = 0,
):
    """First-visit MC control. Returns Q, episode returns, episode lengths."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)

    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    counts = np.zeros((n_states, n_actions), dtype=np.int64)

    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = int(num_episodes * 0.8)

    returns_history = np.zeros(num_episodes, dtype=np.float32)
    lengths_history = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        # Linear epsilon schedule: smoother than exponential decay for FrozenLake.
        frac = min(1.0, ep / max(1, epsilon_decay_episodes))
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        state, _ = env.reset(seed=int(rng.integers(1 << 31)))
        episode = []  # list of (s, a, r)
        done = False
        truncated = False
        ep_return = 0.0
        steps = 0

        while not (done or truncated):
            action = epsilon_greedy(Q, state, epsilon, n_actions, rng)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            ep_return += reward
            state = next_state
            steps += 1

        returns_history[ep] = ep_return
        lengths_history[ep] = steps

        # First-visit MC: compute G backwards, only update on the FIRST occurrence
        # of (s, a) in the trajectory.
        G = 0.0
        seen = set()
        # Iterate trajectory in reverse to compute G incrementally.
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in seen:
                seen.add((s, a))
                # We will apply the update only if this is the first visit;
                # because we go in reverse and check earliest occurrence later,
                # we need to invert: we set the value to G of the FIRST (earliest)
                # visit. Easier: iterate forward and remember first index.
                pass

        # Forward pass to apply first-visit updates with already-computed G's.
        # Since Sutton's pseudo-code uses a backward pass with first-visit check
        # via a "seen" set, we redo it forward with precomputed Gs from the back.
        Gs = np.zeros(len(episode))
        running = 0.0
        for t in range(len(episode) - 1, -1, -1):
            running = gamma * running + episode[t][2]
            Gs[t] = running
        seen = set()
        for t, (s, a, _r) in enumerate(episode):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s, a] += 1
            # Incremental mean: Q <- Q + (G - Q) / N
            Q[s, a] += (Gs[t] - Q[s, a]) / counts[s, a]

    return Q, returns_history, lengths_history


def derive_policy(Q: np.ndarray) -> np.ndarray:
    return Q.argmax(axis=1)


def state_value_from_q(Q: np.ndarray) -> np.ndarray:
    return Q.max(axis=1)
