"""n-step SARSA from Sutton & Barto Chapter 7.

The 1-step methods in `algorithms.py` (regular SARSA, Q-Learning) bootstrap
after a SINGLE environment step. They have low variance but high bias from
the bootstrap. Monte Carlo (n→∞) is the other extreme: zero bias but high
variance from full-return sampling.

n-step SARSA interpolates: bootstrap after n steps instead of 1.

Update rule (S&B Eq. 7.5):

    G_t:t+n = r_{t+1} + γ·r_{t+2} + … + γ^{n-1}·r_{t+n} + γ^n·Q(S_{t+n}, A_{t+n})

    Q(S_t, A_t) ← Q(S_t, A_t) + α · [G_t:t+n − Q(S_t, A_t)]

The bias-variance trade-off as n grows:
  - n=1:    SARSA (low variance, max bias from bootstrap)
  - n=2-8:  sweet spot for many problems — bridges sample efficiency
  - n→∞:    Monte Carlo (zero bias, max variance)

WHY this matters here: CliffWalking is the textbook case where n>1 matters.
The cliff-fall reward of −100 propagates back faster with larger n, so the
agent learns to avoid the cliff edge in fewer episodes.

IMPLEMENTATION NOTE
===================
At episode end, the buffer of partial returns has to be drained: the LAST
n−1 transitions can't bootstrap n steps ahead because the episode is over.
S&B handles this by treating "beyond terminal" as zero bootstrap.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym

from .algorithms import TrainingResult, _epsilon_greedy, _eps_schedule


def n_step_sarsa(
    env: gym.Env,
    n: int = 4,
    num_episodes: int = 10_000,
    alpha: float = 0.5,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.8,
    max_steps: int = 1_000,
    seed: int = 0,
) -> TrainingResult:
    """n-step on-policy SARSA. Follows S&B §7.2 pseudocode literally.

    Sliding window of size n holds the (state, action, reward) triples needed
    to compute the n-step return. We update Q for the state n steps in the
    past, using the rewards observed since plus the bootstrap Q at the current
    state-action.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))

    returns = np.zeros(num_episodes, dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)
    gamma_powers = gamma ** np.arange(n + 1)  # precompute γ^0, γ^1, …, γ^n

    for ep in range(num_episodes):
        eps = _eps_schedule(ep, num_episodes, eps_start, eps_end, decay_frac)
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        a = _epsilon_greedy(Q, s, eps, n_actions, rng)

        # Sliding buffers, indexed by t. T = step at which episode terminates.
        S = [s]                # S[0], S[1], …
        A = [a]
        R = [0.0]              # R[0] unused; rewards start at R[1]
        T = float("inf")
        t = 0
        ep_ret = 0.0

        # tau = the time whose Q value we update at this iteration.
        # We stop when tau = T - 1 (last update of the episode).
        while True:
            if t < T:
                s_next, r, done, truncated, _ = env.step(A[t])
                R.append(float(r))
                S.append(int(s_next))
                ep_ret += r
                if done or truncated or t + 1 >= max_steps:
                    T = t + 1
                else:
                    a_next = _epsilon_greedy(Q, s_next, eps, n_actions, rng)
                    A.append(a_next)

            tau = t - n + 1
            if tau >= 0:
                # n-step return: sum of discounted rewards from tau+1 to
                # min(tau+n, T), then bootstrap on Q(S_{tau+n}, A_{tau+n})
                # if tau+n < T.
                end = min(tau + n, T)
                # Indices into R: rewards from index tau+1 to end (inclusive).
                G = 0.0
                for i in range(tau + 1, int(end) + 1):
                    G += gamma_powers[i - tau - 1] * R[i]
                if tau + n < T:
                    G += gamma_powers[n] * Q[S[tau + n], A[tau + n]]
                # TD update on Q[S[tau], A[tau]].
                Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])

            if tau == T - 1:
                break
            t += 1

        returns[ep] = ep_ret
        lengths[ep] = T if T != float("inf") else max_steps

    return TrainingResult(Q, returns, lengths)


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------

def main():
    import time
    import json
    from pathlib import Path

    # Use the same resolver as run_4algos.py to handle CliffWalking-v0/v1.
    def _resolve(base, versions):
        for v in versions:
            env_id = f"{base}-{v}"
            try:
                gym.spec(env_id)
                return env_id
            except Exception:
                continue
        raise RuntimeError(f"Cannot find {base}")
    CLIFF = _resolve("CliffWalking", ["v0", "v1"])
    SEEDS = [42, 123, 456, 789, 1000]
    NUM_EPISODES = 10_000
    N_VALUES = [1, 4, 16]
    ALPHA = 0.5
    GAMMA = 0.99

    results = {}
    all_n_returns = {}  # n -> (seeds, episodes) array
    t0 = time.time()
    print(f"n-step SARSA on {CLIFF}: n ∈ {N_VALUES}, "
          f"{len(SEEDS)} seeds × {NUM_EPISODES} eps")

    for n in N_VALUES:
        per_n_returns = []
        for seed in SEEDS:
            env = gym.make(CLIFF)
            tic = time.time()
            res = n_step_sarsa(env, n=n, num_episodes=NUM_EPISODES,
                                alpha=ALPHA, gamma=GAMMA, seed=seed)
            env.close()
            per_n_returns.append(res.returns)
            elapsed = time.time() - tic
            final = float(res.returns[-200:].mean())
            print(f"  n={n:>2d}  seed={seed}  final={final:+7.2f}  ({elapsed:.1f}s)")
        rets = np.stack(per_n_returns)  # (seeds, episodes)
        all_n_returns[n] = rets
        final_per_seed = rets[:, -200:].mean(axis=1)
        results[n] = {
            "n": n,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "seeds": SEEDS,
            "final_mean": float(final_per_seed.mean()),
            "final_std": float(final_per_seed.std(ddof=1)),
            "per_seed_final": final_per_seed.tolist(),
        }
        print(f"  → n={n:>2d}: {results[n]['final_mean']:+7.2f} "
              f"± {results[n]['final_std']:5.2f}")

    # Save all returns for plotting
    out_npz = Path("hw2_sarsa_qlearning/results/n_step_sarsa.npz")
    payload = {
        "n_values": np.array(N_VALUES),
        "seeds": np.array(SEEDS),
        "num_episodes": NUM_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
    }
    for n in N_VALUES:
        payload[f"n{n}_returns"] = all_n_returns[n]
    np.savez_compressed(out_npz, **payload)
    out_json = Path("hw2_sarsa_qlearning/results/n_step_sarsa.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
