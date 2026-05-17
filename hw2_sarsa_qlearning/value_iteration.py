"""Exact MDP solution via Value Iteration.

Both Taxi-v3 and CliffWalking-v0 expose the full transition table at
`env.unwrapped.P` as a dict:

    P[state][action] = list of (prob, next_state, reward, terminal) tuples

That is the model. We use it to compute V*(s) and π*(s) exactly (no sampling)
via the Bellman optimality equation:

    V*(s) = max_a Σ_{s', r} P(s', r | s, a) · [r + γ · V*(s')]   if non-terminal
    V*(s) = 0                                                     if terminal

Why we need this:
- Gives a **ground-truth optimum** for tabular methods.
- Lets us measure each algorithm's **regret** vs the true optimum.
- Lets us measure each algorithm's **overestimation bias**:
      bias(s) = max_a Q_algo(s, a) − V*(s)
  which reproduces S&B Figure 6.5 for Q-Learning and shows Double Q-Learning
  corrects it.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def value_iteration(env: gym.Env, gamma: float = 0.99, tol: float = 1e-9,
                    max_iters: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (V*, π*) with V shape (n_states,) and π shape (n_states,) of ints.
    Uses the env's true transition model `env.unwrapped.P`.
    """
    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)

    for it in range(max_iters):
        delta = 0.0
        V_new = V.copy()
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, s_next, r, terminal in P[s][a]:
                    bootstrap = 0.0 if terminal else V[s_next]
                    action_values[a] += prob * (r + gamma * bootstrap)
            v_star = action_values.max()
            delta = max(delta, abs(v_star - V[s]))
            V_new[s] = v_star
        V = V_new
        if delta < tol:
            break

    # Derive deterministic optimal policy from V*.
    policy = np.zeros(n_states, dtype=np.int64)
    Q_star = np.zeros((n_states, n_actions))
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, s_next, r, terminal in P[s][a]:
                bootstrap = 0.0 if terminal else V[s_next]
                action_values[a] += prob * (r + gamma * bootstrap)
        policy[s] = int(np.argmax(action_values))
        Q_star[s] = action_values
    return V, policy, Q_star


def solve_env(env_id: str, gamma: float = 0.99) -> dict:
    env = gym.make(env_id)
    V, policy, Q_star = value_iteration(env, gamma=gamma)
    env.close()
    return {"env_id": env_id, "gamma": gamma, "V_star": V, "policy_star": policy,
            "Q_star": Q_star}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="hw2_sarsa_qlearning/results/value_iteration.npz")
    p.add_argument("--gamma", type=float, default=0.99)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve env ids using same logic as the rest of HW2
    def _resolve(base, versions):
        for v in versions:
            env_id = f"{base}-{v}"
            try:
                gym.spec(env_id)
                return env_id
            except Exception:
                continue
        raise RuntimeError(f"Cannot find {base}")

    TAXI = _resolve("Taxi", ["v3", "v4"])
    CLIFF = _resolve("CliffWalking", ["v0", "v1"])

    print(f"Solving {TAXI} with gamma={args.gamma}...")
    taxi_sol = solve_env(TAXI, gamma=args.gamma)
    print(f"  V* range: [{taxi_sol['V_star'].min():.2f}, {taxi_sol['V_star'].max():.2f}]")
    print(f"  V*(start states, sample mean): {taxi_sol['V_star'].mean():.2f}")

    print(f"\nSolving {CLIFF} with gamma={args.gamma}...")
    cliff_sol = solve_env(CLIFF, gamma=args.gamma)
    print(f"  V* range: [{cliff_sol['V_star'].min():.2f}, {cliff_sol['V_star'].max():.2f}]")
    print(f"  V*(0) [start]: {cliff_sol['V_star'][36]:.2f}  (optimal undiscounted = -13)")

    np.savez(
        out,
        taxi_env_id=TAXI,
        taxi_V_star=taxi_sol["V_star"],
        taxi_policy_star=taxi_sol["policy_star"],
        taxi_Q_star=taxi_sol["Q_star"],
        cliff_env_id=CLIFF,
        cliff_V_star=cliff_sol["V_star"],
        cliff_policy_star=cliff_sol["policy_star"],
        cliff_Q_star=cliff_sol["Q_star"],
        gamma=args.gamma,
    )
    print(f"\nSaved {out}")
