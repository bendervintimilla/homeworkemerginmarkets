"""Train 4 algorithms × 10 seeds × 2 environments.

Algorithms: SARSA, Q-Learning, Expected SARSA, Double Q-Learning.
Environments: Taxi-v3 (20k episodes), CliffWalking-v0 (10k episodes).

Saves everything (Q-tables, learning curves, lengths) to a single npz file
that the notebook + bias analysis + video renderer all read from.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .algorithms import sarsa, q_learning
from .algorithms_extended import expected_sarsa, double_q_learning


ALGOS = {
    "sarsa": sarsa,
    "q_learning": q_learning,
    "expected_sarsa": expected_sarsa,
    "double_q": double_q_learning,
}


def _resolve(base, versions):
    for v in versions:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"Cannot find {base}")


def train_one(algo_name: str, env_id: str, num_episodes: int, alpha: float,
              gamma: float, seed: int) -> dict:
    env = gym.make(env_id)
    fn = ALGOS[algo_name]
    res = fn(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, seed=seed)
    env.close()
    Q = res.Q  # works for both TrainingResult and DoubleQResult (via property)
    return {"Q": Q, "returns": res.returns, "lengths": res.lengths}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="hw2_sarsa_qlearning/results/four_algos.npz")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1000, 1234, 2024, 3141, 5678, 9999])
    p.add_argument("--taxi-eps", type=int, default=20_000)
    p.add_argument("--cliff-eps", type=int, default=10_000)
    p.add_argument("--taxi-alpha", type=float, default=0.1)
    p.add_argument("--cliff-alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    args = p.parse_args()

    TAXI = _resolve("Taxi", ["v3", "v4"])
    CLIFF = _resolve("CliffWalking", ["v0", "v1"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "taxi_env_id": TAXI,
        "cliff_env_id": CLIFF,
        "seeds": np.array(args.seeds),
        "taxi_episodes": args.taxi_eps,
        "cliff_episodes": args.cliff_eps,
        "taxi_alpha": args.taxi_alpha,
        "cliff_alpha": args.cliff_alpha,
        "gamma": args.gamma,
    }

    print(f"[RUN4] Algorithms: {list(ALGOS.keys())}")
    print(f"[RUN4] Seeds: {args.seeds} (n={len(args.seeds)})")
    print(f"[RUN4] Taxi: {TAXI}, {args.taxi_eps} eps, alpha={args.taxi_alpha}")
    print(f"[RUN4] Cliff: {CLIFF}, {args.cliff_eps} eps, alpha={args.cliff_alpha}")

    grand_t0 = time.time()

    for env_label, env_id, num_eps, alpha in [
        ("taxi", TAXI, args.taxi_eps, args.taxi_alpha),
        ("cliff", CLIFF, args.cliff_eps, args.cliff_alpha),
    ]:
        for algo in ALGOS:
            print(f"\n[RUN4] === {env_label} / {algo} ({len(args.seeds)} seeds × {num_eps} eps) ===")
            t0 = time.time()
            all_Q = []
            all_returns = []
            all_lengths = []
            for seed in args.seeds:
                res = train_one(algo, env_id, num_eps, alpha, args.gamma, seed)
                all_Q.append(res["Q"])
                all_returns.append(res["returns"])
                all_lengths.append(res["lengths"])
            elapsed = time.time() - t0
            Qs = np.stack(all_Q)        # (seeds, S, A)
            rets = np.stack(all_returns) # (seeds, N)
            lens = np.stack(all_lengths) # (seeds, N)
            payload[f"{env_label}_{algo}_Q"] = Qs
            payload[f"{env_label}_{algo}_returns"] = rets
            payload[f"{env_label}_{algo}_lengths"] = lens
            print(f"[RUN4]   trained {len(args.seeds)} seeds in {elapsed:.1f}s "
                  f"({elapsed/len(args.seeds):.1f}s/seed)")
            print(f"[RUN4]   last-100 mean reward (over seeds): "
                  f"{rets[:, -100:].mean(axis=1).mean():.2f} ± "
                  f"{rets[:, -100:].mean(axis=1).std():.2f}")

    grand_elapsed = time.time() - grand_t0
    print(f"\n[RUN4] TOTAL: {grand_elapsed:.1f}s = {grand_elapsed/60:.1f} min")

    np.savez_compressed(out, **payload)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[RUN4] Saved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
