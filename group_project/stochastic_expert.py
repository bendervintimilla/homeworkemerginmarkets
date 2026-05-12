"""Phase 4 — Stochastic-expert collection ablation.

Question: does collecting expert demonstrations stochastically (sampling from
the PPO Gaussian) give a BC that beats the BC trained on deterministic demos?

Hypothesis: stochastic expert visits a wider distribution of states (because
the noise perturbs trajectories), so BC sees more state-action coverage and
generalizes better on the (mostly deterministic) policy rollout it does at
test time.

Setup: at fixed N=50, train two BCs:
  (a) BC_det:  demos collected with deterministic=True (mean of Gaussian)
  (b) BC_stoch: demos collected with deterministic=False (sampled actions)

Both BCs are trained with the same architecture, epochs, and seed. Both are
evaluated deterministically (the standard reporting convention).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _exists(env_id: str) -> bool:
    import gymnasium as gym
    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False


def make_env(env_id: str, vecnorm_path: str, seed: int):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    env = make_vec_env(env_id, n_envs=1, seed=seed)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return env


def collect(expert, env, n_episodes: int, deterministic: bool):
    obs_list, act_list = [], []
    obs = env.reset()
    ep_returns = np.zeros(n_episodes)
    ep_lengths = np.zeros(n_episodes, dtype=np.int64)
    state_pos = []  # to estimate state-coverage diversity later
    for ep in range(n_episodes):
        ep_ret, steps, done = 0.0, 0, [False]
        while not done[0]:
            action, _ = expert.predict(obs, deterministic=deterministic)
            obs_list.append(obs[0].copy())
            act_list.append(action[0].copy())
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward[0])
            steps += 1
        ep_returns[ep] = ep_ret
        ep_lengths[ep] = steps
        obs = env.reset()
    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32),
            ep_returns, ep_lengths)


def build_bc_policy(obs_dim: int, act_dim: int, seed: int):
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(int(obs_dim),)),
        keras.layers.Dense(256, activation="tanh"),
        keras.layers.Dense(256, activation="tanh"),
        keras.layers.Dense(int(act_dim), activation="tanh"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def evaluate(bc, env, n_episodes: int = 20):
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    obs = env.reset()
    for ep in range(n_episodes):
        ep_ret, steps, done = 0.0, 0, [False]
        while not done[0] and steps < 1000:
            obs, reward, done, _ = env.step(
                np.clip(bc.predict(obs, verbose=0), -1.0, 1.0))
            ep_ret += float(reward[0])
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps
        obs = env.reset()
    return returns, lengths


def state_coverage(obs_array: np.ndarray) -> dict:
    """Cheap proxy for state diversity: per-dim std of normalized obs.
    Higher std = wider state distribution visited."""
    return {
        "per_dim_std_mean": float(obs_array.std(axis=0).mean()),
        "per_dim_std_max": float(obs_array.std(axis=0).max()),
        "trace_cov": float(np.trace(np.cov(obs_array, rowvar=False))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--n-trajs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from stable_baselines3 import PPO

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))

    out_dir = Path(f"group_project/runs/stoch_expert_{args.env.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"variants": []}
    for variant, deterministic in [("deterministic", True), ("stochastic", False)]:
        print(f"\n[STOCH] === variant: {variant} (deterministic={deterministic}) ===")
        env = make_env(env_id, args.vecnorm, args.seed)
        expert = PPO.load(args.expert, env=env)
        t0 = time.time()
        obs, act, exp_rets, exp_lens = collect(expert, env, args.n_trajs, deterministic)
        env.close()
        print(f"[STOCH]   collected {len(obs)} (s, a) in {time.time() - t0:.1f}s")
        print(f"[STOCH]   expert episodes during collection: "
              f"R={exp_rets.mean():.0f} ± {exp_rets.std():.0f}, len={exp_lens.mean():.0f}")
        cov = state_coverage(obs)
        print(f"[STOCH]   state coverage: per-dim std mean={cov['per_dim_std_mean']:.3f}, "
              f"trace cov={cov['trace_cov']:.2f}")

        bc = build_bc_policy(obs.shape[1], act.shape[1], seed=args.seed)
        bc.fit(obs, act, epochs=args.epochs, batch_size=256,
               validation_split=0.1, verbose=0)

        eval_env = make_env(env_id, args.vecnorm, seed=123)
        rets, lens = evaluate(bc, eval_env, args.eval_episodes)
        eval_env.close()
        print(f"[STOCH]   BC eval: R={rets.mean():.1f} ± {rets.std():.1f}, len={lens.mean():.0f}")

        results["variants"].append({
            "name": variant,
            "deterministic_collection": deterministic,
            "n_pairs": int(len(obs)),
            "expert_during_collection_R": float(exp_rets.mean()),
            "expert_during_collection_R_std": float(exp_rets.std()),
            "state_coverage": cov,
            "bc_eval_R": float(rets.mean()),
            "bc_eval_R_std": float(rets.std()),
            "bc_eval_len": float(lens.mean()),
        })

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[STOCH] Saved {out_json}")

    a, b = results["variants"]
    delta = b["bc_eval_R"] - a["bc_eval_R"]
    print(f"\n[STOCH] Δ stochastic - deterministic = {delta:+.1f} "
          f"({delta / a['bc_eval_R'] * 100:+.1f}%)")


if __name__ == "__main__":
    main()
