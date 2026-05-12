"""Phase 1 — BC scaling law on Walker2D-v5.

Research question: how does BC performance scale with the number of expert
demonstrations? We collect 200 trajectories ONCE from the PPO expert and then
train independent BC models on the first N ∈ {5, 10, 25, 50, 100, 200}
trajectories, evaluating each in the env.

We report:
- Mean ± std eval reward as a function of N (log-x scale).
- Fraction of expert reward recovered.
- Identified "knee" of the curve (point of diminishing returns).
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


def collect_trajectories(expert, env, n_episodes: int):
    """Collect n_episodes trajectories. Returns a list of (obs, act) arrays
    per episode so we can subsample by trajectory count later."""
    trajs = []
    obs = env.reset()
    for ep in range(n_episodes):
        ep_obs, ep_act = [], []
        done = [False]
        while not done[0]:
            action, _ = expert.predict(obs, deterministic=True)
            ep_obs.append(obs[0].copy())
            ep_act.append(action[0].copy())
            obs, _, done, _ = env.step(action)
        trajs.append((np.array(ep_obs, dtype=np.float32),
                      np.array(ep_act, dtype=np.float32)))
        obs = env.reset()
    return trajs


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


def evaluate_keras_policy(bc, env, n_episodes: int = 20):
    returns = np.zeros(n_episodes, dtype=np.float64)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        obs = env.reset()
        ep_ret = 0.0
        steps = 0
        done = [False]
        while not done[0] and steps < 1000:
            action = np.clip(bc.predict(obs, verbose=0), -1.0, 1.0)
            obs, reward, done, _info = env.step(action)
            ep_ret += float(reward[0])
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps
    return returns, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--total-trajs", type=int, default=200)
    p.add_argument("--ns", type=int, nargs="+", default=[5, 10, 25, 50, 100, 200])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from stable_baselines3 import PPO

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))

    out_dir = Path(f"group_project/runs/bc_scaling_{args.env.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect 200 expert trajectories ONCE (deterministic, seed 42).
    print(f"[SCALE] Collecting {args.total_trajs} expert trajectories from {args.env}...")
    t0 = time.time()
    env = make_env(env_id, args.vecnorm, args.seed)
    expert = PPO.load(args.expert, env=env)
    trajs = collect_trajectories(expert, env, args.total_trajs)
    env.close()
    print(f"[SCALE] Collected in {time.time() - t0:.1f}s. "
          f"Avg traj length: {np.mean([len(t[0]) for t in trajs]):.0f}")

    # Save the full trajectory pool so we don't recollect later.
    np.savez(out_dir / "expert_trajs.npz",
             obs=np.concatenate([t[0] for t in trajs]),
             actions=np.concatenate([t[1] for t in trajs]),
             lengths=np.array([len(t[0]) for t in trajs], dtype=np.int32))

    obs_dim = trajs[0][0].shape[1]
    act_dim = trajs[0][1].shape[1]

    # Evaluate the expert itself once for the reference number.
    print("[SCALE] Evaluating expert (reference)...")
    env = make_env(env_id, args.vecnorm, seed=123)
    def expert_predict(obs):
        action, _ = expert.predict(obs, deterministic=True)
        return action
    # Reuse the existing evaluate helper from eval_policies if needed; keep inline:
    expert_returns = np.zeros(args.eval_episodes)
    expert_lengths = np.zeros(args.eval_episodes, dtype=np.int64)
    for ep in range(args.eval_episodes):
        obs = env.reset()
        ep_ret, steps, done = 0.0, 0, [False]
        while not done[0] and steps < 1000:
            obs, reward, done, _ = env.step(expert_predict(obs))
            ep_ret += float(reward[0])
            steps += 1
        expert_returns[ep] = ep_ret
        expert_lengths[ep] = steps
    env.close()
    expert_mean = expert_returns.mean()
    print(f"[SCALE] Expert reference: {expert_mean:.1f} ± {expert_returns.std():.1f}")

    # Sweep over N.
    results = {"N": [], "mean": [], "std": [], "frac_recovered": [], "mean_len": []}
    for N in args.ns:
        N_clamped = min(N, len(trajs))
        sub = trajs[:N_clamped]
        obs_train = np.concatenate([t[0] for t in sub])
        act_train = np.concatenate([t[1] for t in sub])

        print(f"\n[SCALE] === N={N_clamped} trajs ({len(obs_train)} (s,a) pairs) ===")
        bc = build_bc_policy(obs_dim, act_dim, seed=args.seed)
        t0 = time.time()
        bc.fit(obs_train, act_train, epochs=args.epochs, batch_size=256,
               validation_split=0.1, verbose=0)
        train_time = time.time() - t0

        env = make_env(env_id, args.vecnorm, seed=123)
        rets, lens = evaluate_keras_policy(bc, env, args.eval_episodes)
        env.close()
        frac = rets.mean() / expert_mean

        print(f"[SCALE]   train={train_time:.1f}s  eval reward={rets.mean():7.1f} ± {rets.std():6.1f}  "
              f"recovered={frac*100:.1f}%  len={lens.mean():.0f}")

        results["N"].append(int(N_clamped))
        results["mean"].append(float(rets.mean()))
        results["std"].append(float(rets.std()))
        results["frac_recovered"].append(float(frac))
        results["mean_len"].append(float(lens.mean()))

    results["expert_mean"] = float(expert_mean)
    results["expert_std"] = float(expert_returns.std())
    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[SCALE] Saved {out_json}")


if __name__ == "__main__":
    main()
