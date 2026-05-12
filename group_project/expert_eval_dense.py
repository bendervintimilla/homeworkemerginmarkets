"""Dense expert eval: 100 episodes for a tight reference number."""

from __future__ import annotations

import argparse
import json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    from stable_baselines3 import PPO

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))
    env = make_env(env_id, args.vecnorm, args.seed)
    expert = PPO.load(args.expert, env=env)

    returns = np.zeros(args.episodes)
    lengths = np.zeros(args.episodes, dtype=np.int64)
    obs = env.reset()
    for ep in range(args.episodes):
        ep_ret, steps, done = 0.0, 0, [False]
        while not done[0] and steps < 1000:
            action, _ = expert.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward[0])
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps
        obs = env.reset()
        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{args.episodes}] running mean: {returns[:ep+1].mean():.1f}",
                  flush=True)
    env.close()

    sem = returns.std() / np.sqrt(args.episodes)
    print(f"\nExpert (deterministic, {args.episodes} ep): "
          f"mean={returns.mean():.1f}  std={returns.std():.1f}  SEM={sem:.1f}")
    print(f"95% CI: [{returns.mean() - 1.96*sem:.1f}, {returns.mean() + 1.96*sem:.1f}]")

    out = Path(f"group_project/runs/expert_eval_dense_{args.env.lower()}.json")
    out.write_text(json.dumps({
        "episodes": args.episodes,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "sem": float(sem),
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
    }, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
