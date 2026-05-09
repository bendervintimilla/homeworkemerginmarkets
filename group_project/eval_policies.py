"""Evaluate PPO expert vs Behavior-Cloned policy on the same environment.

Uses the saved VecNormalize so observations are normalized identically for
both policies (BC was trained on normalized obs collected by the expert).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _exists(env_id: str) -> bool:
    import gymnasium as gym
    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False


def evaluate_policy(predict_fn, env, n_episodes: int = 20, max_steps: int = 1000):
    returns = np.zeros(n_episodes, dtype=np.float64)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        obs = env.reset()
        ep_ret = 0.0
        steps = 0
        done = [False]
        while not done[0] and steps < max_steps:
            action = predict_fn(obs)
            obs, reward, done, _info = env.step(action)
            ep_ret += float(reward[0])
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps
    return returns, lengths


def main():
    p = argparse.ArgumentParser(description="Evaluate PPO expert vs BC policy")
    p.add_argument("--expert", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--bc", required=True, help="Path to bc_model.keras")
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from tensorflow import keras

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))

    def make_env():
        env = make_vec_env(env_id, n_envs=1, seed=args.seed)
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False
        return env

    print(f"[EVAL] Env: {env_id} | episodes per policy: {args.episodes}")

    env = make_env()
    expert = PPO.load(args.expert, env=env)
    def expert_predict(obs):
        action, _ = expert.predict(obs, deterministic=True)
        return action
    print("[EVAL] PPO expert...")
    exp_ret, exp_len = evaluate_policy(expert_predict, env, args.episodes)
    env.close()

    env = make_env()
    bc = keras.models.load_model(args.bc)
    def bc_predict(obs):
        action = bc.predict(obs, verbose=0)
        return np.clip(action, -1.0, 1.0)
    print("[EVAL] BC policy...")
    bc_ret, bc_len = evaluate_policy(bc_predict, env, args.episodes)
    env.close()

    print()
    print(f"  PPO expert  : reward {exp_ret.mean():8.2f} ± {exp_ret.std():6.2f} | len {exp_len.mean():6.1f}")
    print(f"  BC policy   : reward {bc_ret.mean():8.2f} ± {bc_ret.std():6.2f} | len {bc_len.mean():6.1f}")
    gap = exp_ret.mean() - bc_ret.mean()
    print(f"  Gap (expert - BC): {gap:.2f} ({gap / max(abs(exp_ret.mean()), 1e-9) * 100:.1f}%)")

    out = Path(f"group_project/runs/eval_{args.env.lower()}.npz")
    np.savez(out, expert_returns=exp_ret, expert_lens=exp_len,
             bc_returns=bc_ret, bc_lens=bc_len)
    print(f"[EVAL] Saved {out}")


if __name__ == "__main__":
    main()
