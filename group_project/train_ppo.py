"""Group project – PPO baseline on Walker2D-v5 / Ant-v5 (continuous control).

Usa Stable-Baselines3 como referencia industrial. La idea es:
  1. Establecer un baseline de PPO puro y medir tiempo + retorno.
  2. (En `imitation.py`) implementar Imitation Learning – Behavior Cloning de
     un experto (aquí, el propio modelo PPO entrenado guardado como expert).
  3. (Extensión) inicializar PPO desde la BC policy y ver si converge antes.

Uso:
    python -m group_project.train_ppo --env Walker2d-v5 --steps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np


def _resolve_env(base: str) -> str:
    versions = ["v5", "v4", "v3"]
    for v in versions:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"No registered version of {base}")


def main():
    parser = argparse.ArgumentParser(description="Group project – PPO baseline")
    parser.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        raise SystemExit(
            "Install Stable-Baselines3 + Mujoco: "
            "pip install 'stable-baselines3[extra]' mujoco"
        )

    env_id = _resolve_env(args.env)
    out = Path(args.out or f"group_project/runs/ppo_{args.env.lower()}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[GP] PPO on {env_id}, {args.steps} steps, n_envs={args.n_envs}")

    train_env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0)
    eval_env = make_vec_env(env_id, n_envs=1, seed=args.seed + 100)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    model = PPO(
        "MlpPolicy", train_env,
        n_steps=2048, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
        max_grad_norm=0.5, target_kl=0.02,
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
        verbose=1, seed=args.seed,
        tensorboard_log=str(out / "tb"),
    )

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(out),
        log_path=str(out / "eval"), eval_freq=10_000 // args.n_envs,
        n_eval_episodes=5, deterministic=True, render=False,
    )

    model.learn(total_timesteps=args.steps, callback=eval_callback,
                progress_bar=True)
    model.save(str(out / "final_model"))
    train_env.save(str(out / "vecnorm.pkl"))
    print(f"[GP] Saved final model and VecNormalize stats to {out}")


if __name__ == "__main__":
    main()
