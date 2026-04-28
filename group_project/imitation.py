"""Behavior Cloning (Imitation Learning) baseline para el group project.

Idea:
    1. Cargamos un experto (PPO entrenado con `train_ppo.py`).
    2. Generamos N trayectorias del experto.
    3. Entrenamos una red supervisada que mapea obs -> action.
    4. (Opcional) Evaluamos directamente la BC policy + comparamos con el
       experto y con un PPO entrenado desde cero el mismo nº de steps.

Para tareas continuas usamos una pi(s) determinística (regresión MSE sobre la
acción media del experto). Para extensiones más avanzadas, se podría usar DAgger.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def collect_expert_data(model, env, n_episodes: int = 50):
    obs_list, act_list = [], []
    obs = env.reset()
    for _ in range(n_episodes):
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs_list.append(obs[0].copy())
            act_list.append(action[0].copy())
            obs, _, done, _ = env.step(action)
        obs = env.reset()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def build_bc_policy(obs_dim: int, act_dim: int):
    from tensorflow import keras
    obs_dim = int(obs_dim)
    act_dim = int(act_dim)
    model = keras.Sequential([
        keras.layers.Input(shape=(obs_dim,)),
        keras.layers.Dense(256, activation="tanh"),
        keras.layers.Dense(256, activation="tanh"),
        keras.layers.Dense(act_dim, activation="tanh"),  # MuJoCo actions in [-1, 1]
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main():
    parser = argparse.ArgumentParser(description="Group project – Behavior Cloning")
    parser.add_argument("--expert", required=True, help="Path to a SB3 PPO model .zip")
    parser.add_argument("--vecnorm", required=True, help="Path to VecNormalize .pkl")
    parser.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError:
        raise SystemExit("pip install stable-baselines3 mujoco gymnasium")

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))
    env = make_vec_env(env_id, n_envs=1, seed=args.seed)
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False

    expert = PPO.load(args.expert, env=env)
    print(f"[GP-BC] Collecting {args.episodes} expert trajectories...")
    obs, act = collect_expert_data(expert, env, n_episodes=args.episodes)
    print(f"[GP-BC] Collected {len(obs)} (obs, action) pairs.")

    bc = build_bc_policy(obs.shape[1], act.shape[1])
    bc.fit(obs, act, epochs=args.epochs, batch_size=256, validation_split=0.1)

    out = Path(f"group_project/runs/bc_{args.env.lower()}")
    out.mkdir(parents=True, exist_ok=True)
    bc.save(str(out / "bc_model.keras"))
    print(f"[GP-BC] BC policy saved to {out}/bc_model.keras")


def _exists(env_id: str) -> bool:
    import gymnasium as gym
    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
