"""Phase 2 — DAgger (Dataset Aggregation, Ross et al. 2011).

Iterative imitation: roll out the current policy to collect the states it
ACTUALLY visits (not the expert's), label them with the expert's actions,
add to the dataset, retrain. Closes the BC distribution-shift gap.

We start from a BC trained on N=50 expert trajectories (the same baseline used
in the original group_project REPORT) and run K=5 DAgger iterations, each
collecting ~10 rollouts of the current policy.
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


def collect_expert_trajectories(expert, env, n_episodes: int):
    obs_list, act_list = [], []
    obs = env.reset()
    for _ in range(n_episodes):
        done = [False]
        while not done[0]:
            action, _ = expert.predict(obs, deterministic=True)
            obs_list.append(obs[0].copy())
            act_list.append(action[0].copy())
            obs, _, done, _ = env.step(action)
        obs = env.reset()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def rollout_policy_and_label(bc, expert, env, n_episodes: int,
                             beta: float = 0.0, rng: np.random.Generator | None = None):
    """Roll out a β-mixed policy, but at each visited state ALSO query the
    expert for the label. With probability β take the expert's action, with
    probability (1-β) take the BC's. The action stored in the dataset is the
    EXPERT's at every visited state (DAgger always trains on expert labels)."""
    rng = rng or np.random.default_rng(0)
    obs_list, exp_act_list = [], []
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    obs = env.reset()
    for ep in range(n_episodes):
        ep_ret, steps, done = 0.0, 0, [False]
        while not done[0] and steps < 1000:
            # Always query expert and store its action as the label.
            exp_action, _ = expert.predict(obs, deterministic=True)
            obs_list.append(obs[0].copy())
            exp_act_list.append(exp_action[0].copy())
            # β-mixing: take expert action with prob β, BC action with prob 1-β.
            if beta > 0 and rng.random() < beta:
                action = exp_action
            else:
                action = np.clip(bc.predict(obs, verbose=0), -1.0, 1.0)
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward[0])
            steps += 1
        returns[ep] = ep_ret
        lengths[ep] = steps
        obs = env.reset()
    return (np.array(obs_list, dtype=np.float32),
            np.array(exp_act_list, dtype=np.float32),
            returns, lengths)


def evaluate_policy(bc, env, n_episodes: int = 20):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--initial-trajs", type=int, default=50,
                   help="N of expert trajs for the iteration-0 BC")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--rollouts-per-iter", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--beta", type=float, default=0.0,
                   help="β-mixing: probability of using expert action during DAgger rollout (Ross 2011)")
    p.add_argument("--out-suffix", type=str, default="",
                   help="Suffix on output dir name (e.g. '_beta0.3')")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    from stable_baselines3 import PPO

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))

    out_dir = Path(f"group_project/runs/dagger_{args.env.lower()}{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load expert + reference env.
    env = make_env(env_id, args.vecnorm, args.seed)
    expert = PPO.load(args.expert, env=env)

    # Iteration 0: BC on N=initial_trajs expert demos.
    print(f"[DAGGER] Iter 0: BC baseline on {args.initial_trajs} expert trajs")
    obs0, act0 = collect_expert_trajectories(expert, env, args.initial_trajs)
    print(f"[DAGGER]   collected {len(obs0)} (s, a) pairs")

    bc = build_bc_policy(obs0.shape[1], act0.shape[1], seed=args.seed)
    bc.fit(obs0, act0, epochs=args.epochs, batch_size=256,
           validation_split=0.1, verbose=0)
    env.close()

    eval_env = make_env(env_id, args.vecnorm, seed=123)
    rets, lens = evaluate_policy(bc, eval_env, args.eval_episodes)
    eval_env.close()

    history = {
        "iteration": [0],
        "mean_reward": [float(rets.mean())],
        "std_reward": [float(rets.std())],
        "mean_length": [float(lens.mean())],
        "dataset_size": [len(obs0)],
    }
    print(f"[DAGGER]   iter 0  reward={rets.mean():7.1f} ± {rets.std():6.1f}  len={lens.mean():.0f}")

    # Accumulating dataset.
    D_obs, D_act = obs0, act0

    for it in range(1, args.iterations + 1):
        # Roll out current BC, label with expert.
        roll_env = make_env(env_id, args.vecnorm, seed=args.seed + 1000 + it)
        t0 = time.time()
        new_obs, new_exp_act, roll_returns, roll_lengths = rollout_policy_and_label(
            bc, expert, roll_env, args.rollouts_per_iter, beta=args.beta, rng=rng)
        roll_env.close()
        rollout_time = time.time() - t0

        # Aggregate.
        D_obs = np.concatenate([D_obs, new_obs])
        D_act = np.concatenate([D_act, new_exp_act])

        # Retrain BC from scratch on the aggregated dataset (DAgger paper
        # spec: retrain, not fine-tune, to avoid catastrophic forgetting).
        bc = build_bc_policy(D_obs.shape[1], D_act.shape[1], seed=args.seed)
        bc.fit(D_obs, D_act, epochs=args.epochs, batch_size=256,
               validation_split=0.1, verbose=0)

        # Eval.
        eval_env = make_env(env_id, args.vecnorm, seed=123)
        rets, lens = evaluate_policy(bc, eval_env, args.eval_episodes)
        eval_env.close()

        history["iteration"].append(it)
        history["mean_reward"].append(float(rets.mean()))
        history["std_reward"].append(float(rets.std()))
        history["mean_length"].append(float(lens.mean()))
        history["dataset_size"].append(len(D_obs))
        print(f"[DAGGER]   iter {it}  rollout={rollout_time:.1f}s  D|={len(D_obs)}  "
              f"reward={rets.mean():7.1f} ± {rets.std():6.1f}  len={lens.mean():.0f}  "
              f"(rollout R: {roll_returns.mean():.0f})")

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(history, indent=2))
    bc.save(str(out_dir / "bc_dagger_final.keras"))
    print(f"\n[DAGGER] Saved {out_json}")


if __name__ == "__main__":
    main()
