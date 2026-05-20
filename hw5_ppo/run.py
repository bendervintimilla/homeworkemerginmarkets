"""HW5 driver: PPO on LunarLander-v3 (discrete actions).

Standard PPO loop:

    repeat:
        roll out T steps with the current policy, store transitions
        compute GAE advantages and returns
        for K epochs:
            shuffle and iterate over mini-batches
            compute clipped policy loss + value loss + entropy bonus
            backprop & optimizer step
        (early-stop if KL too large)
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

from utils import plot_learning_curve, set_global_seed

from .buffer import RolloutBuffer
from .ppo import PPOAgent, PPOConfig


def _resolve_env(base: str, version_preference: list[str]) -> str:
    for v in version_preference:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"No registered version of {base}")


def main():
    parser = argparse.ArgumentParser(description="HW5 - PPO on LunarLander")
    parser.add_argument("--env", default="LunarLander")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    env_id = _resolve_env(args.env, ["v3", "v2"]) if args.env == "LunarLander" \
        else _resolve_env(args.env, ["v1"])
    out = Path(args.out or f"hw5_ppo/results_{args.env.lower()}")
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    cfg = PPOConfig(
        obs_dim=obs_dim,
        n_actions=n_actions,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        seed=args.seed,
    )
    agent = PPOAgent(cfg)
    buffer = RolloutBuffer(cfg.rollout_steps, (obs_dim,), cfg.gamma, cfg.gae_lambda)

    print(f"\n[HW5] Training PPO on {env_id} for {args.total_steps} steps")
    print(f"      rollout={cfg.rollout_steps}, epochs={cfg.epochs}, "
          f"minibatch={cfg.minibatch_size}, lr={cfg.lr}")

    total_steps = 0
    update = 0
    obs, _ = env.reset(seed=args.seed)
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_return_curve: list[float] = []
    cur_ret, cur_len = 0.0, 0
    t0 = time.time()

    while total_steps < args.total_steps:
        buffer.reset()
        for _ in range(cfg.rollout_steps):
            action, logp, value = agent.policy(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            buffer.add(obs, action, logp, reward, value, done or truncated)
            obs = next_obs
            cur_ret += reward
            cur_len += 1
            total_steps += 1
            if done or truncated:
                ep_returns.append(cur_ret)
                ep_lengths.append(cur_len)
                ep_return_curve.append(cur_ret)
                cur_ret, cur_len = 0.0, 0
                obs, _ = env.reset()

        # Bootstrap value for the last partial step.
        last_value = 0.0 if (cur_len == 0) else agent.value(obs)
        buffer.compute_gae(last_value)
        stats = agent.update(buffer.get())
        update += 1

        if update % 5 == 0 or total_steps >= args.total_steps:
            mean_r = np.mean(ep_returns) if ep_returns else float("nan")
            mean_l = np.mean(ep_lengths) if ep_lengths else float("nan")
            elapsed = time.time() - t0
            print(f"  upd={update:>4d}  steps={total_steps:>7d}  "
                  f"R(100)={mean_r:7.2f}  len={mean_l:5.1f}  "
                  f"pg={stats['pg_loss']:+.3f}  v={stats['v_loss']:.3f}  "
                  f"H={stats['entropy']:.3f}  KL={stats['kl']:.4f}  "
                  f"cf={stats['clipfrac']:.2f}  ep={stats['epochs_used']}  "
                  f"({elapsed:.0f}s)")

    plot_learning_curve(
        ep_return_curve,
        title=f"PPO – {env_id}",
        out_path=out / "learning_curve.png",
        window=50,
        ylabel="Episode return",
    )

    final_mean = float(np.mean(ep_return_curve[-100:])) if ep_return_curve else float("nan")
    report = [
        f"# HW5 – PPO en {env_id}",
        "",
        f"Total env-steps: {args.total_steps}",
        f"Rollout: {cfg.rollout_steps}, Epochs: {cfg.epochs}, "
        f"Mini-batch: {cfg.minibatch_size}, LR: {cfg.lr}",
        f"GAE: gamma={cfg.gamma}, lambda={cfg.gae_lambda}, clip={cfg.clip_eps}",
        "",
        f"Retorno medio últimos 100 episodios: **{final_mean:.2f}**",
        f"\"Resuelto\" en LunarLander es R >= 200.",
        "",
        "## Notas de implementación",
        "",
        "- Inicialización ortogonal (gain=sqrt(2) en hidden, 0.01 en logits, 1.0 en V).",
        "  Sigue las guidelines del paper original y del *PPO Implementation Details* blog.",
        "- Normalización de advantages a media 0, std 1 por minibatch.",
        "- Early stopping por KL aprox > 1.5 * target_kl (target_kl=0.02).",
        "- Entropy bonus = 0.01 (favorece exploración temprana).",
        "- Value coefficient = 0.5, gradient clipping = 0.5.",
    ]
    (out / "REPORT.md").write_text("\n".join(report))

    print(f"\n=== HW5 results ===")
    print(f"  Final R(100): {final_mean:.2f}")
    print(f"  -> {out}/")


if __name__ == "__main__":
    main()
