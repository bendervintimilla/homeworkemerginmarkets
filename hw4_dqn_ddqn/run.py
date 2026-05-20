"""HW4 driver: DQN and DDQN on CartPole-v1 and LunarLander-v3.

Trains both algorithms and dumps learning curves so we can visually compare
the value-overestimation reduction of DDQN.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from utils import ReplayBuffer, plot_multiple_curves, set_global_seed

from .agent import DQNAgent, DQNConfig


def _resolve_env(base: str, version_preference: list[str]) -> str:
    for v in version_preference:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"No registered version of {base}")


def _default_cfg(env_id: str) -> DQNConfig:
    if env_id.startswith("CartPole"):
        # CartPole + DQN is famously unstable: the Q values blow up once the
        # agent gets long episodes, causing catastrophic forgetting. The fix
        # is conservative: small LR, slow target-net sync, longer eps decay,
        # train a bit less aggressively.
        return DQNConfig(
            obs_dim=4, n_actions=2,
            hidden=(64, 64), lr=5e-4, gamma=0.99,
            buffer_size=50_000, batch_size=64, learning_starts=1_000,
            target_update_freq=500, train_freq=4,
            eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000,
        )
    elif env_id.startswith("LunarLander"):
        return DQNConfig(
            obs_dim=8, n_actions=4,
            hidden=(128, 128), lr=5e-4, gamma=0.99,
            buffer_size=100_000, batch_size=64, learning_starts=10_000,
            target_update_freq=1_000, train_freq=4,
            eps_start=1.0, eps_end=0.02, eps_decay_steps=200_000,
        )
    else:
        raise ValueError(f"No default config for {env_id}")


def train_agent(
    env_id: str,
    total_steps: int,
    double: bool,
    seed: int,
    eval_every: int = 10_000,
) -> dict:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    cfg = _default_cfg(env_id)
    cfg.obs_dim = obs_dim
    cfg.n_actions = n_actions
    cfg.double = double
    cfg.seed = seed

    agent = DQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,))

    ep_returns: list[float] = []
    eval_points: list[tuple[int, float]] = []
    ep_return, ep_steps = 0.0, 0
    losses: list[float] = []
    last_log = time.time()

    for step in range(1, total_steps + 1):
        eps = agent.epsilon(step)
        a = agent.act(obs, eps)
        next_obs, r, done, truncated, _ = env.step(a)
        buffer.add(obs, a, r, next_obs, done)
        obs = next_obs
        ep_return += r
        ep_steps += 1

        if done or truncated:
            ep_returns.append(ep_return)
            obs, _ = env.reset()
            ep_return, ep_steps = 0.0, 0

        if len(buffer) >= cfg.learning_starts and step % cfg.train_freq == 0:
            batch = buffer.sample(cfg.batch_size)
            loss = agent.train_step(batch)
            losses.append(loss)

        if step % eval_every == 0:
            recent = np.mean(ep_returns[-20:]) if ep_returns else 0.0
            eval_points.append((step, float(recent)))
            elapsed = time.time() - last_log
            print(f"  step={step:>7d} eps={eps:.3f}  "
                  f"recent_R={recent:7.2f}  loss={np.mean(losses[-200:] or [0]):.4f}  "
                  f"({elapsed:.1f}s)")
            last_log = time.time()

    env.close()
    return {
        "returns": np.array(ep_returns, dtype=np.float32),
        "eval_points": eval_points,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="HW4 - DQN vs DDQN")
    parser.add_argument("--env", choices=["CartPole", "LunarLander"], default="CartPole")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="Env steps for each agent. Use ~300k for LunarLander.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    versions = ["v1"] if args.env == "CartPole" else ["v3", "v2"]
    env_id = _resolve_env(args.env, versions)
    out = Path(args.out or f"hw4_dqn_ddqn/results_{args.env.lower()}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[HW4] Training DQN on {env_id} for {args.steps} env-steps")
    dqn = train_agent(env_id, args.steps, double=False, seed=args.seed,
                      eval_every=args.eval_every)
    print(f"\n[HW4] Training DDQN on {env_id} for {args.steps} env-steps")
    ddqn = train_agent(env_id, args.steps, double=True, seed=args.seed + 1,
                       eval_every=args.eval_every)

    plot_multiple_curves(
        {"DQN": dqn["returns"], "DDQN": ddqn["returns"]},
        title=f"{env_id} – training returns",
        out_path=out / "returns.png",
        window=max(20, len(dqn["returns"]) // 30),
        ylabel="Episode return",
    )

    last_n = 50
    dqn_last = float(np.mean(dqn["returns"][-last_n:]))
    ddqn_last = float(np.mean(ddqn["returns"][-last_n:]))
    report = [
        f"# HW4 – DQN vs DDQN en {env_id}",
        "",
        f"Env steps: {args.steps}, seed_DQN={args.seed}, seed_DDQN={args.seed + 1}.",
        "",
        "## Resultados (media de los últimos 50 episodios de entrenamiento)",
        f"- DQN  : {dqn_last:.2f}",
        f"- DDQN : {ddqn_last:.2f}",
        "",
        "## Discusión",
        "",
        "- DQN puede sobreestimar Q porque toma `max` sobre estimaciones ruidosas",
        "  del propio target net. La curva suele ser más volátil.",
        "- DDQN desacopla selección y evaluación de la siguiente acción y",
        "  típicamente ofrece curvas más estables y un retorno asintótico igual",
        "  o mejor – más notable en LunarLander que en CartPole.",
        "- En CartPole DDQN aporta poco (recompensa acotada en [0,500] y la",
        "  optimización es fácil). En LunarLander la diferencia se ve mejor.",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    print(f"\n=== HW4 results ===")
    print(f"  DQN  last-50 mean: {dqn_last:.2f}")
    print(f"  DDQN last-50 mean: {ddqn_last:.2f}")
    print(f"  -> {out}/")


if __name__ == "__main__":
    main()
