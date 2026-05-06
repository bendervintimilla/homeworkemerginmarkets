"""Visual replay of the trained agents.

Loads the Q-tables from `results_full/` and runs the greedy policy with
gymnasium's "human" render mode, so you see the FrozenLake / Taxi / Volcano
agent actually playing.

Usage:
    python -m hw1_frozen_lake.watch --env frozenlake_4x4_slippery
    python -m hw1_frozen_lake.watch --env frozenlake_8x8_deterministic
    python -m hw1_frozen_lake.watch --env taxi
    python -m hw1_frozen_lake.watch --env volcano   # ASCII rendering only

Args:
    --episodes : how many episodes to play (default 5).
    --fps      : optional override for the render speed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .volcano_world import VolcanoWorldEnv


SCENARIOS = {
    "frozenlake_4x4_deterministic": dict(
        env_id="FrozenLake-v1", env_kwargs=dict(map_name="4x4", is_slippery=False),
        npz="frozenlake_4x4_deterministic.npz", q_key="Q_mc"),
    "frozenlake_4x4_slippery": dict(
        env_id="FrozenLake-v1", env_kwargs=dict(map_name="4x4", is_slippery=True),
        npz="frozenlake_4x4_slippery.npz", q_key="Q_td"),
    "frozenlake_8x8_deterministic": dict(
        env_id="FrozenLake-v1", env_kwargs=dict(map_name="8x8", is_slippery=False),
        npz="frozenlake_8x8_deterministic.npz", q_key="Q_mc"),
    "taxi": dict(
        env_id=None,  # resolved at runtime: Taxi-v4 or v3
        env_kwargs={},
        npz="taxi.npz", q_key="Q_td"),
    "volcano": dict(
        env_id="<volcano>",
        env_kwargs={},
        npz="volcano.npz", q_key="Q"),
}


def _resolve_taxi() -> str:
    for v in ("v4", "v3"):
        try:
            gym.spec(f"Taxi-{v}")
            return f"Taxi-{v}"
        except Exception:
            continue
    raise RuntimeError("No Taxi env registered")


def watch(scenario: str, episodes: int, fps: int | None,
          results_dir: Path) -> None:
    cfg = SCENARIOS[scenario]
    npz = np.load(results_dir / cfg["npz"])
    Q = npz[cfg["q_key"]]
    print(f"Loaded Q-table {Q.shape} from {cfg['npz']} (key={cfg['q_key']})")

    if cfg["env_id"] == "<volcano>":
        env = VolcanoWorldEnv(is_slippery=True)
        ascii_render = True
    elif cfg["env_id"] is None and scenario == "taxi":
        env_id = _resolve_taxi()
        env = gym.make(env_id, render_mode="human")
        ascii_render = False
    else:
        env = gym.make(cfg["env_id"], render_mode="human", **cfg["env_kwargs"])
        ascii_render = False

    rng = np.random.default_rng(0)
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_r, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < 500:
            qs = Q[s]
            cand = np.flatnonzero(qs == qs.max())
            a = int(rng.choice(cand))
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
            steps += 1
            if ascii_render:
                print(env.render())
                if fps is not None:
                    time.sleep(1.0 / fps)
        print(f"Episode {ep + 1}: return={ep_r:.2f}, steps={steps}")
        rewards.append(ep_r)

    env.close()
    print(f"\nMean over {episodes} eps: {np.mean(rewards):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(SCENARIOS.keys()),
                        default="frozenlake_4x4_slippery")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=None,
                        help="Frames per second (only meaningful for ANSI rendering)")
    parser.add_argument("--results", default="hw1_frozen_lake/results_full")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        raise SystemExit(
            f"Results dir {results_dir} not found. Run `python -m hw1_frozen_lake.run_full` first."
        )
    watch(args.env, args.episodes, args.fps, results_dir)


if __name__ == "__main__":
    main()
