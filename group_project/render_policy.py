"""Render a trained policy as MP4/GIF for embedding in the report page.

Loads the PPO expert (or any saved BC model) and runs N episodes in an env
with `render_mode='rgb_array'`. Each frame is captured and stitched into a
video at `group_project/web/<name>.mp4` (falls back to .gif if no ffmpeg).
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


def make_env(env_id: str, vecnorm_path: str, seed: int):
    """Make an env wrapped with VecNormalize. The underlying env must have
    render_mode='rgb_array' so we can call render() through the wrappers."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    env = make_vec_env(env_id, n_envs=1, seed=seed,
                       env_kwargs={"render_mode": "rgb_array"})
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return env


def render_expert_episodes(expert, env, n_episodes: int, max_frames: int = 2000):
    """Roll out the expert (deterministic). Returns a list of (H, W, 3) uint8
    frames concatenated across episodes."""
    frames = []
    obs = env.reset()
    for ep in range(n_episodes):
        steps, done = 0, [False]
        ep_R = 0.0
        while not done[0] and steps < 1000 and len(frames) < max_frames:
            frame = env.render()  # VecEnv → returns the inner env's render
            if isinstance(frame, np.ndarray):
                frames.append(frame.astype(np.uint8))
            action, _ = expert.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_R += float(reward[0])
            steps += 1
        print(f"  ep {ep+1}: {steps} steps, R={ep_R:.0f}")
        if not done[0]:
            # Truncated by max_frames; reset still
            pass
        obs = env.reset()
        if len(frames) >= max_frames:
            break
    return frames


def render_keras_policy_episodes(bc, env, n_episodes: int, max_frames: int = 2000):
    from tensorflow import keras  # noqa: F401  (ensures TF is initialized)
    frames = []
    obs = env.reset()
    for ep in range(n_episodes):
        steps, done = 0, [False]
        ep_R = 0.0
        while not done[0] and steps < 1000 and len(frames) < max_frames:
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame.astype(np.uint8))
            action = np.clip(bc.predict(obs, verbose=0), -1.0, 1.0)
            obs, reward, done, _ = env.step(action)
            ep_R += float(reward[0])
            steps += 1
        print(f"  ep {ep+1}: {steps} steps, R={ep_R:.0f}")
        obs = env.reset()
        if len(frames) >= max_frames:
            break
    return frames


def write_video(frames: list, out_path: Path, fps: int = 60):
    """Write MP4 if ffmpeg available, else GIF."""
    import imageio
    if out_path.suffix == ".mp4":
        try:
            with imageio.get_writer(out_path, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=1) as w:
                for f in frames:
                    w.append_data(f)
            print(f"saved {out_path}  ({len(frames)} frames, {fps} fps)")
            return
        except Exception as e:
            print(f"  mp4 writer failed ({e}); falling back to gif")
            out_path = out_path.with_suffix(".gif")
    # GIF fallback (subsample to keep size reasonable)
    step = max(1, len(frames) // 200)
    imageio.mimsave(out_path, frames[::step], fps=fps, loop=0)
    print(f"saved {out_path}  ({len(frames[::step])} frames at 1/{step} subsample)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", default="group_project/runs/ppo_walker2d/final_model.zip")
    p.add_argument("--vecnorm", default="group_project/runs/ppo_walker2d/vecnorm.pkl")
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--policy", choices=["expert", "bc_det", "bc_stoch", "bc_dagger"],
                   default="expert")
    p.add_argument("--out-name", type=str, default=None,
                   help="Output filename (without extension). Defaults to '<policy>_<env>'")
    args = p.parse_args()

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))
    print(f"[RENDER] env={env_id}  policy={args.policy}  seed={args.seed}  episodes={args.episodes}")

    env = make_env(env_id, args.vecnorm, args.seed)

    if args.policy == "expert":
        from stable_baselines3 import PPO
        expert = PPO.load(args.expert, env=env)
        frames = render_expert_episodes(expert, env, args.episodes, args.max_frames)
    else:
        from tensorflow import keras
        model_paths = {
            "bc_det":    "group_project/runs/bc_walker2d/bc_model.keras",
            "bc_dagger": "group_project/runs/dagger_walker2d/bc_dagger_final.keras",
        }
        path = model_paths.get(args.policy)
        if not path or not Path(path).exists():
            raise SystemExit(f"No saved model for policy='{args.policy}'. Tried {path}.")
        bc = keras.models.load_model(path)
        frames = render_keras_policy_episodes(bc, env, args.episodes, args.max_frames)

    env.close()

    if not frames:
        raise SystemExit("No frames captured. Ensure env was created with render_mode='rgb_array'.")
    print(f"[RENDER] captured {len(frames)} frames at {frames[0].shape}")

    out_name = args.out_name or f"{args.policy}_{args.env.lower()}"
    out_dir = Path("group_project/web")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.mp4"
    write_video(frames, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
