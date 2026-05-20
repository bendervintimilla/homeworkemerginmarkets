"""Render a side-by-side comparison: random policy vs trained PPO.

The comparison shows what RL actually learned: keep the robot upright
and move forward. Random policy collapses in <50 steps; PPO walks 1000.
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


def render_random_episode(env_id: str, seed: int, max_frames: int):
    import gymnasium as gym
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    rewards = 0.0
    steps = 0
    done = False
    truncated = False
    while not (done or truncated) and steps < max_frames:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        frames.append(env.render())
        rewards += float(reward)
        steps += 1
    env.close()
    print(f"  random: {steps} steps, R={rewards:.1f}")
    return frames, rewards, steps


def render_ppo_episode(expert_path: str, vecnorm_path: str, env_id: str, seed: int, max_frames: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    env = make_vec_env(env_id, n_envs=1, seed=seed,
                       env_kwargs={"render_mode": "rgb_array"})
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    expert = PPO.load(expert_path, env=env)
    obs = env.reset()
    frames, rewards, steps, done = [], 0.0, 0, [False]
    while not done[0] and steps < max_frames:
        frames.append(env.render())
        action, _ = expert.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards += float(reward[0])
        steps += 1
    env.close()
    print(f"  PPO: {steps} steps, R={rewards:.1f}")
    return frames, rewards, steps


def stack_side_by_side(left, right, label_l, label_r, R_l, R_r):
    """Stack frames horizontally. If one ends before the other, freeze the
    last frame (random will end early; we keep it on a 'fallen' freeze-frame
    while PPO continues walking)."""
    import cv2 as cv  # noqa: F401  (would be nice but optional)

    n = max(len(left), len(right))
    h = max(left[0].shape[0], right[0].shape[0])

    def pad(frames, n_target, label, R):
        out = list(frames)
        # Freeze on the last frame to make the contrast obvious.
        while len(out) < n_target:
            out.append(frames[-1])
        return out

    L = pad(left, n, label_l, R_l)
    R = pad(right, n, label_r, R_r)

    # Pad heights if needed (Walker2D renders are 480x480 same dims usually).
    out = []
    for fl, fr in zip(L, R):
        if fl.shape[0] != h:
            fl = np.pad(fl, ((0, h - fl.shape[0]), (0, 0), (0, 0)), mode="constant")
        if fr.shape[0] != h:
            fr = np.pad(fr, ((0, h - fr.shape[0]), (0, 0), (0, 0)), mode="constant")
        # Vertical divider
        divider = np.full((h, 4, 3), 60, dtype=np.uint8)
        out.append(np.concatenate([fl, divider, fr], axis=1))
    return out


def add_labels(frames, label_l, label_r):
    """Burn text labels onto each frame using PIL (avoids cv2 dependency)."""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 28)
        font_small = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 20)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big

    w_half = frames[0].shape[1] // 2
    h = frames[0].shape[0]
    for f in frames:
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img, "RGBA")

        # Top labels with translucent backdrop
        for x_start, label in [(20, label_l), (w_half + 20, label_r)]:
            d.rectangle([x_start - 6, 12, x_start + 320, 80], fill=(0, 0, 0, 170))
            d.text((x_start, 20), label, fill=(255, 122, 61, 255), font=font_big)
            d.text((x_start, 55), "Walker2D-v5", fill=(180, 180, 180, 255), font=font_small)
        out.append(np.array(img))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", default="group_project/runs/ppo_walker2d/final_model.zip")
    p.add_argument("--vecnorm", default="group_project/runs/ppo_walker2d/vecnorm.pkl")
    p.add_argument("--env", choices=["Walker2d", "Ant"], default="Walker2d")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()

    versions = ["v5", "v4", "v3"]
    env_id = next(f"{args.env}-{v}" for v in versions if _exists(f"{args.env}-{v}"))

    print(f"[CMP] random episode...")
    rand_frames, R_rand, n_rand = render_random_episode(env_id, args.seed, args.max_frames)
    print(f"[CMP] PPO episode...")
    ppo_frames, R_ppo, n_ppo = render_ppo_episode(args.expert, args.vecnorm, env_id,
                                                   args.seed, args.max_frames)

    label_l = f"RANDOM   R = {R_rand:.0f}"
    label_r = f"TRAINED  R = {R_ppo:.0f}"

    print("[CMP] composing side-by-side...")
    sxs = stack_side_by_side(rand_frames, ppo_frames, label_l, label_r, R_rand, R_ppo)
    sxs_labeled = add_labels(sxs, label_l, label_r)

    out = Path("group_project/web/before_after_walker2d.mp4")
    import imageio
    with imageio.get_writer(out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1) as w:
        for f in sxs_labeled:
            w.append_data(f)
    print(f"saved {out}  ({len(sxs_labeled)} frames at {args.fps} fps, "
          f"{args.fps} → {len(sxs_labeled)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
