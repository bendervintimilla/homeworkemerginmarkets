"""Render a 4-panel side-by-side animation of the 4 algorithms' greedy
policies on CliffWalking.

Each panel: 4×12 grid with the agent traced step-by-step. Below each panel:
algorithm name + running cumulative reward. The 4 agents move in lockstep
(one env step per frame) so the viewer sees who falls into the cliff and who
makes it through.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLIFF_SHAPE = (4, 12)
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def _resolve(base, versions):
    for v in versions:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"Cannot find {base}")


def rollout_greedy(env_id, Q, max_steps=200):
    env = gym.make(env_id)
    s, _ = env.reset(seed=0)
    path = [s]
    rewards = [0.0]
    cum = 0.0
    for _ in range(max_steps):
        a = int(Q[s].argmax())
        s, r, done, trunc, _ = env.step(a)
        path.append(s)
        cum += float(r)
        rewards.append(cum)
        if done or trunc:
            break
    env.close()
    return path, rewards


def render_grid_frame(path_so_far, panel_w=520, panel_h=260, title="",
                      cum_reward=0.0):
    """Draw one panel: 4×12 grid with start, goal, cliff, and the agent
    trail. Returns RGB array."""
    img = Image.new("RGB", (panel_w, panel_h), (10, 10, 12))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 18)
        font_lbl = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 11)
    except Exception:
        font = font_lbl = font_small = ImageFont.load_default()

    # Title bar
    d.rectangle([0, 0, panel_w, 40], fill=(20, 20, 26))
    d.text((12, 10), title, fill=(255, 122, 61), font=font)
    d.text((panel_w - 160, 14), f"R = {cum_reward:+.0f}", fill=(94, 230, 163), font=font_lbl)

    # Grid
    grid_top = 50
    grid_h = panel_h - grid_top - 30
    grid_w = panel_w - 20
    cell_w = grid_w // 12
    cell_h = grid_h // 4
    grid_left = (panel_w - cell_w * 12) // 2

    for r in range(4):
        for c in range(12):
            x0 = grid_left + c * cell_w
            y0 = grid_top + r * cell_h
            x1 = x0 + cell_w - 2
            y1 = y0 + cell_h - 2

            if r == 3 and 1 <= c <= 10:
                d.rectangle([x0, y0, x1, y1], fill=(80, 30, 30))
                d.text((x0 + cell_w // 2 - 4, y0 + cell_h // 2 - 7), "C",
                       fill=(160, 160, 160), font=font_lbl)
            elif r == 3 and c == 0:
                d.rectangle([x0, y0, x1, y1], fill=(30, 60, 80))
                d.text((x0 + cell_w // 2 - 4, y0 + cell_h // 2 - 7), "S",
                       fill=(255, 255, 255), font=font_lbl)
            elif r == 3 and c == 11:
                d.rectangle([x0, y0, x1, y1], fill=(30, 80, 50))
                d.text((x0 + cell_w // 2 - 4, y0 + cell_h // 2 - 7), "G",
                       fill=(255, 255, 255), font=font_lbl)
            else:
                d.rectangle([x0, y0, x1, y1], fill=(28, 28, 34))

    # Trail (visited cells before the agent's current position)
    for i, s in enumerate(path_so_far[:-1]):
        rr, cc = divmod(s, 12)
        if rr == 3 and 0 < cc < 11:
            continue  # don't overdraw cliff/start cells
        x0 = grid_left + cc * cell_w
        y0 = grid_top + rr * cell_h
        alpha = int(50 + (i / max(len(path_so_far), 1)) * 150)
        d.rectangle([x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8],
                    fill=(255, 122, 61, alpha))

    # Agent (current head)
    if path_so_far:
        s = path_so_far[-1]
        rr, cc = divmod(s, 12)
        x0 = grid_left + cc * cell_w
        y0 = grid_top + rr * cell_h
        cx = x0 + cell_w // 2
        cy = y0 + cell_h // 2
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(255, 255, 255),
                  outline=(255, 122, 61), width=2)

    # Step counter
    d.text((12, panel_h - 22), f"step {len(path_so_far) - 1}",
           fill=(154, 154, 163), font=font_small)
    return np.array(img)


def composite_4(frames_dict, step_idx):
    """Take a dict of {algo_name: (panel_w, panel_h, 3)} and tile 2×2."""
    keys = list(frames_dict.keys())
    h = frames_dict[keys[0]].shape[0]
    w = frames_dict[keys[0]].shape[1]
    canvas = np.full((h * 2 + 20, w * 2 + 20, 3), 10, dtype=np.uint8)
    canvas[0:h, 0:w] = frames_dict[keys[0]]
    canvas[0:h, w + 20:w * 2 + 20] = frames_dict[keys[1]]
    canvas[h + 20:h * 2 + 20, 0:w] = frames_dict[keys[2]]
    canvas[h + 20:h * 2 + 20, w + 20:w * 2 + 20] = frames_dict[keys[3]]
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="hw2_sarsa_qlearning/web/cliff_4agents.mp4")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed-index", type=int, default=0,
                   help="Which of the 10 trained seeds to use")
    args = p.parse_args()

    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    CLIFF = str(runs["cliff_env_id"])
    print(f"Loaded 4-algos run for {CLIFF}, using seed index {args.seed_index}")

    rollouts = {}
    for algo in ["sarsa", "q_learning", "expected_sarsa", "double_q"]:
        Q_seeds = runs[f"cliff_{algo}_Q"]
        Q = Q_seeds[args.seed_index]
        path, rewards = rollout_greedy(CLIFF, Q, max_steps=args.max_steps)
        rollouts[algo] = (path, rewards)
        print(f"  {algo}: path length {len(path) - 1}, final reward {rewards[-1]:.0f}")

    # Number of frames = max path length, hold last frame for 3 fps after each
    # agent terminates so the viewer sees the final state.
    max_steps = max(len(p) for p, _ in rollouts.values())
    n_frames = max_steps + args.fps * 2

    title_map = {
        "sarsa": "SARSA (on-policy)",
        "q_learning": "Q-Learning (off-policy)",
        "expected_sarsa": "Expected SARSA",
        "double_q": "Double Q-Learning",
    }

    print(f"Rendering {n_frames} frames at {args.fps} fps "
          f"({n_frames/args.fps:.1f}s clip)...")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=1)
    try:
        for t in range(n_frames):
            panels = {}
            for algo, (path, rewards) in rollouts.items():
                cap = min(t + 1, len(path))
                panel = render_grid_frame(
                    path[:cap], title=title_map[algo],
                    cum_reward=rewards[cap - 1] if cap > 0 else 0.0,
                )
                panels[algo] = panel
            canvas = composite_4(panels, t)
            writer.append_data(canvas)
    finally:
        writer.close()
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Saved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
