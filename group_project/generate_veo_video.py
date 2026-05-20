"""Generate an explainer video with Google Veo for the report front page.

Reads the API key from the GEMINI_API_KEY environment variable — never
commit the key. Run with:

    GEMINI_API_KEY=...  python -m group_project.generate_veo_video
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="veo-2.0-generate-001")
    p.add_argument("--prompt", default=(
        "Educational motion-graphic animation, dark background, minimalist "
        "technical engineering style. A small 2D bipedal stick-figure robot in a "
        "physics simulation. Left half labeled 'RANDOM' — the robot wobbles, "
        "stumbles, and falls flat onto a checkered floor within two seconds. "
        "Right half labeled 'TRAINED PPO' — the same robot walks confidently "
        "forward across the same checkered floor, smooth gait, never falls. "
        "Side view, no humans, clean monochrome with orange accents. "
        "Caption overlay: 'PPO learned to map joint angles to motor torques.'"))
    p.add_argument("--aspect", default="16:9")
    p.add_argument("--out", default="group_project/web/veo_explainer.mp4")
    p.add_argument("--duration", type=int, default=8)
    p.add_argument("--person-gen", default="allow_adult",
                   choices=["dont_allow", "allow_adult", "allow_all"])
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY env var. Aborting.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    print(f"[VEO] model={args.model}  aspect={args.aspect}  duration={args.duration}s")
    print(f"[VEO] prompt: {args.prompt[:120]}...")
    print("[VEO] starting generation (may take 30-120s)...")

    operation = client.models.generate_videos(
        model=args.model,
        prompt=args.prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=args.aspect,
            number_of_videos=1,
            duration_seconds=args.duration,
            person_generation=args.person_gen,
        ),
    )

    t0 = time.time()
    while not operation.done:
        time.sleep(10)
        operation = client.operations.get(operation)
        elapsed = time.time() - t0
        print(f"  ... polling (t={elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"[VEO] done in {elapsed:.0f}s")

    if operation.error:
        sys.exit(f"[VEO] generation failed: {operation.error}")

    result = operation.result if hasattr(operation, "result") else operation.response
    if not result or not getattr(result, "generated_videos", None):
        sys.exit(f"[VEO] no video returned. Operation: {operation}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video = result.generated_videos[0].video
    # The SDK may already include bytes, or require a download call.
    try:
        client.files.download(file=video)
    except Exception:
        pass  # already inlined

    if hasattr(video, "save"):
        video.save(str(out_path))
    elif hasattr(video, "video_bytes"):
        out_path.write_bytes(video.video_bytes)
    else:
        sys.exit(f"[VEO] cannot persist video object: {type(video)}")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[VEO] saved {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
