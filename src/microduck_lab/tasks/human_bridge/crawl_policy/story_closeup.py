"""Side-view close-up video of Sister attempting to cross onto Brother at the notch.

Saved as crawl_try6.mp4 for diagnostic inspection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mujoco
import numpy as np

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, VIDEOS_DIR, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L


class SideCamera:
    """Close-up side view of the notch (x in [-0.15, +0.10])."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width, self.height = width, height
        self.renderer = None

    def frame(self, story: Story):
        w = story.world
        if self.renderer is None:
            self.renderer = mujoco.Renderer(w.model, self.height, self.width)
        c = mujoco.MjvCamera()
        # Look right at the notch between ledge and Brother
        c.lookat[:] = (-0.03, 0.0, L.near_z + 0.01)
        c.distance = 0.52
        c.azimuth = 90.0   # Pure lateral side view
        c.elevation = -8.0  # Slightly elevated
        self.renderer.update_scene(w.data, c)
        return self.renderer.render()


def main():
    video_path = VIDEOS_DIR / "crawl_try8.mp4"
    report_path = Path(__file__).resolve().parents[5] / "local_storage" / "hb_dev" / "crawl_policy" / "crawl_try8.json"

    if video_path.exists():
        raise SystemExit(f"{video_path} already exists")

    story = Story(dx=0.0, dy=0.0)

    import imageio.v2 as imageio
    cam = SideCamera(1280, 720)
    writer = imageio.get_writer(
        str(video_path),
        fps=25,
        codec="libx264",
        quality=None,
        macro_block_size=8,
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )

    tick = 0
    try:
        while story.world.t < 13.0 and not (story.phase == "across" and story.settle_time >= 3.0):
            story.tick()
            if tick % 2 == 0:
                writer.append_data(cam.frame(story))
            tick += 1
    except RuntimeError as exc:
        story.failure = str(exc)
        story.note(f"FAILED: {exc}")
    finally:
        for _ in range(30):
            try:
                story.world.step()
            except RuntimeError:
                break
            writer.append_data(cam.frame(story))
        writer.close()

    rear_final = rearmost(story.she)
    success = (story.failure is None) and (story.phase == "across") and (rear_final > L.far_edge + 0.005)
    report = dict(
        success=success,
        failure=story.failure,
        seconds=round(story.world.t, 2),
        furthest_x=round(story.furthest_x, 3),
        final_rearmost=round(rear_final, 3),
        he_up=round(story.he.up(), 3),
        she_up=round(story.she.up(), 3),
        events=story.events,
        video=str(video_path),
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
