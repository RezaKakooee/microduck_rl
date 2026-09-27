"""Human bridge full story execution with scripted controller and video filming.

Videos saved into: videos/human_bridge/crawl_policy/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mujoco
import numpy as np

from scipy.spatial.transform import Rotation as R

from microduck_lab.tasks.human_bridge.crawl_policy.world import World
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK, LieDown, HoldStraight, smooth
from microduck_lab.tasks.human_bridge.crawl_policy.controller import SisterScriptedController, CRAWL_BASE

STAND_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float64)

START_X = -0.22
DROP_Z = min(L.near_z, L.far_z) + 0.05
VIDEOS_DIR = Path(__file__).resolve().parents[5] / "videos" / "human_bridge" / "crawl_policy"


def rearmost(duck) -> float:
    """x position of the back-most point of her solid geoms."""
    m, d = duck.model, duck.data
    g = np.asarray(duck.solid)
    return float(np.min(d.geom_xpos[g, 0] - m.geom_rbound[g]))


class Story:
    PHASES = ("wait", "he_lies", "cross", "across")

    def __init__(self, dx: float = 0.0, dy: float = 0.0, verbose: bool = True):
        self.world = w = World(L.design())
        self.he, self.she = w.ducks["he"], w.ducks["she"]
        self.controller = SisterScriptedController(self.she)

        # Place brother on near step in STAND_POSE
        self.he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, STAND_POSE)

        # Place sister on near top ledge in prone crawl orientation
        start_x = START_X + dx
        start_y = 0.0 + dy
        rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        q_xyzw = rot.as_quat()
        quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        self.she.place((start_x, start_y, L.near_z + 0.05), q=CRAWL_BASE)
        self.she.data.qpos[self.she.adr + 3 : self.she.adr + 7] = quat_wxyz
        mujoco.mj_forward(w.model, w.data)

        w.start()
        self.verbose = verbose
        self.phase, self.t_phase = "wait", 0.0
        self.events, self.failure = [], None
        self.scratch = {}
        self.max_offset = 0.0
        self.furthest_x = start_x
        self.settle_time = 0.0

    def note(self, text: str):
        self.events.append(dict(t=round(self.world.t, 2), event=text))
        if self.verbose:
            print(f"{self.world.t:6.2f}s  {text}", flush=True)

    def go(self, phase: str):
        self.note(f"{self.phase} -> {phase}")
        self.phase, self.t_phase = phase, self.world.t
        self.scratch = {}
        self.controller.set_phase(phase, self.world.t)

    def tick(self):
        w, he, she = self.world, self.he, self.she
        t = w.t
        u = t - self.t_phase

        # Sister is controlled across all phases by the scripted controller
        q_she = self.controller.compute_targets(t)
        she.hold(q_she)

        if self.phase == "wait":
            he.hold(STAND_POSE)
            if u >= 1.5:
                self.go("he_lies")

        elif self.phase == "he_lies":
            start = self.scratch.setdefault("start", he.target.copy())
            lie_progress = smooth(u / 2.0)
            he.hold((1.0 - lie_progress) * start + lie_progress * PLANK)
            if u >= 3.5:
                self.go("cross")

        elif self.phase == "cross":
            self.scratch.setdefault("hold", HoldStraight(he))()

            pos = she.pos()
            self.furthest_x = max(self.furthest_x, pos[0])
            if L.near_edge < pos[0] < L.far_edge:
                self.max_offset = max(self.max_offset, abs(pos[1]))

            rear = rearmost(she)
            if rear > L.far_edge + 0.005:
                self.note(f"all of her is on the far ledge after {u:.2f} s (rearmost {rear:.3f} m)")
                self.go("across")

        elif self.phase == "across":
            self.scratch.setdefault("hold", HoldStraight(he))()
            self.furthest_x = max(self.furthest_x, she.pos()[0])
            self.settle_time += 0.02

        w.step()
        self.check()

    def check(self):
        she, he = self.she, self.he
        # Rolled completely over onto side or upside down (> 70 deg)
        r_roll = abs(float(she.data.xmat[she.trunk, 7]))
        if r_roll > 1.20:
            raise RuntimeError(f"she rolled onto side at x={she.pos()[0]:.3f} m, roll={r_roll:.3f}")
        # Drifted off bridge sideways
        if abs(she.pos()[1]) > 0.18:
            raise RuntimeError(f"she drifted off bridge sideways: y={she.pos()[1]:.3f} m")
        # Dropped into gap
        if self.phase in ("cross", "across") and she.pos()[2] < L.shelf_z - 0.05:
            raise RuntimeError(f"she dropped into gap at x={she.pos()[0]:.3f}, z={she.pos()[2]:.3f} m")
        # Brother must remain stable
        if self.phase in ("cross", "across") and abs(he.up()) > 0.35:
            raise RuntimeError(f"brother was knocked loose: he.up() = {he.up():.3f}")


class Camera:
    """In front of sister, tracking her across the set."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width, self.height = width, height
        self.renderer = None

    def frame(self, story: Story):
        w = story.world
        if self.renderer is None:
            self.renderer = mujoco.Renderer(w.model, self.height, self.width)
        c = mujoco.MjvCamera()
        x = float(np.clip(story.she.pos()[0], -0.25, 0.40))
        c.lookat[:] = (0.05 + 0.6 * (x - 0.05), 0.0, L.near_z)
        c.distance = 0.95
        c.azimuth = 90.0 - 20.0 * np.cos(w.t / 10.0)
        c.elevation = -20.0
        self.renderer.update_scene(w.data, c)
        return self.renderer.render()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--step-period", type=float, default=0.20)
    ap.add_argument("--stride", type=float, default=0.18)
    ap.add_argument("--lift", type=float, default=0.32)
    ap.add_argument("--video")
    ap.add_argument("--report")
    a = ap.parse_args()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if a.video and Path(a.video).exists():
        raise SystemExit(f"{a.video} already exists; videos are never overwritten")

    story = Story(dx=a.dx, dy=a.dy)
    story.controller.step_period = a.step_period
    story.controller.stride_amp = a.stride
    story.controller.lift_amp = a.lift

    writer = cam = None
    if a.video:
        import imageio.v2 as imageio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        cam = Camera(1280, 720)
        writer = imageio.get_writer(
            a.video,
            fps=25,
            codec="libx264",
            quality=None,
            macro_block_size=8,
            output_params=["-crf", "22", "-movflags", "+faststart"],
        )

    tick = 0
    try:
        while story.world.t < a.seconds and not (story.phase == "across" and story.settle_time >= 3.0):
            story.tick()
            if writer is not None and tick % 2 == 0:
                writer.append_data(cam.frame(story))
            tick += 1
        if story.phase != "across":
            raise RuntimeError(f"timed out in phase {story.phase} at x={story.she.pos()[0]:.3f} m")
    except RuntimeError as exc:
        story.failure = str(exc)
        story.note(f"FAILED: {exc}")
    finally:
        if writer is not None:
            for _ in range(40):
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
        dx=a.dx,
        dy=a.dy,
        furthest_x=round(story.furthest_x, 3),
        final_rearmost=round(rear_final, 3),
        max_offset_mm=round(1000 * story.max_offset, 1),
        he_up=round(story.he.up(), 3),
        she_up=round(story.she.up(), 3),
        events=story.events,
        video=a.video,
    )
    text = json.dumps(report, indent=2)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text + "\n")
    print(text)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
