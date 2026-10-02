"""Films of trampoline runs: a real-time clip and a slow-motion clip.

Each frame has two views side by side. Small bed:
- left: the duck and the trampoline from the side. The camera height is
  fixed, so the bounce height shows against the frame ring;
- right: a close view of the feet at the bed's rest height, so the gap
  between the soles and the bed shows.
Big bed:
- left: a wide fixed view, with the ruler behind the bed (a mark at every
  robot height above the bed's rest level);
- right: a close view that follows the duck, to show its attitude.

A green "AIR" label means no part of the duck touches anything. A red
"BODY TOUCH" label means a part other than the foot boxes touches the bed or
the floor (a fall).

The slow clip takes a frame every 5 ms inside a time window and plays it at
25 fps: 8x slower than real time.

Needs MUJOCO_GL=egl on a GPU node. Files are never overwritten: each clip
gets the next free number in videos/trampoline/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.tasks.trampoline.world import ROBOT_HEIGHT, SMALL

VIDEO_DIR = Path(paths.VIDEOS) / "trampoline"


def next_free(stem, ext=".mp4"):
    """videos/trampoline/NN_<stem><ext> with NN one above the highest number in use."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for p in VIDEO_DIR.iterdir() if (m := re.match(r"(\d+)_", p.name))]
    n = max(used, default=0) + 1
    return VIDEO_DIR / f"{n:02d}_{stem}{ext}"


class BedFilm:
    def __init__(self, world, slow_window=None, fps=25, slow_dt=0.005, width=640, height=480, camera="both"):
        """camera="both": two views (see the module docstring); "single": one 1280x720
        side view that follows the duck along the bed at a fixed height (as the RL clips)."""
        self.w = world
        m = world.model
        self.camera = camera
        self.tag = ""            # extra caption, e.g. which controller is driving
        if camera == "single":
            width, height = 1280, 720
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
            m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.r = mujoco.Renderer(m, height=height, width=width)
        dt = m.opt.timestep
        self.every = max(1, int(round(1.0 / (fps * dt))))
        self.slow_every = max(1, int(round(slow_dt / dt)))
        self.fps = 1.0 / (self.every * dt)
        self.slow_fps = fps
        self.slow_factor = 1.0 / (self.slow_every * dt * fps)
        self.window = slow_window
        self.n = 0
        self.real, self.slow = [], []

        self.side = mujoco.MjvCamera()
        self.side.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.side.distance, self.side.azimuth, self.side.elevation = 0.75, 90.0, -6.0
        self.feet = mujoco.MjvCamera()
        self.feet.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Inside the ring (radius 0.22 m) and between two frame legs (at 30, 90,
        # 150 ... deg), so neither blocks the view of the feet.
        self.feet.distance, self.feet.azimuth, self.feet.elevation = 0.16, 120.0, 0.0
        self.small = world.frame is SMALL
        if not self.small:
            self.side.distance, self.side.elevation = (2.6 if world.frame.half < 0.5 else 3.2 if world.frame.half < 0.8 else 4.2), -4.0
            self.feet.distance, self.feet.azimuth, self.feet.elevation = 0.7, 60.0, -5.0

    def __call__(self, world):
        """Substep hook: world.on_substep = film."""
        t = world.data.time
        if self.n % self.every == 0:
            self.real.append(self._frame())
        if (self.window is not None and self.window[0] <= t <= self.window[1]
                and self.n % self.slow_every == 0):
            self.slow.append(self._frame())
        self.n += 1

    def _frame(self):
        w, d = self.w, self.w.data
        x, y, z = d.subtree_com[w.trunk]
        top = w.frame.top
        if self.small:
            self.side.lookat[:] = [x, y, top + 0.07]
            self.feet.lookat[:] = [x, y, top - 0.01]
        else:
            self.side.lookat[:] = [0.0, 0.0, top + 0.45]
            self.feet.lookat[:] = [x, y, z]
        cams = (self.side, self.feet)
        if self.camera == "single":
            self.side.distance, self.side.elevation = 1.6, -6.0
            self.side.lookat[:] = [x, y, top + 0.30]
            cams = (self.side,)
        views = []
        for cam in cams:
            self.r.update_scene(d, camera=cam)
            views.append(self.r.render())
        img = np.concatenate(views, axis=1)
        feet, other = w.touching()
        air = not feet and not other
        gap = min(w.sole_heights()) * 1000
        text = f"t {d.time:5.3f} s   bed {1000 * d.qpos[w.bed_q]:+5.1f} mm   sole gap {gap:5.1f} mm"
        if not self.small:
            over = gap / 1000 + d.qpos[w.bed_q]
            text += f"   soles {over:+.2f} m = {over / ROBOT_HEIGHT:.2f} robot heights above the bed"
        if self.tag:
            text += "   " + self.tag
        return self._label(img, text, "AIR" if air else "", other)

    @staticmethod
    def _label(img, text, flag, other):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.text((10, 10), text, fill=(255, 255, 255))
        if flag:
            dr.rectangle((10, 28, 60, 48), fill=(0, 160, 0))
            dr.text((18, 32), flag, fill=(255, 255, 255))
        if other:
            dr.rectangle((70, 28, 170, 48), fill=(200, 0, 0))
            dr.text((76, 32), "BODY TOUCH", fill=(255, 255, 255))
        return np.asarray(im)

    def write(self, stem):
        """Write the real-time clip and (if any) the slow clip. Returns the paths."""
        import imageio.v3 as iio
        out = []
        path = next_free(stem)
        iio.imwrite(path, np.stack(self.real), fps=self.fps)
        out.append(path)
        if self.slow:
            slow_path = path.with_name(f"{path.stem}_slow{self.slow_factor:.0f}x.mp4")
            if slow_path.exists():
                raise FileExistsError(slow_path)
            iio.imwrite(slow_path, np.stack(self.slow), fps=self.slow_fps)
            out.append(slow_path)
        return out
