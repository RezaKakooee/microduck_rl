"""Films of jump attempts: a real-time clip and a slow-motion clip.

A flight of 50 ms is about one frame at 25 fps, so a real-time clip alone
cannot show it. The slow clip captures every physics substep (5 ms) inside a
time window and plays it at 25 fps: 8x slower than real time.

Each frame has two views side by side:
- left: the whole duck from the side, camera at a fixed height;
- right: a close view of the feet at floor level, so a few mm of gap shows.
A label says "AIR" when no part of the duck touches the floor.

Needs MUJOCO_GL=egl on a GPU node. Files are never overwritten: each clip
gets the next free number in videos/jump/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import mujoco

from microduck_lab import paths

VIDEO_DIR = Path(paths.REPO) / "videos" / "jump"


def next_free(stem):
    """videos/jump/NN_<stem>.mp4 with NN one above the highest number in use."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for p in VIDEO_DIR.iterdir() if (m := re.match(r"(\d+)_", p.name))]
    n = max(used, default=0) + 1
    return VIDEO_DIR / f"{n:02d}_{stem}.mp4"


class JumpFilm:
    def __init__(self, world, slow_window=None, fps=25, width=640, height=480):
        self.w = world
        m = world.model
        self.r = mujoco.Renderer(m, height=height, width=width)
        self.dt = m.opt.timestep
        self.every = max(1, int(round(1.0 / (fps * self.dt))))
        self.fps = 1.0 / (self.every * self.dt)
        self.slow_fps = fps
        self.slow_factor = (1.0 / self.dt) / fps
        self.window = slow_window
        self.n = 0
        self.real, self.slow = [], []

        self.side = mujoco.MjvCamera()
        self.side.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.side.distance, self.side.azimuth, self.side.elevation = 0.55, 90.0, -8.0
        self.feet = mujoco.MjvCamera()
        self.feet.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.feet.distance, self.feet.azimuth, self.feet.elevation = 0.16, 90.0, 0.0

    def __call__(self, world):
        """Substep hook: world.on_substep = film."""
        t = world.data.time
        in_slow = self.window is not None and self.window[0] <= t <= self.window[1]
        if self.n % self.every == 0:
            self.real.append(self._frame())
        if in_slow:
            self.slow.append(self._frame())
        self.n += 1

    def _frame(self):
        w, d = self.w, self.w.data
        x, y = d.subtree_com[w.trunk][:2]
        self.side.lookat[:] = [x, y, 0.10]
        self.feet.lookat[:] = [x, y, 0.015]
        views = []
        for cam in (self.side, self.feet):
            self.r.update_scene(d, camera=cam)
            views.append(self.r.render())
        img = np.concatenate(views, axis=1)
        feet, other = w.touching()
        air = not feet and not other
        return self._label(img, f"t {d.time:5.3f} s", "AIR" if air else "", other)

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
            dr.rectangle((70, 28, 190, 48), fill=(200, 0, 0))
            dr.text((76, 32), "BODY ON FLOOR", fill=(255, 255, 255))
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
