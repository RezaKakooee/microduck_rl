"""Name the geoms that fight for the same pixels.

Z-fighting is invisible in a single frame and obvious in motion: two surfaces
at nearly the same depth swap which one wins, pixel by pixel, as the camera
moves. This renders the same view twice with the camera nudged by a hair and
reports which geoms trade places. A correctly layered scene barely changes;
a fighting pair flips over a large, connected patch.

    python -m microduck_lab.tools.find_flicker --look she_jaw_soft
"""

from __future__ import annotations

import argparse
import collections

import mujoco
import numpy as np


def segment(model, data, cam, width, height):
    r = mujoco.Renderer(model, height=height, width=width)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=cam)
    seg = r.render()[:, :, 0].copy()          # geom id per pixel, -1 for nothing
    r.close()
    return seg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default="she_jaw_soft", help="body to point the camera at")
    ap.add_argument("--distance", type=float, default=0.22)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-8.0)
    ap.add_argument("--nudge", type=float, default=0.02, help="camera move, in degrees")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--min-pixels", type=int, default=40)
    args = ap.parse_args()

    from microduck_lab.film import scene_v2 as scene
    model, data = scene.build_model()
    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.look)
    if bid < 0:
        raise SystemExit(f"no body called {args.look!r}")

    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.xpos[bid]
    cam.distance, cam.elevation = args.distance, args.elevation

    cam.azimuth = args.azimuth
    a = segment(model, data, cam, args.width, args.height)
    cam.azimuth = args.azimuth + args.nudge
    b = segment(model, data, cam, args.width, args.height)

    name = lambda g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                      or f"geom{int(g)}") if g >= 0 else "background"

    swaps = collections.Counter()
    changed = (a != b) & (a >= 0) & (b >= 0)
    for ga, gb in zip(a[changed], b[changed]):
        swaps[tuple(sorted((int(ga), int(gb))))] += 1

    total = int(changed.sum())
    print(f"camera nudged {args.nudge} deg; {total} of {a.size} pixels changed geom")
    print(f"showing pairs over {args.min_pixels} pixels -- a big count in a still "
          f"scene means those two surfaces fight:\n")
    for (g1, g2), n in swaps.most_common():
        if n < args.min_pixels:
            break
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])) or "?"
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])) or "?"
        print(f"  {n:6d} px   {name(g1):<22} ({b1}) <-> {name(g2):<22} ({b2})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
