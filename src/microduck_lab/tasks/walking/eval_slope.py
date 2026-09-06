#!/usr/bin/env python3
"""Run a walking ONNX policy on the slope / steps evaluation scene and report
how far it got before falling.

    uv run python -m microduck_lab.tasks.walking.eval_slope --walking ../microduck/policies/alpha_walking.onnx \
        --lane up --lin-vel-x 0.3 --seconds 20
    uv run python -m microduck_lab.tasks.walking.eval_slope --walking ... --lane steps
    uv run python -m microduck_lab.tasks.walking.eval_slope --walking ... --lane down
    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.walking.eval_slope --walking ... --lane up --video videos/eval_slope/slope_baseline.mp4

Scene: src/microduck_lab/models/scene_slope.xml (plane + boxes).
Lanes are separated in y; the robot spawns at x=0 facing +x and is commanded
forward, so "distance along x" is progress into the obstacle. `--ramp-deg`
re-poses both ramps for another angle at load time.

Reports, per run: distance along x, max x reached, height gained, whether the
top / each riser was reached, and when it fell. Same 50 Hz / decimation-4
setup as headless_rollout.py via duck_sim.
"""

import argparse
import math
import os

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim.duck_sim import (  # noqa: E402
    REPO, CONTROL_DT, DECIMATION, FALL_HEIGHT, Recorder, load_scene, make_policy,
)

SCENE = "src/microduck_lab/models/scene_slope.xml"

# Scene geometry (must match scene_slope.xml).
RAMP_START_X = 0.5      # ramp foot (lane up) / ramp head (lane down)
RAMP_RUN = 1.5          # horizontal length of both ramps
RAMP_THICK = 0.10       # box thickness
RAMP_HALF_W = 0.6
LANE_Y = {"up": 0.0, "steps": 1.5, "down": -1.5}
STEP_EDGES_X = (0.5, 0.8, 1.1)          # riser positions, lane "steps"
STEP_RISERS_MM = (10, 15, 20)


def repose_ramps(model, deg):
    """Rewrite the two ramp boxes and the raised platforms for a new angle."""
    a = math.radians(deg)
    rise = RAMP_RUN * math.tan(a)
    surf_len = RAMP_RUN / math.cos(a)
    t = RAMP_THICK
    half = a / 2.0

    def set_geom(name, pos, quat=None, size=None):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_pos[g] = pos
        if quat is not None:
            model.geom_quat[g] = quat
        if size is not None:
            model.geom_size[g] = size

    # Lane up: rises with +x. Rotation -a about y lifts the +x end.
    cx = RAMP_START_X + RAMP_RUN / 2.0 + (t / 2.0) * math.sin(a)
    cz = rise / 2.0 - (t / 2.0) * math.cos(a)
    set_geom("ramp_up", (cx, LANE_Y["up"], cz), (math.cos(half), 0, -math.sin(half), 0),
             (surf_len / 2.0, RAMP_HALF_W, t / 2.0))
    set_geom("ramp_up_top", (RAMP_START_X + RAMP_RUN + 0.75, LANE_Y["up"], rise - t / 2.0))
    # Lane down: starts high at x=0.5, falls with +x. Rotation +a about y.
    cx = RAMP_START_X + RAMP_RUN / 2.0 - (t / 2.0) * math.sin(a)
    cz = rise / 2.0 - (t / 2.0) * math.cos(a)
    set_geom("ramp_down", (cx, LANE_Y["down"], cz), (math.cos(half), 0, math.sin(half), 0),
             (surf_len / 2.0, RAMP_HALF_W, t / 2.0))
    set_geom("ramp_down_top", (RAMP_START_X - 0.75, LANE_Y["down"], rise - t / 2.0))
    return rise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walking", required=True, help="Walking policy ONNX")
    p.add_argument("--lane", choices=sorted(LANE_Y), default="up")
    p.add_argument("--ramp-deg", type=float, default=10.0, help="Ramp angle for lanes up/down")
    p.add_argument("--lin-vel-x", type=float, default=0.3)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--xml", default=None, help="Scene XML (default: scene_slope.xml)")
    p.add_argument("--video", default=None, help="Write an mp4 (needs MUJOCO_GL=egl and a GPU)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    xml = args.xml or os.path.join(REPO, SCENE)
    model, data = load_scene(xml)
    rise = repose_ramps(model, args.ramp_deg)
    policy, adr = make_policy(model, data, walking_onnx_path=args.walking)
    policy.set_vel_cmd(args.lin_vel_x, 0.0, 0.0)

    # Spawn on the lane. Lane "down" starts on the raised platform.
    y0 = LANE_Y[args.lane]
    z0 = 0.125 + (rise if args.lane == "down" else 0.0)
    data.qpos[adr + 0] = 0.0
    data.qpos[adr + 1] = y0
    data.qpos[adr + 2] = z0
    mujoco.mj_forward(model, data)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    rec = Recorder(model, fps=args.fps, azimuth=150.0, elevation=-12.0, distance=1.0) if args.video else None

    steps = int(round(args.seconds / CONTROL_DT))
    max_x = 0.0
    fell_at = None
    z_start = float(data.xpos[trunk][2])
    z_max_gain = 0.0
    for i in range(steps):
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        x, y, z = (float(v) for v in data.xpos[trunk])
        if fell_at is None:
            max_x = max(max_x, x)
            z_max_gain = max(z_max_gain, z - z_start)
        # Fall test relative to the local ground under the trunk (ray down).
        geomid = np.zeros(1, dtype=np.int32)
        d = mujoco.mj_ray(model, data, np.array([x, y, z]), np.array([0, 0, -1.0]), None, 1, -1, geomid)
        ground_clear = d if d >= 0 else z
        if fell_at is None and ground_clear < FALL_HEIGHT:
            fell_at = i * CONTROL_DT
        if rec is not None:
            rec.maybe_capture(i, data)

    x_end = float(data.xpos[trunk][0])
    elapsed = steps * CONTROL_DT
    print()
    print(f"lane={args.lane} ramp={args.ramp_deg:.0f} deg (rise {rise:.3f} m over {RAMP_RUN} m)  "
          f"cmd vx={args.lin_vel_x}  {elapsed:.0f} s")
    print(f"  max x reached: {max_x:+.3f} m (end x {x_end:+.3f} m)")
    print(f"  fell:          {'yes, at %.1f s' % fell_at if fell_at is not None else 'no'}")
    if args.lane == "up":
        on_ramp = min(max(max_x - RAMP_START_X, 0.0), RAMP_RUN)
        print(f"  ramp progress: {on_ramp:.2f} m of {RAMP_RUN} m up the ramp "
              f"({100 * on_ramp / RAMP_RUN:.0f}%), height gained {z_max_gain:.3f} m of {rise:.3f} m")
        print(f"  reached top:   {'yes' if max_x >= RAMP_START_X + RAMP_RUN else 'no'}")
    elif args.lane == "down":
        on_ramp = min(max(max_x - RAMP_START_X, 0.0), RAMP_RUN)
        print(f"  ramp progress: {on_ramp:.2f} m of {RAMP_RUN} m down the ramp ({100 * on_ramp / RAMP_RUN:.0f}%)")
        print(f"  reached bottom:{' yes' if max_x >= RAMP_START_X + RAMP_RUN else ' no'}")
    else:
        cleared = [max_x > ex + 0.15 for ex in STEP_EDGES_X]  # a foot length past the riser
        for mm, ex, ok in zip(STEP_RISERS_MM, STEP_EDGES_X, cleared):
            print(f"  riser {mm:2d} mm at x={ex:.2f}: {'cleared' if ok else 'NOT cleared'}")
        print(f"  height gained: {z_max_gain * 1000:.0f} mm of 45 mm")
    if rec is not None:
        n = rec.write(args.video)
        print(f"  video:         {args.video} ({n} frames)")


if __name__ == "__main__":
    main()
