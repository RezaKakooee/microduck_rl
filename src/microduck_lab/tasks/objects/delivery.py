#!/usr/bin/env python3
"""Delivery: find a cube, pick it up in the beak, carry it to a target zone on
the floor, and drop it there.

This is task_pick_up.py with a second leg. The approach, the stand-off, the
settle-then-fire rule, the grip (a weld to the jaw) and the mouth servo are all
reused from there -- nothing about the pick is re-derived here. What is new:

  deliver   steer at the spot where the duck must STAND for the held cube to
            hang over the target centre (aiming, not strafing: this policy
            cannot sidestep), and stop once the held cube is inside the zone
  settle    zero command, let the gait settle
  release   weld off, mouth open -- the cube falls from the beak
  retreat   a short backward command, then stand (it barely moves, see the doc)

Success = the cube comes to rest on the floor within --radius of the target
centre and the trunk never went below the walking fall height.

    uv run python -m microduck_lab.tasks.objects.delivery --cube 1.0 0.5 --target 1.0 -0.5
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \\
        src/microduck_lab/tasks/objects/delivery.py --cube 1.0 0.5 --target 1.0 -0.5 --video d.mp4
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION, FALL_HEIGHT  # noqa: E402
from microduck_lab.tasks.objects.pick_up import (  # noqa: E402
    grip, PICK_X, PICK_Y, STOP_X, STOP_Y, POCKET_X, POCKET_Y, GRAB_RADIUS,
    MOUTH_OPEN, MOUTH_SHUT, SETTLE_S, BACKOFF_S, AIM_TOL, TURN_GAIN, CRUISE,
)

SCENE = "src/microduck_lab/models/scene_delivery.xml"
# The pretrained policies live in the sibling repo (HANDOFF.md).
POLICIES = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies")

# Where the held cube rides in the duck's yaw frame (+x ahead, +y left) once
# the duck is back on its feet: measured by tracing the cube through a carry
# in this task (see docs/tasks/delivery.md). The delivery approach steers at
# the STAND spot, target minus this offset, because the duck cannot strafe and
# must therefore aim off rather than correct sideways at the end.
HOLD_X = 0.105
HOLD_Y = 0.0

# Stop walking once the held cube is this close to the target centre. Same
# lesson as the pick: the walking policy does nothing below ~0.25 m/s
# commanded, so it cannot take a small step to close a gap -- the stop window
# has to sit well inside the success radius, and the duck walks into it at
# ~2.6 mm per control tick, so it cannot overshoot.
DROP_STOP = 0.06
# After settling, release only if the cube is still inside this fraction of the
# success radius; otherwise back off and come in again.
RELEASE_FRACTION = 0.7
DROP_SETTLE_S = 2.0
RETREAT_S = 2.0
# "On the floor": the cube is 30 mm, so its centre rests at 15 mm flat and
# ~21 mm on an edge. Also require it to have stopped moving.
FLOOR_Z = 0.03
STILL_SPEED = 0.02


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walking", default=os.path.join(POLICIES, "alpha_walking.onnx"))
    p.add_argument("--ground-pick", default=os.path.join(POLICIES, "alpha_ground_pick.onnx"))
    p.add_argument("--cube", type=float, nargs=2, default=[1.0, 0.5], metavar=("X", "Y"))
    p.add_argument("--target", type=float, nargs=2, default=[1.0, -0.5], metavar=("X", "Y"))
    p.add_argument("--radius", type=float, default=0.15,
                   help="Success radius around the target centre (m)")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam-distance", type=float, default=1.2,
                   help="Tracking camera distance; ~0.5 for a close look at the beak")
    p.add_argument("--debug", action="store_true", help="Trace the held cube every 0.5 s")
    args = p.parse_args()

    model, data = duck_sim.load_scene(os.path.join(duck_sim.REPO, SCENE))
    policy, adr = duck_sim.make_policy(
        model, data,
        walking_onnx_path=args.walking,
        ground_pick_onnx_path=args.ground_pick,
    )

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "mouth_grip")
    cube_q = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")])
    cube_v = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")])
    mouth_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mouth")
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouth_jaw")
    head_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")

    # Put the target marker where the target is, and make the disc the size of
    # the success radius, so the video shows exactly what is being scored.
    target = np.array(args.target, dtype=float)
    for name, radius in (("target_zone", args.radius), ("target_centre", None)):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_pos[g, :2] = target
        if radius is not None:
            model.geom_size[g, 0] = radius

    def apply(action, mouth_cmd):
        """Same helper as task_pick_up.py: PolicyInference.apply_action assumes
        ctrl is exactly the 14 policy joints; ctrl is 15 wide here, so drive
        the servos and the mouth ourselves."""
        data.ctrl[:14] = policy.default_pose + action * policy.action_scale
        data.ctrl[mouth_act] = mouth_cmd

    def in_duck_frame(xy, duck_xy, yaw):
        d = xy - duck_xy
        c, s = np.cos(-yaw), np.sin(-yaw)
        return c * d[0] - s * d[1], s * d[0] + c * d[1]

    data.qpos[cube_q:cube_q + 7] = [args.cube[0], args.cube[1], 0.015, 1, 0, 0, 0]
    data.eq_active[eq_id] = 0
    mujoco.mj_forward(model, data)

    rec = duck_sim.Recorder(model, fps=args.fps, distance=args.cam_distance) if args.video else None
    steps = int(round(args.seconds / CONTROL_DT))

    phase = "approach"
    until = 0.0
    gripped_at = released_at = None
    best_beak = 9.9
    lift_peak = 0.015
    trunk_z_min = 9.9
    retries = 0

    print(f"cube at ({args.cube[0]:+.2f}, {args.cube[1]:+.2f}), "
          f"target at ({target[0]:+.2f}, {target[1]:+.2f}) radius {args.radius:.2f} m, "
          f"duck at (0.00, 0.00)")

    for i in range(steps):
        t = i * CONTROL_DT
        quiet = contextlib.redirect_stdout(io.StringIO())

        duck_xy = data.qpos[adr:adr + 2].copy()
        cube_xy = data.xpos[cube][:2].copy()
        yaw = duck_sim.trunk_yaw(data, adr)
        rel_x, rel_y = in_duck_frame(cube_xy, duck_xy, yaw)       # cube, duck frame
        tgt_x, tgt_y = in_duck_frame(target, duck_xy, yaw)        # target, duck frame
        cube_to_target = float(np.linalg.norm(cube_xy - target))

        def drive(err_x, err_y):
            """Turn on the spot if badly aimed, otherwise walk in."""
            aim = float(np.arctan2(err_y, err_x))
            turn = float(np.clip(TURN_GAIN * aim, -1.5, 1.5))
            with quiet:
                policy.set_vel_cmd(0.0 if abs(aim) > AIM_TOL else CRUISE, 0.0, turn)

        def stop():
            with quiet:
                policy.set_vel_cmd(0.0, 0.0, 0.0)

        # ---- leg 1: the pick, exactly as task_pick_up.py does it ----------
        if phase == "approach":
            if STOP_X[0] < rel_x < STOP_X[1] and abs(rel_y - PICK_Y) < STOP_Y:
                phase, until = "settle", t + SETTLE_S
                stop()
                print(f"t={t:5.1f}s  at the cube (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm) -- settling")
            else:
                drive(rel_x - PICK_X, rel_y - PICK_Y)

        elif phase == "settle" and t >= until:
            if POCKET_X[0] < rel_x < POCKET_X[1] and abs(rel_y - PICK_Y) < POCKET_Y:
                with quiet:
                    policy.trigger_ground_pick()
                phase = "picking"
                print(f"t={t:5.1f}s  beak down (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm)")
            else:
                phase, until = "backoff", t + BACKOFF_S
                print(f"t={t:5.1f}s  settled off-mark (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm) -- backing off")

        elif phase == "backoff":
            if t >= until:
                phase = "approach"
            else:
                with quiet:
                    policy.set_vel_cmd(-CRUISE, 0.0, 0.0)

        elif phase == "picking":
            reach = float(np.linalg.norm(data.site_xpos[mouth] - data.xpos[cube]))
            best_beak = min(best_beak, reach)
            if gripped_at is None and reach < GRAB_RADIUS:
                grip(model, data, eq_id, jaw, cube, cube_q,
                     data.site_xpos[mouth].copy(), head_body)
                gripped_at = t
                print(f"t={t:5.1f}s  GOT IT (beak {reach*1000:.0f} mm from the cube)")
            if not policy.ground_pick_mode:
                if gripped_at is None:
                    print(f"t={t:5.1f}s  stood up empty-beaked -- giving up")
                    break
                phase = "deliver"
                print(f"t={t:5.1f}s  standing up with the cube -- "
                      f"target {cube_to_target:.2f} m away")

        # ---- leg 2: carry it to the target and drop it --------------------
        elif phase == "deliver":
            if cube_to_target < DROP_STOP:
                phase, until = "drop_settle", t + DROP_SETTLE_S
                stop()
                print(f"t={t:5.1f}s  over the target (cube {cube_to_target*1000:.0f} mm "
                      f"from centre) -- settling")
            else:
                drive(tgt_x - HOLD_X, tgt_y - HOLD_Y)

        elif phase == "drop_settle" and t >= until:
            if cube_to_target < RELEASE_FRACTION * args.radius:
                data.eq_active[eq_id] = 0
                released_at = t
                phase, until = "release", t + 3.0
                print(f"t={t:5.1f}s  DROPPED (cube {cube_to_target*1000:.0f} mm from centre, "
                      f"{data.xpos[cube][2]*1000:.0f} mm up)")
            else:
                retries += 1
                phase, until = "drop_backoff", t + BACKOFF_S
                print(f"t={t:5.1f}s  settled off the target (cube {cube_to_target*1000:.0f} mm "
                      f"from centre) -- backing off")

        elif phase == "drop_backoff":
            if t >= until:
                phase = "deliver"
            else:
                with quiet:
                    policy.set_vel_cmd(-CRUISE, 0.0, 0.0)

        elif phase == "release":
            speed = float(np.linalg.norm(data.qvel[cube_v:cube_v + 3]))
            landed = data.xpos[cube][2] < FLOOR_Z and speed < STILL_SPEED
            if landed or t >= until:
                phase, until = "retreat", t + RETREAT_S
                print(f"t={t:5.1f}s  cube {'at rest' if landed else 'still moving'} "
                      f"({cube_to_target*1000:.0f} mm from centre) -- standing clear")

        elif phase == "retreat":
            if t >= until:
                phase, until = "done", t + 1.0
                stop()
            else:
                with quiet:
                    policy.set_vel_cmd(-CRUISE, 0.0, 0.0)

        elif phase == "done" and t >= until:
            break

        with quiet:
            policy.update_ground_pick_phase(CONTROL_DT)
            policy.update_behavior(CONTROL_DT)
        # Beak opens on the way down, shuts on the cube, opens again to let go.
        mouth_open = ((phase == "picking" and gripped_at is None)
                      or phase == "release")
        apply(policy.infer(), MOUTH_OPEN if mouth_open else MOUTH_SHUT)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        trunk_z_min = min(trunk_z_min, float(data.qpos[adr + 2]))
        if gripped_at and released_at is None:
            lift_peak = max(lift_peak, float(data.xpos[cube][2]))
        if rec:
            rec.maybe_capture(i, data)

        if args.debug and i % 25 == 0:
            print(f"    t={t:5.1f} {phase:12s} duck=({duck_xy[0]:+.2f},{duck_xy[1]:+.2f}) "
                  f"cube rel=({rel_x*1000:+.0f},{rel_y*1000:+.0f})mm z={data.xpos[cube][2]*1000:.0f} "
                  f"tgt rel=({tgt_x*1000:+.0f},{tgt_y*1000:+.0f})mm "
                  f"cube->tgt {cube_to_target*1000:.0f}mm trunk z {data.qpos[adr+2]*1000:.0f}")

    # ---- summary -----------------------------------------------------------
    cube_end = data.xpos[cube].copy()
    dist = float(np.linalg.norm(cube_end[:2] - target))
    speed = float(np.linalg.norm(data.qvel[cube_v:cube_v + 3]))
    on_floor = cube_end[2] < FLOOR_Z and speed < STILL_SPEED
    fell = trunk_z_min < FALL_HEIGHT
    print()
    print(f"closest the beak got:  {best_beak*1000:.0f} mm")
    print(f"picked at:      {'t=%.1fs' % gripped_at if gripped_at is not None else 'never'}")
    print(f"dropped at:     {'t=%.1fs' % released_at if released_at is not None else 'never'}"
          f"{'' if not retries else '  (%d retry)' % retries}")
    if gripped_at is not None:
        print(f"lifted to:      {lift_peak*1000:.0f} mm")
    print(f"cube final:     ({cube_end[0]:+.3f}, {cube_end[1]:+.3f}, z {cube_end[2]*1000:.0f} mm), "
          f"speed {speed*1000:.0f} mm/s")
    print(f"target:         ({target[0]:+.3f}, {target[1]:+.3f})")
    print(f"cube to target: {dist:.3f} m  (radius {args.radius:.2f})")
    print(f"on the floor:   {'yes' if on_floor else 'no'}")
    print(f"duck fell:      {'YES (trunk z %.0f mm)' % (trunk_z_min*1000) if fell else 'no (min trunk z %.0f mm)' % (trunk_z_min*1000)}")
    ok = (gripped_at is not None and released_at is not None
          and dist < args.radius and on_floor and not fell)
    print(f"task:           {'SUCCESS' if ok else 'FAIL'}")
    if rec:
        print(f"video:          {args.video} ({rec.write(args.video)} frames)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
