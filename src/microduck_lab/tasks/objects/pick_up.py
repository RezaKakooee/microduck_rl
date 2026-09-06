#!/usr/bin/env python3
"""The duck finds an object, walks to it, and picks it up in its mouth.

Closed loop like the kick task: nothing is scheduled, every tick reads where the
cube actually is. The duck walks to it, settles, runs the ground-pick policy
(beak to the floor), and the moment the beak reaches the cube it grips, stands
back up with it, and carries it away.

There is no jaw actuator -- the 14 servos are legs and neck -- so the grip is a
weld between the jaw body and the cube, switched on when the beak arrives and
off to drop it. scene_pickup.xml carries the constraint.

    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.objects.pick_up --walking w.onnx \\
        --ground-pick gp.onnx --cube 1.0 0.5 --video pick.mp4
"""

import argparse
import contextlib
import io
import os

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION  # noqa: E402

SCENE = "src/microduck_lab/models/scene_pickup.xml"

# Where to stand so the beak lands ON the cube, measured by tracing mouth_tip
# through a pick in this task (not from a standing start -- the two differ):
# the beak bottoms out 117 mm ahead at z = 30 mm, and the duck shuffles BACK
# about 36 mm as it crouches. So the cube wants to be ~81 mm ahead at the moment
# the pick is triggered, and it drifts out to ~117 mm by the time the beak is
# down. The cube's top face sits at 30 mm, exactly beak height.
PICK_X = 0.081
PICK_Y = 0.0

# Same lesson as the kick: stop INSIDE the target window, a little nearer than
# the mark, because the duck rocks back a few mm as the gait settles and then
# cannot take a small step to correct.
STOP_X = (0.068, 0.090)
STOP_Y = 0.030
POCKET_X = (0.068, 0.095)
POCKET_Y = 0.032

# Beak-to-cube-centre distance that counts as "got it". The cube is 30 mm, so
# 30 mm here means the beak is at its surface -- at 55 mm it gripped from a
# clear 38 mm away, which looks like telekinesis on camera.
GRAB_RADIUS = 0.030

# The mouth is the 15th servo on the real robot, driven by robotd rather than by
# any policy (duck-control/src/model.rs: "neck, head, mouth"), and the RL export
# leaves it out entirely. add_mouth.py puts the joint back as actuator 14, so the
# policy's 14 actions still map straight onto ctrl[0:14] and we drive the mouth
# ourselves -- exactly the split the real robot uses.
MOUTH_OPEN = 0.55
MOUTH_SHUT = 0.0
SETTLE_S = 3.0
BACKOFF_S = 2.0
AIM_TOL = 0.25
TURN_GAIN = 2.5
CRUISE = 0.3


def grip(model, data, eq_id, jaw, cube, cube_q, tip, head):
    """Weld the cube to the jaw, held just beyond the beak tip.

    Welding it where it happens to be lying looks fine on the ground and wrong
    the moment the duck stands: the head pitches up and carries the cube into
    its own shell, so it ends up half inside the face. Instead seat it on the
    beak tip along the head's own forward axis, which is clear of every head
    geom, and weld it there.
    """
    forward = tip - data.xpos[head]
    forward /= np.linalg.norm(forward)
    seat = tip + forward * 0.020

    # Move the cube to the seat (matching the jaw's orientation) before welding,
    # so the constraint starts satisfied and nothing snaps.
    data.qpos[cube_q:cube_q + 3] = seat
    data.qpos[cube_q + 3:cube_q + 7] = data.xquat[jaw]
    data.qvel[cube_q - 0:cube_q + 0] = 0.0
    mujoco.mj_forward(model, data)

    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, data.xquat[jaw])
    relpos = np.zeros(3)
    mujoco.mju_rotVecQuat(relpos, data.xpos[cube] - data.xpos[jaw], neg)
    relquat = np.zeros(4)
    mujoco.mju_mulQuat(relquat, neg, data.xquat[cube])
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = relpos
    model.eq_data[eq_id, 6:10] = relquat
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walking", required=True)
    p.add_argument("--ground-pick", required=True)
    p.add_argument("--cube", type=float, nargs=2, default=[1.0, 0.5], metavar=("X", "Y"))
    p.add_argument("--carry", type=float, default=6.0, help="Seconds to walk carrying it")
    p.add_argument("--seconds", type=float, default=70.0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam-distance", type=float, default=1.2,
                   help="Tracking camera distance; ~0.5 for a close look at the beak")
    args = p.parse_args()

    model, data = duck_sim.load_scene(os.path.join(duck_sim.REPO, SCENE))
    policy, adr = duck_sim.make_policy(
        model, data,
        walking_onnx_path=args.walking,
        ground_pick_onnx_path=args.ground_pick,
    )

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "mouth_grip")
    cube_q = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")])
    mouth_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mouth")
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouth_jaw")
    head_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")

    def apply(action, mouth_cmd):
        """PolicyInference.apply_action assumes ctrl is exactly the 14 policy
        joints; ctrl is 15 wide now, so drive the servos and the mouth here."""
        data.ctrl[:14] = policy.default_pose + action * policy.action_scale
        data.ctrl[mouth_act] = mouth_cmd

    data.qpos[cube_q:cube_q + 7] = [args.cube[0], args.cube[1], 0.015, 1, 0, 0, 0]
    data.eq_active[eq_id] = 0
    mujoco.mj_forward(model, data)

    rec = duck_sim.Recorder(model, fps=args.fps, distance=args.cam_distance) if args.video else None
    steps = int(round(args.seconds / CONTROL_DT))

    phase = "approach"
    settle_until = backoff_until = carry_until = 0.0
    gripped_at = None
    best_beak = 9.9
    lift_peak = 0.015

    print(f"cube at ({args.cube[0]:+.2f}, {args.cube[1]:+.2f}), duck at (0.00, 0.00)")

    for i in range(steps):
        t = i * CONTROL_DT
        quiet = contextlib.redirect_stdout(io.StringIO())

        duck_xy = data.qpos[adr:adr + 2].copy()
        cube_xy = data.xpos[cube][:2].copy()
        yaw = duck_sim.trunk_yaw(data, adr)
        d = cube_xy - duck_xy
        c, s = np.cos(-yaw), np.sin(-yaw)
        rel_x, rel_y = c * d[0] - s * d[1], s * d[0] + c * d[1]

        err_x, err_y = rel_x - PICK_X, rel_y - PICK_Y
        aim = float(np.arctan2(err_y, err_x))

        def drive():
            turn = float(np.clip(TURN_GAIN * aim, -1.5, 1.5))
            with quiet:
                policy.set_vel_cmd(0.0 if abs(aim) > AIM_TOL else CRUISE, 0.0, turn)

        if phase == "approach":
            if STOP_X[0] < rel_x < STOP_X[1] and abs(rel_y - PICK_Y) < STOP_Y:
                phase, settle_until = "settle", t + SETTLE_S
                with quiet:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
                print(f"t={t:5.1f}s  at the cube (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm) -- settling")
            else:
                drive()

        elif phase == "settle" and t >= settle_until:
            if POCKET_X[0] < rel_x < POCKET_X[1] and abs(rel_y - PICK_Y) < POCKET_Y:
                with quiet:
                    policy.trigger_ground_pick()
                phase = "picking"
                print(f"t={t:5.1f}s  beak down (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm)")
            else:
                phase, backoff_until = "backoff", t + BACKOFF_S
                print(f"t={t:5.1f}s  settled off-mark (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm) -- backing off")

        elif phase == "backoff":
            if t >= backoff_until:
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
                phase, carry_until = "carry", t + args.carry
                print(f"t={t:5.1f}s  standing up "
                      f"{'with the cube' if gripped_at else 'empty-beaked'}")

        elif phase == "carry":
            with quiet:
                policy.set_vel_cmd(CRUISE, 0.0, 0.0)
            if t >= carry_until:
                break

        with quiet:
            policy.update_ground_pick_phase(CONTROL_DT)
            policy.update_behavior(CONTROL_DT)
        # Beak opens on the way down and shuts on the cube, like the real robot.
        mouth_cmd = MOUTH_OPEN if (phase == "picking" and gripped_at is None) else MOUTH_SHUT
        apply(policy.infer(), mouth_cmd)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        if gripped_at:
            lift_peak = max(lift_peak, float(data.xpos[cube][2]))
        if rec:
            rec.maybe_capture(i, data)

    cube_end = data.xpos[cube][:2].copy()
    print()
    print(f"closest the beak got:  {best_beak*1000:.0f} mm")
    if gripped_at is None:
        print("task:         never gripped the cube")
    else:
        carried = float(np.linalg.norm(cube_end - np.array(args.cube)))
        print(f"gripped at t={gripped_at:.1f}s")
        print(f"  lifted to:    {lift_peak*1000:.0f} mm (it started at 15 mm)")
        print(f"  carried:      {carried:.2f} m")
        print(f"  still held:   {'yes' if data.eq_active[eq_id] else 'no'}")
        print(f"  task:         {'SUCCESS' if lift_peak > 0.05 and carried > 0.2 else 'gripped but barely moved it'}")
    if rec:
        print(f"  video:        {args.video} ({rec.write(args.video)} frames)")


if __name__ == "__main__":
    main()
