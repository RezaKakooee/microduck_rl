#!/usr/bin/env python3
"""A hand-written spiral controller — no policy, no learning.

Six RL runs have failed to produce a one-legged glide, and each failure was read
as a reward-shaping problem. This script asks the prior question: CAN this robot
hold the pose at all? If a hand-tuned balance controller cannot do it, no reward
function will, and the answer is to change the task or the morphology.

It is also useful if it works: a scripted expert is a reference motion, which is
what an imitation reward needs.

Control:
  free leg   -> commanded straight to the extended pose, open loop.
  support leg-> PD on trunk roll and pitch, read from projected gravity.
                roll  is corrected with the support hip_roll,
                pitch with the support ankle.
Everything else holds the home pose.

    uv run python -m microduck_lab.tasks.skating.expert_spiral --search      # grid-search the gains
    uv run python -m microduck_lab.tasks.skating.expert_spiral --kp-roll 2.0 --kd-roll 0.15 --seconds 10
"""

import argparse
import contextlib
import io
import os

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION  # noqa: E402

HIP_EXTENSION = 1.1
WHEEL_RADIUS = 0.0175

# Canonical 14-servo layout: 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
# ankle), 5-8 neck/head, 9-13 right leg.
L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 1, 2, 3, 4
R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 10, 11, 12, 13


def lean_for_turn(speed, yaw_rate):
    """Lean angle that a banked turn holds up: tan(theta) = v * omega / g.

    A single roller is a line contact fore-aft and a knife edge sideways, so
    there is NO lateral support width and a straight-line one-wheel glide has no
    restoring force at all — measured, it topples from 0.8 to 64 degrees of roll
    in 0.32 s whatever the gains. A real spiral is skated on a CURVE: the skater
    leans in, and centripetal force holds the lean up. That is the only way this
    pose is stable, so the controller commands a turn.
    """
    if yaw_rate == 0.0:
        return 0.0
    return float(np.arctan2(speed * yaw_rate, 9.81))


def set_ice(model, mu):
    """Make floor AND feet slippery. MuJoCo takes the MAX of the two geoms'
    friction, so lowering only one side does nothing (verified in the ice task)."""
    import re
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name == "floor" or (name and re.match(r"^(left|right)_foot_collision$", name)):
            model.geom_friction[g, 0] = mu


def run(kp_roll, kd_roll, kp_pitch, kd_pitch, seconds=10.0, speed=0.4,
        free="left", yaw_rate=0.0, steer=0.0, feet="rollers",
        hip_roll_shift=0.44, ice_mu=None, video=None, verbose=False):
    """Hold the spiral on one leg. `feet` picks rollers or the normal flat sole.

    On rollers the support is a line contact fore-aft and a knife edge sideways,
    which is why it topples in 0.30 s whatever the gains. A normal foot has a
    real support polygon in BOTH directions, so one-legged balance should be a
    genuinely different problem.
    """
    buf = io.StringIO()
    rollers = feet == "rollers"
    model, data = duck_sim.load_scene(
        "rollers" if rollers else "walk", roller_friction=rollers
    )
    if ice_mu is not None:
        set_ice(model, ice_mu)
    with contextlib.redirect_stdout(buf):
        policy, adr = duck_sim.make_policy(
            model, data,
            walking_onnx_path=os.path.join(duck_sim.REPO, "spiral_v5.onnx"),
        )
    # We only wanted the pose/limit setup; the network is never queried.
    target = policy.default_pose.copy()

    if free == "left":
        target[L_HIP_PITCH], target[L_KNEE] = HIP_EXTENSION, 0.0
        sup_roll, sup_ankle, roll_sign = R_HIP_ROLL, R_ANKLE, 1.0
        shift_sign = 1.0
    else:
        target[R_HIP_PITCH], target[R_KNEE] = -HIP_EXTENSION, 0.0
        sup_roll, sup_ankle, roll_sign = L_HIP_ROLL, L_ANKLE, -1.0
        shift_sign = -1.0

    # Shift the body sideways OVER the support foot before standing on it.
    # Without this the centre of mass starts 42.8 mm outside the support foot —
    # already falling at t=0, which no controller can recover. This is what a
    # person does before lifting a foot: transfer the weight first. Measured:
    # +0.44 rad on both hip_rolls puts the CoM within a few mm of the foot.
    target[L_HIP_ROLL] += shift_sign * hip_roll_shift
    target[R_HIP_ROLL] += shift_sign * hip_roll_shift

    # Lean into the turn, and steer the support wheel along the circle.
    lean = lean_for_turn(speed, yaw_rate)
    if free == "left":
        lean = -lean          # turning left: lean left, i.e. toward the free leg
    target[sup_roll] += lean
    target[R_HIP_ROLL if free == "left" else L_HIP_ROLL] += 0.0
    sup_yaw = R_HIP_ROLL - 1 if free == "left" else L_HIP_ROLL - 1  # hip_yaw
    target[sup_yaw] += steer

    # Start already moving. On rollers the wheels spin to match. On normal feet
    # this only makes sense on ice, where the foot slides: a one-legged GLIDE.
    data.qvel[0] = speed
    if rollers:
        for name in ("passive_LF_wheel", "passive_LR_wheel",
                     "passive_RF_wheel", "passive_RR_wheel"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qvel[int(model.jnt_dofadr[jid])] = speed / WHEEL_RADIUS
    # And already in the pose, so we measure holding rather than entering.
    for idx, qidx in enumerate(policy.joint_qpos_indices):
        data.qpos[qidx] = target[idx]
    mujoco.mj_forward(model, data)

    # Lifting one leg moves the whole body relative to the support wheel, so the
    # trunk height from the neutral reset no longer fits: measured, the support
    # foot started 11.7 mm BELOW the floor and the contact impulse threw the duck
    # sideways in 0.28 s regardless of any gain, lean or steer. Drop the robot
    # onto the ground instead: find the lowest collision point and shift the
    # trunk so it rests exactly on the floor.
    # Binary-search the trunk height for the lowest position with NO contact.
    # (Not geom_size: mesh geoms carry no meaningful radius, so an earlier
    # version of this silently did nothing and every gain gave the same 0.28 s.)
    lo, hi = 0.05, 0.30
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        data.qpos[adr + 2] = mid
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            lo = mid
        else:
            hi = mid
    data.qpos[adr + 2] = hi
    mujoco.mj_forward(model, data)

    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
             for s in ("left_foot", "right_foot")]
    free_site = sites[0] if free == "left" else sites[1]
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    rec = duck_sim.Recorder(model, distance=0.9) if video else None
    steps = int(round(seconds / CONTROL_DT))
    prev_roll = prev_pitch = 0.0
    held = 0.0
    lost = False

    for i in range(steps):
        # Projected gravity in the trunk frame: 0,0,-1 when upright.
        quat = data.qpos[adr + 3:adr + 7].astype(np.float32)
        g = policy.quat_rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32))
        roll = float(np.arctan2(g[1], -g[2]))
        pitch = float(np.arctan2(g[0], -g[2]))
        d_roll, d_pitch = (roll - prev_roll) / CONTROL_DT, (pitch - prev_pitch) / CONTROL_DT
        prev_roll, prev_pitch = roll, pitch

        ctrl = target.copy()
        # PD holds the roll at the BANK angle, not at zero.
        roll_err = roll - lean
        ctrl[sup_roll] += roll_sign * (kp_roll * roll_err + kd_roll * d_roll)
        ctrl[sup_ankle] += kp_pitch * pitch + kd_pitch * d_pitch
        data.ctrl[:14] = ctrl
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        if rec:
            rec.maybe_capture(i, data)

        t = i * CONTROL_DT
        free_down = float(data.site_xpos[free_site][2]) < 0.015
        fallen = float(data.xpos[trunk][2]) < 0.07
        if free_down or fallen:
            # When recording, keep simulating so the clip shows what happens
            # after the pose is lost, not a third of a second of nothing.
            if rec is None:
                break
            lost = True
        elif not lost:
            held = t
        if verbose and i % 50 == 0:
            print(f"  t={t:4.1f}s roll={np.degrees(roll):+6.1f}deg "
                  f"pitch={np.degrees(pitch):+6.1f}deg")

    if rec:
        print(f"  video: {video} ({rec.write(video)} frames)")
    return held


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="store_true", help="Grid-search the gains")
    p.add_argument("--kp-roll", type=float, default=2.0)
    p.add_argument("--kd-roll", type=float, default=0.1)
    p.add_argument("--kp-pitch", type=float, default=1.0)
    p.add_argument("--kd-pitch", type=float, default=0.05)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--free", choices=("left", "right"), default="left")
    p.add_argument("--feet", choices=("rollers", "normal"), default="rollers")
    p.add_argument("--hip-roll-shift", type=float, default=0.44,
                   help="Sideways weight shift onto the support foot (rad)")
    p.add_argument("--ice-mu", type=float, default=None,
                   help="Sliding friction for floor and feet (e.g. 0.1 = ice)")
    p.add_argument("--yaw-rate", type=float, default=0.0,
                   help="Turn rate rad/s; sets the bank angle. 0 = straight line.")
    p.add_argument("--steer", type=float, default=0.0,
                   help="Support hip_yaw offset, to point the wheel along the curve")
    p.add_argument("--video", type=str, default=None)
    args = p.parse_args()

    if args.search:
        best = (0.0, None)
        for yaw in (0.0, 1.0, 2.0, 3.0, 4.0):
            for steer in (0.0, 0.15, 0.3):
                for kp_r in (0.0, 2.0, 6.0):
                    held = run(kp_r, 0.15, 1.0, 0.05, seconds=args.seconds,
                               speed=args.speed, free=args.free, feet=args.feet,
                               yaw_rate=yaw, steer=steer)
                    if held > best[0]:
                        best = (held, (yaw, steer, kp_r))
                        print(f"  new best {held:5.2f}s  yaw={yaw} steer={steer} kp_roll={kp_r}")
        print(f"\nbest hold {best[0]:.2f}s at yaw/steer/kp_roll = {best[1]}")
    else:
        held = run(args.kp_roll, args.kd_roll, args.kp_pitch, args.kd_pitch,
                   seconds=args.seconds, speed=args.speed, free=args.free,
                   yaw_rate=args.yaw_rate, steer=args.steer, feet=args.feet,
                   hip_roll_shift=args.hip_roll_shift, ice_mu=args.ice_mu,
                   video=args.video, verbose=True)
        print(f"held the spiral for {held:.2f}s of {args.seconds:.0f}s")


if __name__ == "__main__":
    main()
