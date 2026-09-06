#!/usr/bin/env python3
"""A task, not a replay: the duck finds a ball, walks to it, and kicks it.

The behaviour clips fire a policy at a fixed time and hope the duck is in a
sensible state. Here nothing is scheduled -- every tick reads where the ball
actually is, steers toward it, and the kick fires only once the ball is in the
pocket in front of a foot. Two policies run behind the same 61-dim observation,
swapped mid-episode, which is what the runtime does on the real robot.

    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.objects.kick_ball --ball 1.2 0.6 --video kick.mp4

Needs a GPU for --video (EGL); the task itself runs anywhere.
"""

import argparse
import contextlib
import io
import sys

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION  # noqa: E402

# Where the kick policy expects the ball, in the duck's yaw frame: 90 mm ahead
# and 42 mm to one side (infer_policy.py's BALL_OFFSET_*, matching training's
# reset_ball_in_front_of_foot).
POCKET_X = 0.09
POCKET_Y = 0.042

# How much slack we allow before firing. The approach is coarse: the walking
# policy tracks ~0.13 m/s against a 0.3 command (~2.6 mm per control tick) and
# does not move at all below ~0.25.
# Measured, by teleporting the ball off-pocket and reading how far it flies:
# 0 error -> 3.9 m, +15 mm forward -> 1.8 m, +30 mm forward -> dead miss.
# Lateral is more forgiving: +/-15 mm still gets 3.3-3.8 m. Forward is the axis
# that has to be right, so the window below is tight on x and looser on y.
POCKET_TOL_X = (0.080, 0.112)
POCKET_TOL_Y = 0.022

# This policy CANNOT strafe -- a lateral command produces exactly zero lateral
# motion -- so the duck cannot sidestep into alignment at the end. It has to
# aim off during the approach: steer at the spot where the duck must STAND for
# the ball to land in the pocket, not at the ball itself.

AIM_TOL = 0.25      # rad of bearing error we will walk through rather than turn out
TURN_GAIN = 2.5
CRUISE = 0.3        # the walking policy's max, and the only speed that moves it

# The kick policies were trained to start from a stand and end standing. Fired
# mid-stride -- duck on one foot, body pitched -- the swing misses entirely even
# with the ball perfectly placed. So stop, let the gait settle, then fire.
#
# Stop INSIDE the pocket, not short of it: the duck coasts only a few mm after
# the command goes to zero, and the walking policy's deadband means it cannot
# then take a small step to close a gap. Parked at 140 mm it stays at 140 mm
# forever. At ~2.6 mm of travel per control tick it lands in this window easily.
# Shifted ~10 mm nearer than the pocket: the duck rocks BACKWARD about 7 mm as
# the gait settles, so stopping at 78-98 mm lands it on 85-105 mm when it fires.
APPROACH_STOP_X = (0.075, 0.098)
APPROACH_STOP_Y = 0.022
SETTLE_S = 3.0
BACKOFF_S = 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walking", required=True)
    p.add_argument("--kick-left", required=True)
    p.add_argument("--kick-right", required=True)
    p.add_argument("--ball", type=float, nargs=2, default=[1.2, 0.6],
                   metavar=("X", "Y"), help="Where to put the ball (m)")
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    model, data = duck_sim.load_scene("ball")
    policy, adr = duck_sim.make_policy(
        model, data,
        walking_onnx_path=args.walking,
        kick_left_onnx_path=args.kick_left,
        kick_right_onnx_path=args.kick_right,
    )

    ball_q, ball_v = policy.ball_qpos_adr, policy.ball_qvel_adr
    if ball_q is None:
        sys.exit("no ball in this scene")
    data.qpos[ball_q:ball_q + 7] = [args.ball[0], args.ball[1], 0.035, 1, 0, 0, 0]
    data.qvel[ball_v:ball_v + 6] = 0.0
    mujoco.mj_forward(model, data)

    rec = duck_sim.Recorder(model, fps=args.fps, distance=1.4) if args.video else None
    steps = int(round(args.seconds / CONTROL_DT))

    # Which foot, decided once at the start from which side the ball is on, and
    # then held: it fixes which side of the ball the duck aims for all the way in.
    foot = None

    phase = "approach"
    settle_until = 0.0
    backoff_until = 0.0
    kicked_at = None
    ball_at_kick = None

    print(f"ball at ({args.ball[0]:+.2f}, {args.ball[1]:+.2f}), duck at (0.00, 0.00)")

    for i in range(steps):
        t = i * CONTROL_DT

        duck_xy = data.qpos[adr:adr + 2].copy()
        ball_xy = data.qpos[ball_q:ball_q + 2].copy()
        yaw = duck_sim.trunk_yaw(data, adr)

        # Ball in the duck's own frame: +x ahead, +y to its left.
        d = ball_xy - duck_xy
        c, s = np.cos(-yaw), np.sin(-yaw)
        rel_x, rel_y = c * d[0] - s * d[1], s * d[0] + c * d[1]
        bearing = float(np.arctan2(rel_y, rel_x))

        quiet = contextlib.redirect_stdout(io.StringIO())

        # Which foot, decided once from the side the ball starts on, then held:
        # it fixes which side of the ball the duck aims for all the way in.
        if foot is None:
            foot = "kick_left" if bearing >= 0 else "kick_right"
            want_y = POCKET_Y if foot == "kick_left" else -POCKET_Y
            print(f"aiming for the {foot.split('_')[1]} foot "
                  f"(ball {'left' if bearing >= 0 else 'right'} of us)")

        # Error to where the duck must STAND for the ball to be in the pocket,
        # in its own frame. Steering at this rather than at the ball is what
        # pulls the ball off-centre into the pocket, since we cannot sidestep.
        err_x, err_y = rel_x - POCKET_X, rel_y - want_y
        aim = float(np.arctan2(err_y, err_x))

        def drive():
            """Turn on the spot if badly aimed, otherwise walk in."""
            turn = float(np.clip(TURN_GAIN * aim, -1.5, 1.5))
            with quiet:
                policy.set_vel_cmd(0.0 if abs(aim) > AIM_TOL else CRUISE, 0.0, turn)

        if phase == "approach":
            near = (APPROACH_STOP_X[0] < rel_x < APPROACH_STOP_X[1]
                    and abs(rel_y - want_y) < APPROACH_STOP_Y)
            if near:
                phase = "settle"
                settle_until = t + SETTLE_S
                with quiet:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
                print(f"t={t:5.1f}s  close enough (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm) -- stopping to settle")
            else:
                drive()

        elif phase == "settle" and t >= settle_until:
            in_pocket = (POCKET_TOL_X[0] < rel_x < POCKET_TOL_X[1]
                         and abs(rel_y - want_y) < POCKET_TOL_Y)
            if in_pocket:
                # trigger_behavior teleports the ball into the pocket; the whole
                # point here is that the duck walked to it, so put it back.
                keep_q = data.qpos[ball_q:ball_q + 7].copy()
                keep_v = data.qvel[ball_v:ball_v + 6].copy()
                with quiet:
                    policy.trigger_behavior(foot)
                data.qpos[ball_q:ball_q + 7] = keep_q
                data.qvel[ball_v:ball_v + 6] = keep_v
                ball_at_kick = ball_xy.copy()
                kicked_at = t
                phase = "kick"
                print(f"t={t:5.1f}s  settled in the pocket (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm, wanted {want_y*1000:+.0f}) -> {foot}")
            else:
                # Settled out of the pocket. Walking straight at it from here
                # just shins the ball away, so back off and come in again.
                phase = "backoff"
                backoff_until = t + BACKOFF_S
                print(f"t={t:5.1f}s  settled off-pocket (ahead {rel_x*1000:.0f} mm, "
                      f"side {rel_y*1000:+.0f} mm) -- backing off to retry")

        elif phase == "backoff":
            if t >= backoff_until:
                phase = "approach"
            else:
                with quiet:
                    policy.set_vel_cmd(-CRUISE, 0.0, 0.0)

        elif phase == "kick" and policy.behavior_mode is None:
            phase = "done"
            with quiet:
                policy.set_vel_cmd(0.0, 0.0, 0.0)

        policy.update_behavior(CONTROL_DT)
        policy.apply_action(policy.infer())
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        if rec:
            rec.maybe_capture(i, data)

        if False and i % 20 == 0:
            bxy = data.qpos[ball_q:ball_q + 2]
            print(f"    t={t:5.1f} phase={phase} mode={policy.behavior_mode} "
                  f"pol={policy.current_policy} duck=({data.qpos[adr]:+.2f},{data.qpos[adr+1]:+.2f}) "
                  f"ball=({bxy[0]:+.3f},{bxy[1]:+.3f}) rel=({rel_x*1000:+.0f},{rel_y*1000:+.0f})mm")

        if i % 100 == 0 and phase == "approach":
            print(f"t={t:5.1f}s  ball {np.hypot(rel_x, rel_y):.2f} m away "
                  f"(ahead {rel_x*1000:+.0f} mm, side {rel_y*1000:+.0f} mm), "
                  f"aim {np.degrees(aim):+6.1f} deg")

        if phase == "done" and t - kicked_at > 3.0:
            break

    ball_end = data.qpos[ball_q:ball_q + 2].copy()
    print()
    if kicked_at is None:
        print(f"never reached the ball in {args.seconds:.0f} s "
              f"(still {np.hypot(rel_x, rel_y):.2f} m away)")
    else:
        moved = np.linalg.norm(ball_end - ball_at_kick)
        print(f"kicked at t={kicked_at:.1f}s")
        print(f"  ball moved:   {moved:.2f} m "
              f"({ball_at_kick[0]:+.2f},{ball_at_kick[1]:+.2f}) -> "
              f"({ball_end[0]:+.2f},{ball_end[1]:+.2f})")
        print(f"  task:         {'SUCCESS' if moved > 0.1 else 'ball barely moved'}")

    if rec:
        print(f"  video:        {args.video} ({rec.write(args.video)} frames)")


if __name__ == "__main__":
    main()
