#!/usr/bin/env python3
"""Egg on the head: walk with an egg balanced on the crown of the head.

The pretrained walking policy does the walking. This script does the head:
every control tick it reads the trunk's roll and pitch (projected gravity,
harness style) and the head pad's own roll and pitch (from the pad geom's
rotation), and writes a head-pose COMMAND into the policy's command block
(`policy.head_offset`, obs cmd[3:7] = [neck_pitch, head_pitch, head_yaw,
head_roll]). The policy tracks that command with its own neck servos, so the
head counter-rotates against the trunk and the pad stays level.

Head command for each axis (pitch and roll are independent):

    tilt_trunk = roll or pitch of the trunk, from projected gravity
    tilt_head  = roll or pitch of the pad, same formula on the pad frame
    I         += tilt_head * dt                       (clipped)
    cmd        = trim - kp * tilt_trunk - kh * tilt_head - ki * I
    cmd        = clip(cmd, -cap, cap)

With --no-stabilise the command is just the constant `trim`, which is what
levels the head when the duck stands still (at zero head command the policy
holds the head 11 deg nose-down).

What was measured (docs/tasks/egg_on_head.md has the numbers):
- Standing: the egg's 25 g pulls the head 2.5 deg nose-down and the egg rolls
  off in 0.4 s with the trim alone; the ki term removes the sag and the egg
  stays (pad tilt 0.1 deg rms). So the loop needs the pad's OWN tilt (kh, ki);
  the trunk term kp sees nothing, the trunk barely pitches.
- Walking: the gait itself swings the head, +-5 deg roll and 4.5 m/s2 rms
  (10 m/s2 peak) sideways at ~2 Hz. The loop zeroes the mean tilt but cannot
  follow the swing, and no head-pose or body-pose command reduces the
  shaking. The egg is thrown off within about a second at every speed.

The egg is an ellipsoid on a free joint (scene_egg.xml). It starts parked on
the floor, the duck stands for SETTLE_S, then the egg is set on the pad lying
on its side, rests EGG_SETTLE_S, and only then does the walk command start.
"Time until the egg falls" counts from the walk command. A fall is the egg's
centre dropping FALL_DROP below its rest height or the egg touching the floor.

    uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0.3
    uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0.3 --no-stabilise
    uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0 --seconds 10          # standing
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \\
        src/microduck_lab/tasks/objects/egg_on_head.py --speed 0.3 --video videos/egg_on_head/egg_walk_stab.mp4 \\
        --azimuth 270 --cam-distance 0.6

--video needs a GPU (EGL); the task itself runs anywhere.
"""

import argparse
import contextlib
import io
import math
import os

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION  # noqa: E402

ROBOT_DIR = os.path.join(duck_sim.REPO, "src", "mjlab_microduck", "robot", "microduck")
SCENE = os.path.join(ROBOT_DIR, "scene_egg.xml")
WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")

# Duck stands (head command active) before the egg is placed.
SETTLE_S = 1.5
# Egg rests on the standing duck before the walk command.
EGG_SETTLE_S = 1.0
# Keep simulating this long after a fall so a video shows it.
AFTER_FALL_S = 1.0
# Egg centre this far below its rest height = fallen.
FALL_DROP = 0.015
# Gap the egg is dropped from onto the pad.
EGG_GAP = 0.0005
# Egg semi-axes, as in scene_egg.xml: lying on its side the centre is 10 mm up.
EGG_SIDE = 0.010
# Training caps on the head command slots (see infer_policy.py): head_pitch
# +-1.1, head_roll +-0.31. The policy was never asked for more.
CAP_PITCH = 1.1
CAP_ROLL = 0.31
# Head command that levels the head on a standing duck (measured: at zero
# command the head sits +11 deg nose-down; -0.3 gives -0.5 deg).
TRIM_PITCH = -0.29
TRIM_ROLL = 0.0


def roll_pitch(R):
    """Harness-style roll and pitch of a frame whose level orientation is the identity.

    g = R^T [0,0,-1]; roll = atan2(g_y, -g_z); pitch = atan2(g_x, -g_z).
    pitch > 0 is nose down, roll > 0 is left side down.
    """
    g = R.T @ np.array([0.0, 0.0, -1.0])
    return math.atan2(g[1], -g[2]), math.atan2(g[0], -g[2])


def _rotmat_from_axis(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), angle)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class HeadLeveller:
    """Turns trunk and head tilt into the 4-slot head-pose command."""

    def __init__(self, kp_pitch=0.0, kp_roll=0.0, kh_pitch=0.0, kh_roll=0.0,
                 ki_pitch=0.0, ki_roll=0.0, trim_pitch=TRIM_PITCH, trim_roll=TRIM_ROLL,
                 tau=0.0, enabled=True):
        self.kp = np.array([kp_pitch, kp_roll])
        self.kh = np.array([kh_pitch, kh_roll])
        self.ki = np.array([ki_pitch, ki_roll])
        self.trim = np.array([trim_pitch, trim_roll])
        self.cap = np.array([CAP_PITCH, CAP_ROLL])
        self.tau = tau
        self.enabled = enabled
        self.integral = np.zeros(2)
        self.filtered = None
        self.cmd = self.trim.copy()

    def update(self, trunk_pr, head_pr, dt):
        """trunk_pr / head_pr = (pitch, roll) in rad. Returns (pitch_cmd, roll_cmd)."""
        if not self.enabled:
            self.cmd = self.trim.copy()
            return self.cmd
        trunk = np.array(trunk_pr)
        head = np.array(head_pr)
        if self.tau > 0:
            a = dt / (self.tau + dt)
            self.filtered = trunk if self.filtered is None else self.filtered + a * (trunk - self.filtered)
            trunk = self.filtered
        self.integral = np.clip(self.integral + head * dt, -self.cap, self.cap)
        cmd = self.trim - self.kp * trunk - self.kh * head - self.ki * self.integral
        self.cmd = np.clip(cmd, -self.cap, self.cap)
        return self.cmd

    def head_offset(self):
        # [neck_pitch, head_pitch, head_yaw, head_roll]
        return np.array([0.0, self.cmd[0], 0.0, self.cmd[1]], dtype=np.float32)


def run(speed=0.3, yaw=0.0, seconds=10.0, stabilise=True, kp_pitch=1.0, kp_roll=1.0,
        kh_pitch=2.0, kh_roll=2.0, ki_pitch=5.0, ki_roll=5.0, trim_pitch=TRIM_PITCH,
        trim_roll=TRIM_ROLL, tau=0.0, egg_axis="x", pad=True, rolling=None, place_at=None, walking=WALKING_ONNX,
        video=None, fps=30, azimuth=270.0, cam_distance=0.6, elevation=-10.0,
        track="head", log_every=1.0, verbose=True):
    say = print if verbose else (lambda *a, **k: None)

    with contextlib.redirect_stdout(io.StringIO()):
        model, data = duck_sim.load_scene(SCENE)
        policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=walking)
        policy.set_vel_cmd(0.0, 0.0, 0.0)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    pad_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_pad")
    egg_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "egg_geom")
    egg_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "egg")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    egg_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "egg_free")
    egg_q = int(model.jnt_qposadr[egg_joint])
    egg_v = int(model.jnt_dofadr[egg_joint])
    if not pad:
        # Bare shell: the pad stops colliding and disappears; the egg is still
        # placed at the pad's spot, which is the flattest part of the crown.
        model.geom_contype[pad_geom] = 0
        model.geom_conaffinity[pad_geom] = 0
        model.geom_rgba[pad_geom, 3] = 0.0
    if rolling is not None:
        # MuJoCo takes the max of the two geoms' coefficients, so set both.
        model.geom_friction[egg_geom, 2] = rolling
        model.geom_friction[pad_geom, 2] = rolling
    rolling_used = float(max(model.geom_friction[egg_geom, 2], model.geom_friction[pad_geom, 2]))

    mujoco.mj_forward(model, data)
    # The pad frame in the home pose is level: this is the reference "level".
    P0 = data.geom_xmat[pad_geom].reshape(3, 3).copy()
    pad_top0 = data.geom_xpos[pad_geom] + P0[:, 2] * model.geom_size[pad_geom, 1]
    say(f"pad top (home pose):   x={pad_top0[0]:+.4f} y={pad_top0[1]:+.4f} z={pad_top0[2]:.4f} m"
        f"  ({'pad on' if pad else 'pad OFF, bare shell'}, rolling friction {rolling_used:g})")

    egg_parked = data.qpos[egg_q:egg_q + 7].copy()

    def pad_frame():
        R = data.geom_xmat[pad_geom].reshape(3, 3)
        top = data.geom_xpos[pad_geom] + R[:, 2] * model.geom_size[pad_geom, 1]
        return R, top

    def place_egg():
        R, top = pad_frame()
        if egg_axis == "x":
            Q = _rotmat_from_axis([0, 1, 0], math.pi / 2)    # egg long axis -> pad x (forward)
        elif egg_axis == "y":
            Q = _rotmat_from_axis([1, 0, 0], -math.pi / 2)   # egg long axis -> pad y (lateral)
        else:
            Q = np.eye(3)                                     # standing on its end
        R_egg = R @ Q
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_egg.reshape(-1))
        up = EGG_SIDE if egg_axis in ("x", "y") else 0.014
        data.qpos[egg_q:egg_q + 3] = top + R[:, 2] * (up + EGG_GAP)
        data.qpos[egg_q + 3:egg_q + 7] = q
        data.qvel[egg_v:egg_v + 6] = 0.0

    def egg_on_floor():
        for c in data.contact[:data.ncon]:
            if {c.geom1, c.geom2} == {egg_geom, floor}:
                return True
        return False

    ctl = HeadLeveller(kp_pitch, kp_roll, kh_pitch, kh_roll, ki_pitch, ki_roll,
                       trim_pitch, trim_roll, tau, enabled=stabilise)

    rec = None
    if video:
        rec = duck_sim.Recorder(model, fps=fps, distance=cam_distance, azimuth=azimuth,
                                elevation=elevation)
        if track == "head":
            rec.cam.trackbodyid = head

    # Normal protocol: stand, place the egg, let it settle, then walk; the egg
    # clock starts at the walk command. With place_at: walk first, set the egg
    # on the head place_at seconds into the walk; the clock starts at placement.
    if place_at is None:
        t_place = SETTLE_S
        t_walk = SETTLE_S + EGG_SETTLE_S
        t_clock = t_walk
    else:
        t_walk = SETTLE_S
        t_place = t_walk + place_at
        t_clock = t_place
    t_end = t_clock + seconds
    clock_on = False
    egg_placed = False
    walking = False
    rest_z = None
    fall_t = None
    fall_reason = None
    stop_at = None
    walk_start_xy = None
    walk_start_yaw = None
    trace = []          # (t since walk cmd, egg rel x, rel y, rel z, head tilt deg)
    max_tilt = 0.0
    tilt_sq = 0.0
    tilt_n = 0
    max_trunk = np.zeros(2)
    cmd_log = []
    next_log = 0.0
    walked = 0.0
    yawed = 0.0
    step = 0
    t = 0.0
    while True:
        t = step * CONTROL_DT
        if stop_at is not None and t >= stop_at:
            break
        if t >= t_end and fall_t is None:
            break

        if not egg_placed and t >= t_place:
            place_egg()
            egg_placed = True
        if not walking and t >= t_walk:
            with contextlib.redirect_stdout(io.StringIO()):
                policy.set_vel_cmd(speed, 0.0, yaw)
            walking = True
        if not clock_on and t >= t_clock and egg_placed:
            clock_on = True
            walk_start_xy = data.xpos[trunk][:2].copy()
            walk_start_yaw = duck_sim.trunk_yaw(data, adr)
            R, top = pad_frame()
            rest_z = float(data.xpos[egg_body][2])
            rel0 = R.T @ (data.xpos[egg_body] - top)
            say(f"egg rest height:       {rest_z:.4f} m (centre), {rest_z - top[2]:+.4f} m above the pad top, "
                f"rel xy=({rel0[0] * 1000:+.1f},{rel0[1] * 1000:+.1f}) mm "
                + (f"after {EGG_SETTLE_S:.1f} s standing" if place_at is None else f"placed {place_at:.1f} s into the walk"))

        # --- head command ---
        g = policy.get_projected_gravity()
        trunk_roll = math.atan2(g[1], -g[2])
        trunk_pitch = math.atan2(g[0], -g[2])
        R_pad, top = pad_frame()
        head_roll, head_pitch = roll_pitch(R_pad @ P0.T)
        ctl.update((trunk_pitch, trunk_roll), (head_pitch, head_roll), CONTROL_DT)
        policy.head_offset[:] = ctl.head_offset()
        policy._update_command()

        action = policy.infer()
        policy.apply_action(action)
        if not egg_placed:
            data.qpos[egg_q:egg_q + 7] = egg_parked
            data.qvel[egg_v:egg_v + 6] = 0.0
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        # --- measurements ---
        if egg_placed:
            R_pad, top = pad_frame()
            rel = R_pad.T @ (data.xpos[egg_body] - top)
            tilt = math.degrees(math.acos(float(np.clip(R_pad[2, 2], -1, 1))))
            if clock_on and fall_t is None:
                max_tilt = max(max_tilt, tilt)
                tilt_sq += tilt * tilt
                tilt_n += 1
                max_trunk = np.maximum(max_trunk, np.abs(np.degrees([trunk_pitch, trunk_roll])))
                cmd_log.append(ctl.cmd.copy())
                walked = float(np.linalg.norm(data.xpos[trunk][:2] - walk_start_xy))
                yawed = duck_sim.trunk_yaw(data, adr) - walk_start_yaw
                yawed = math.atan2(math.sin(yawed), math.cos(yawed))
            ts = t - t_clock
            if clock_on and fall_t is None and ts >= next_log:
                trace.append((ts, rel[0], rel[1], rel[2], tilt))
                say(f"  t={ts:5.2f}s  egg rel xy=({rel[0] * 1000:+6.1f},{rel[1] * 1000:+6.1f}) mm  "
                    f"z={data.xpos[egg_body][2]:.4f}  pad tilt={tilt:4.1f} deg  "
                    f"trunk p/r=({math.degrees(trunk_pitch):+4.1f},{math.degrees(trunk_roll):+4.1f})  "
                    f"cmd p/r=({ctl.cmd[0]:+.2f},{ctl.cmd[1]:+.2f})  walked {walked:.2f} m")
                next_log += log_every
            if clock_on and fall_t is None:
                dropped = (data.xpos[egg_body][2] < rest_z - FALL_DROP) or (rel[2] < EGG_SIDE - FALL_DROP)
                on_floor = egg_on_floor()
                if dropped or on_floor:
                    fall_t = ts
                    fall_reason = "on the floor" if on_floor else f"dropped {FALL_DROP * 1000:.0f} mm"
                    stop_at = t + AFTER_FALL_S
                    say(f"  EGG FELL at t={ts:.2f} s ({fall_reason}); "
                        f"rel xy=({rel[0] * 1000:+.1f},{rel[1] * 1000:+.1f}) mm")
            elif fall_t is None and place_at is None and t >= t_place + 0.3:
                # Egg must survive the standing settle too.
                if data.xpos[egg_body][2] < top[2] - FALL_DROP or egg_on_floor():
                    fall_t = -(t_clock - t)
                    fall_reason = "fell while standing, before the walk command"
                    stop_at = t + AFTER_FALL_S
                    say(f"  EGG FELL while standing, {t - t_place:.2f} s after placement")

        if rec is not None:
            rec.maybe_capture(step, data)
        step += 1
        if duck_sim and data.xpos[trunk][2] < duck_sim.FALL_HEIGHT and stop_at is None:
            say(f"  DUCK FELL at t={t - t_clock:.2f} s")
            stop_at = t + AFTER_FALL_S
            if fall_t is None:
                fall_t = t - t_clock
                fall_reason = "duck fell"

    cmd_log = np.array(cmd_log) if cmd_log else np.zeros((1, 2))
    rms_tilt = math.sqrt(tilt_sq / tilt_n) if tilt_n else 0.0
    result = dict(speed=speed, yaw=yaw, stabilise=stabilise, pad=pad, egg_axis=egg_axis, rolling=rolling_used, place_at=place_at,
                  rest_z=rest_z, fall_t=fall_t, fall_reason=fall_reason, walked=walked,
                  yawed=yawed, max_tilt=max_tilt, rms_tilt=rms_tilt,
                  max_trunk_pitch=float(max_trunk[0]), max_trunk_roll=float(max_trunk[1]),
                  cmd_pitch_range=(float(cmd_log[:, 0].min()), float(cmd_log[:, 0].max())),
                  cmd_roll_range=(float(cmd_log[:, 1].min()), float(cmd_log[:, 1].max())),
                  trace=trace, seconds=seconds)
    if verbose:
        mode = f"{'stabilised' if stabilise else 'no stabilisation'}, {'pad' if pad else 'bare shell'}, egg axis {egg_axis}"
        say()
        say(f"command:               vx={speed} wz={yaw}  ({mode})")
        say(f"egg rest height:       {rest_z:.4f} m" if rest_z is not None else "egg rest height:       (egg fell before the walk)")
        clock = "after the walk command" if place_at is None else f"after placement ({place_at:.1f} s into the walk)"
        if fall_t is None:
            say(f"time until egg falls:  stayed ({seconds:.1f} s {clock}, "
                f"{seconds + (EGG_SETTLE_S if place_at is None else 0):.1f} s on the head)")
        else:
            say(f"time until egg falls:  {fall_t:.2f} s {clock} ({fall_reason})")
        say(f"distance walked:       {walked:.3f} m with the egg on"
            + (f", turned {math.degrees(yawed):+.0f} deg" if yaw else ""))
        say(f"max head (pad) tilt:   {max_tilt:.1f} deg  (rms {rms_tilt:.1f} deg); "
            f"max trunk pitch {max_trunk[0]:.1f} deg, roll {max_trunk[1]:.1f} deg")
        say(f"head cmd range:        pitch [{cmd_log[:, 0].min():+.2f},{cmd_log[:, 0].max():+.2f}]  "
            f"roll [{cmd_log[:, 1].min():+.2f},{cmd_log[:, 1].max():+.2f}] rad")
        if trace:
            xs = np.array([[a[1], a[2]] for a in trace]) * 1000
            say(f"egg drift on the pad:  x {xs[0, 0]:+.1f} -> {xs[-1, 0]:+.1f} mm, y {xs[0, 1]:+.1f} -> {xs[-1, 1]:+.1f} mm "
                f"(max |x| {np.abs(xs[:, 0]).max():.1f}, max |y| {np.abs(xs[:, 1]).max():.1f} mm, 1 s samples)")
    if rec is not None:
        n = rec.write(video)
        say(f"video:                 {video} ({n} frames)")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=float, default=0.3, help="forward command, m/s (0 = stand)")
    p.add_argument("--yaw", type=float, default=0.0, help="yaw-rate command, rad/s")
    p.add_argument("--seconds", type=float, default=10.0, help="run time after the walk command, s")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--stabilise", dest="stabilise", action="store_true", default=True)
    g.add_argument("--no-stabilise", dest="stabilise", action="store_false",
                   help="constant trim head command only")
    p.add_argument("--kp-pitch", type=float, default=1.0, help="head_pitch cmd per rad of trunk pitch")
    p.add_argument("--kp-roll", type=float, default=1.0, help="head_roll cmd per rad of trunk roll")
    p.add_argument("--kh-pitch", type=float, default=2.0, help="head_pitch cmd per rad of measured pad pitch")
    p.add_argument("--kh-roll", type=float, default=2.0, help="head_roll cmd per rad of measured pad roll")
    p.add_argument("--ki-pitch", type=float, default=5.0, help="integral gain on measured pad pitch, 1/s")
    p.add_argument("--ki-roll", type=float, default=5.0, help="integral gain on measured pad roll, 1/s")
    p.add_argument("--trim-pitch", type=float, default=TRIM_PITCH, help="constant head_pitch command")
    p.add_argument("--trim-roll", type=float, default=TRIM_ROLL, help="constant head_roll command")
    p.add_argument("--tau", type=float, default=0.0, help="low-pass time constant on trunk tilt, s (0 = off)")
    p.add_argument("--egg-axis", choices=["x", "y", "z"], default="x",
                   help="egg long axis: x = along the walk (rolls sideways), y = across (rolls fore/aft), z = on end")
    p.add_argument("--no-pad", action="store_true", help="switch the head pad off: egg on the bare shell")
    p.add_argument("--place-at", type=float, default=None,
                   help="set the egg on the head this many seconds INTO the walk instead of before it")
    p.add_argument("--rolling", type=float, default=None,
                   help="override the egg/pad rolling friction (scene default 0.003; 0.0002 = free rolling)")
    p.add_argument("--walking", default=WALKING_ONNX)
    p.add_argument("--video", default=None, help="write an mp4 here (needs MUJOCO_GL=egl and a GPU)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--azimuth", type=float, default=270.0, help="camera azimuth; 270 = side view")
    p.add_argument("--cam-distance", type=float, default=0.6)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--track", choices=["head", "trunk"], default="head", help="body the camera follows")
    p.add_argument("--log-every", type=float, default=1.0, help="print the egg position this often, s")
    args = p.parse_args()

    run(speed=args.speed, yaw=args.yaw, seconds=args.seconds, stabilise=args.stabilise,
        kp_pitch=args.kp_pitch, kp_roll=args.kp_roll, kh_pitch=args.kh_pitch, kh_roll=args.kh_roll,
        ki_pitch=args.ki_pitch, ki_roll=args.ki_roll, trim_pitch=args.trim_pitch,
        trim_roll=args.trim_roll, tau=args.tau, egg_axis=args.egg_axis, pad=not args.no_pad, rolling=args.rolling, place_at=args.place_at,
        walking=args.walking, video=args.video, fps=args.fps, azimuth=args.azimuth,
        cam_distance=args.cam_distance, elevation=args.elevation, track=args.track,
        log_every=args.log_every)


if __name__ == "__main__":
    main()
