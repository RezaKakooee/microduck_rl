"""Scripted two-foot glide on ice.

Both feet down in the normal stance, sliding forward. A two-foot stance is
statically stable (the CoM sits between the feet), so unlike the one-leg
spiral this is mainly a question of how far the duck slides before friction
stops it. Kinetic friction decelerates a sliding body at a = mu * g regardless
of mass, so the slide distance and time from speed v are:

    d = v^2 / (2 mu g)        t = v / (mu g)

    mu = 0.02, v = 0.8 m/s  ->  d = 1.63 m,  t = 4.1 s
    mu = 0.05, v = 0.8 m/s  ->  d = 0.65 m,  t = 1.6 s

The controller holds the trunk level with a PD on both ankles (pitch) and both
hip_rolls (roll), from projected gravity. Feed-forward: a slight forward lean of
the ankles so the slide does not tip the duck backward.

    uv run python src/microduck_lab/tasks/skating/ice_experts/two_leg_glide.py --mu 0.02 --speed 0.8
    MUJOCO_GL=egl ... --video videos/ice_experts/two_leg_glide.mp4 --azimuth 270   (needs a GPU)
"""
import argparse

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.tasks.skating.ice_experts.harness import (  # noqa: E402
    make, state, HOME, CONTROL_DT, DECIMATION,
    L_HIP_ROLL, R_HIP_ROLL, L_ANKLE, R_ANKLE, L_HIP_PITCH, L_KNEE, HEAD_YAW, HEAD_ROLL,
    TRUNK_MIN_Z,
)


def glide(mu=0.02, speed=0.8, seconds=8.0, kp_pitch=3.0, kd_pitch=0.2,
          kp_roll=3.0, kd_roll=0.2, lean=0.0, video=None, azimuth=270.0,
          verbose=False, head_yaw=0.0, head_roll=0.0, lift=0.0, knee=0.0,
          trick_at=1.0, ramp=0.5):
    """`head_yaw`/`head_roll`: head target (rad) reached by `trick_at + ramp`.
    `lift`: how much the LEFT hip_pitch moves toward extension (rad) — lifts
    the left foot. Both ramp in together, to test whether the head's mass can
    make up for the lifted leg."""
    model, data, policy, adr, target = make(ice_mu=mu, speed=speed, pose=HOME)
    rec = duck_sim.Recorder(model, distance=0.7, azimuth=azimuth, elevation=-8.0) if video else None
    x0 = float(data.qpos[adr])
    prev = {"roll": 0.0, "pitch": 0.0}
    upright_until = 0.0
    fallen = False
    off_floor_steps = 0
    for i in range(int(seconds / CONTROL_DT)):
        t = i * CONTROL_DT
        s = state(model, data, policy, adr, target, t, "left")
        d_roll = (s["roll"] - prev["roll"]) / CONTROL_DT
        d_pitch = (s["pitch"] - prev["pitch"]) / CONTROL_DT
        prev["roll"], prev["pitch"] = s["roll"], s["pitch"]

        c = target.copy()
        # the trick: ramp the head and the left leg together
        f = float(np.clip((t - trick_at) / ramp, 0.0, 1.0)) if ramp > 0 else float(t >= trick_at)
        c[HEAD_YAW] += f * head_yaw
        c[HEAD_ROLL] += f * head_roll
        c[L_HIP_PITCH] += f * lift          # + = toward extension = foot up and back
        c[L_KNEE] += f * knee               # bend the knee too, so the foot really clears
        # pitch: both ankles, same sign (both feet forward); roll: hip_rolls mirror.
        c[L_ANKLE] += kp_pitch * s["pitch"] + kd_pitch * d_pitch + lean
        c[R_ANKLE] -= kp_pitch * s["pitch"] + kd_pitch * d_pitch + lean
        c[L_HIP_ROLL] += kp_roll * s["roll"] + kd_roll * d_roll
        c[R_HIP_ROLL] += kp_roll * s["roll"] + kd_roll * d_roll
        data.ctrl[:14] = c
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        if rec:
            rec.maybe_capture(i, data)
        # is the LEFT foot really off the floor? count its floor contacts
        lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        l_contacts = sum(1 for k in range(data.ncon)
                         if {data.contact[k].geom1, data.contact[k].geom2} == {lf, floor})
        if t >= trick_at + ramp:
            off_floor_steps = off_floor_steps + 1 if l_contacts == 0 else off_floor_steps
        if not fallen:
            if float(s["trunk_pos"][2]) < TRUNK_MIN_Z or abs(s["roll"]) > 0.6 or abs(s["pitch"]) > 0.6:
                fallen = True
            else:
                upright_until = t
        if verbose and i % 25 == 0:
            e = s["com_rel_support"]
            print(f"  t={t:4.1f} vx={s['trunk_vel'][0]:.2f} roll={np.degrees(s['roll']):+5.1f} "
                  f"pitch={np.degrees(s['pitch']):+5.1f} Lfoot={1000*s['free_foot'][2]:4.0f}mm "
                  f"Lcontacts={l_contacts} CoM-Rfoot y={1000*e[1]:+5.1f}mm x={float(data.qpos[adr])-x0:.2f}")
    dist = float(data.qpos[adr]) - x0
    if rec:
        rec.write(video)
    if verbose and lift:
        print(f"  left foot clear of the floor for {off_floor_steps*CONTROL_DT:.2f}s after the lift")
    return upright_until, dist, fallen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.02)
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--lean", type=float, default=0.0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--azimuth", type=float, default=270.0)
    p.add_argument("--head-yaw", type=float, default=0.0, help="rad, + = left")
    p.add_argument("--head-roll", type=float, default=0.0)
    p.add_argument("--lift", type=float, default=0.0, help="left hip_pitch toward extension, rad")
    p.add_argument("--knee", type=float, default=0.0, help="left knee bend, rad")
    p.add_argument("--trick-at", type=float, default=1.0)
    a = p.parse_args()
    up, dist, fell = glide(a.mu, a.speed, a.seconds, lean=a.lean, video=a.video,
                           azimuth=a.azimuth, verbose=True, head_yaw=a.head_yaw,
                           head_roll=a.head_roll, lift=a.lift, knee=a.knee, trick_at=a.trick_at)
    print(f"upright for {up:.2f}s of {a.seconds:.0f}s | slid {dist:.2f} m | "
          f"{'FELL' if fell else 'stayed up'}")


if __name__ == "__main__":
    main()
