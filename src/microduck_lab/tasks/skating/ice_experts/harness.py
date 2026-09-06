"""Shared harness for scripted one-legged ice-glide controllers.

Every strategy is scored by THIS file with the SAME criteria, so results are
comparable and cannot be gamed by changing the test.

Task: normal feet, ice (mu on floor AND feet), start sliding at `speed` m/s
along +x, one leg extended behind (the arabesque / spiral pose), hold as long
as possible. Score = seconds until the pose is lost.

The pose counts as HELD while all of these are true:
  - free foot at least 15 mm above the floor  (no touching down)
  - trunk higher than 70 mm                    (not fallen)
  - free-leg hip_pitch extended past 0.6 rad   (not tucked; 1.1 is the target)

Usage from a strategy file:

    from microduck_lab.tasks.skating.ice_experts.harness import evaluate, BALANCED_POSE
    def my_controller(s):      # s is a dict, see `state()` below
        ctrl = s["target"].copy()
        ...
        return ctrl            # 14 joint position targets
    print(evaluate(my_controller))

Joint order (14): 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle),
5-8 neck_pitch, head_pitch, head_yaw, head_roll, 9-13 right leg.
"""

import contextlib
import io
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

import mujoco  # noqa: E402
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION  # noqa: E402

HIP_EXTENSION = 1.1
HIP_ROLL_LIMIT = 0.384
FREE_FOOT_MIN_Z = 0.015
TRUNK_MIN_Z = 0.07
MIN_EXTENSION = 0.6

L_HIP_YAW, L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4
NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = 5, 6, 7, 8
R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 9, 10, 11, 12, 13

HOME = np.array([0, -0.0873, -0.4579, -0.0049, 0.453,
                 0.3491, 0.3491, 0, 0,
                 0, 0.0873, 0.4579, 0.0049, -0.453], dtype=np.float64)


def balanced_pose(free="left", hip_roll_shift=HIP_ROLL_LIMIT):
    """The statically balanced arabesque. Measured: with both hip_rolls shifted
    to the support side the CoM sits inside the support foot (+2 mm against a
    foot spanning -34..+6 mm). Without the shift it is 42.8 mm outside."""
    t = HOME.copy()
    if free == "left":
        t[L_HIP_PITCH], t[L_KNEE] = HIP_EXTENSION, 0.0
        t[L_HIP_ROLL] = hip_roll_shift
        t[R_HIP_ROLL] = hip_roll_shift
    else:
        t[R_HIP_PITCH], t[R_KNEE] = -HIP_EXTENSION, 0.0
        t[L_HIP_ROLL] = -hip_roll_shift
        t[R_HIP_ROLL] = -hip_roll_shift
    return t


BALANCED_POSE = balanced_pose("left")


def _set_ice(model, mu):
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name == "floor" or (name and re.match(r"^(left|right)_foot_collision$", name)):
            model.geom_friction[g, 0] = mu


def _ground(model, data, adr):
    """Lowest trunk height with no contact: rest the support foot on the floor."""
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


def make(ice_mu=0.1, speed=0.4, free="left", hip_roll_shift=HIP_ROLL_LIMIT, pose=None):
    """Scene, data, policy helper and the trunk qpos address, set up in the pose.
    `pose` overrides the balanced one-leg pose (e.g. HOME for a two-foot stance)."""
    buf = io.StringIO()
    model, data = duck_sim.load_scene("walk")
    _set_ice(model, ice_mu)
    with contextlib.redirect_stdout(buf):
        policy, adr = duck_sim.make_policy(
            model, data,
            walking_onnx_path=os.path.join(duck_sim.REPO, "spiral_v5.onnx"),
        )
    target = balanced_pose(free, hip_roll_shift) if pose is None else np.array(pose, dtype=np.float64)
    for i, qi in enumerate(policy.joint_qpos_indices):
        data.qpos[qi] = target[i]
    _ground(model, data, adr)
    data.qvel[:] = 0.0
    data.qvel[0] = speed
    data.ctrl[:14] = target
    mujoco.mj_forward(model, data)
    return model, data, policy, adr, target


def state(model, data, policy, adr, target, t, free):
    """Everything a controller may reasonably read."""
    quat = data.qpos[adr + 3:adr + 7].astype(np.float32)
    g = policy.quat_rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32))
    q = np.array([data.qpos[i] for i in policy.joint_qpos_indices])
    dq = np.array([data.qvel[i] for i in policy.joint_qvel_indices])
    com = np.sum(model.body_mass[:, None] * data.xipos, axis=0) / model.body_mass.sum()
    sid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    sid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    sup = sid_r if free == "left" else sid_l
    fr = sid_l if free == "left" else sid_r
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    return {
        "t": t,
        "q": q, "dq": dq,
        "target": target.copy(),
        "free": free,
        "roll": float(np.arctan2(g[1], -g[2])),
        "pitch": float(np.arctan2(g[0], -g[2])),
        "gyro": np.array(data.qvel[adr + 3:adr + 6]),           # trunk angular vel (world dofs)
        "trunk_pos": np.array(data.xpos[trunk]),
        "trunk_vel": np.array(data.qvel[adr:adr + 3]),
        "com": com,
        "support_foot": np.array(data.site_xpos[sup]),
        "free_foot": np.array(data.site_xpos[fr]),
        "com_rel_support": com - data.site_xpos[sup],            # the balance error, world frame
        "model": model, "data": data,
    }


def evaluate(controller, seconds=10.0, ice_mu=0.1, speed=0.4, free="left",
             hip_roll_shift=HIP_ROLL_LIMIT, video=None, verbose=False):
    """Run `controller` and return how long the pose was held, in seconds."""
    model, data, policy, adr, target = make(ice_mu, speed, free, hip_roll_shift)
    rec = duck_sim.Recorder(model, distance=0.9) if video else None
    steps = int(round(seconds / CONTROL_DT))
    held = 0.0
    lost = None
    ext_idx = L_HIP_PITCH if free == "left" else R_HIP_PITCH
    ext_sign = 1.0 if free == "left" else -1.0

    for i in range(steps):
        t = i * CONTROL_DT
        s = state(model, data, policy, adr, target, t, free)
        ctrl = np.asarray(controller(s), dtype=np.float64)
        assert ctrl.shape == (14,), f"controller must return 14 targets, got {ctrl.shape}"
        data.ctrl[:14] = ctrl
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        if rec:
            rec.maybe_capture(i, data)

        free_z = float(s["free_foot"][2])
        trunk_z = float(s["trunk_pos"][2])
        extension = ext_sign * float(s["q"][ext_idx])
        if lost is None:
            if free_z < FREE_FOOT_MIN_Z:
                lost = f"free foot touched down at {t:.2f}s"
            elif trunk_z < TRUNK_MIN_Z:
                lost = f"fell at {t:.2f}s"
            elif extension < MIN_EXTENSION:
                lost = f"free leg tucked (hip_pitch {extension:.2f} < {MIN_EXTENSION}) at {t:.2f}s"
            else:
                held = t
            if lost and not rec:
                break
        if verbose and i % 25 == 0:
            e = s["com_rel_support"]
            print(f"  t={t:4.2f} roll={np.degrees(s['roll']):+6.1f} pitch={np.degrees(s['pitch']):+6.1f} "
                  f"com-foot x={1000*e[0]:+5.1f} y={1000*e[1]:+5.1f} mm  vx={s['trunk_vel'][0]:.2f}")

    if rec:
        rec.write(video)
    if verbose:
        print(f"  held {held:.2f}s" + (f"  ({lost})" if lost else "  (full run)"))
    return held


if __name__ == "__main__":
    # Baseline: PD on support hip_roll and ankle, the controller from expert_spiral.py.
    prev = {"roll": 0.0, "pitch": 0.0}

    def pd(s):
        c = s["target"].copy()
        d_roll = (s["roll"] - prev["roll"]) / CONTROL_DT
        d_pitch = (s["pitch"] - prev["pitch"]) / CONTROL_DT
        prev["roll"], prev["pitch"] = s["roll"], s["pitch"]
        c[R_HIP_ROLL] += 0.0 * s["roll"] + 1.5 * d_roll
        c[R_ANKLE] += 4.0 * s["pitch"] + 0.3 * d_pitch
        return c

    print("baseline PD:", evaluate(pd, verbose=True))
