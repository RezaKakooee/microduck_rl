"""Jump world: one Microduck on flat ground, BAM XL330 motors.

Only motor targets move her. There are no external forces and no teleports:
`step` refuses to run if anything writes `xfrc_applied` or `qfrc_applied`.

Setup comes from the shared code, not copied here:
- motors: `load_bam_model` + `load_mujoco_with_bam` from scripts/infer_policy.py
  (the same BAM M6 XL330 model the policies train against);
- timestep and control rate: `duck_sim`;
- stand pose and start height: what `duck_sim.make_policy` sets
  (`DEFAULT_POSE`, trunk at 0.125 m). `make_policy` itself needs an ONNX
  policy, and the jump has none, so its two values are used directly.

The firmware current limit (1.75 A) is ON by default, as on the real servo
and as `duck_sim.load_scene` clamps it. Training runs without it, so it can be
turned off with `max_current=None` to compare.
"""

from __future__ import annotations

import os

import numpy as np
import mujoco

from microduck_lab.sim import duck_sim
import microduck_lab.sim.upstream  # noqa: F401  (puts scripts/ on sys.path)
from microduck_lab.sim.upstream import DEFAULT_POSE
from infer_policy import BAM_KP_FW, BAM_VIN_MIN, load_bam_model, load_mujoco_with_bam

# Leg pitch joints: the ones that lift her. Right leg values are the left
# ones negated (the right joint axes point the other way).
LEFT_LEG = ("left_hip_pitch", "left_knee", "left_ankle")
RIGHT_LEG = ("right_hip_pitch", "right_knee", "right_ankle")
FEET = ("left_foot_collision", "right_foot_collision")
START_Z = 0.125  # duck_sim.make_policy


class JumpWorld:
    def __init__(self, vin=7.4, max_current=1.75, vin_drop_gain=0.1,
                 timestep=duck_sim.CONTROL_TIMESTEP, decimation=duck_sim.DECIMATION,
                 scene="walk"):
        xml = duck_sim.SCENES.get(scene, scene)
        xml = xml if os.path.isabs(xml) else os.path.join(duck_sim.REPO, xml)
        self.bam = load_bam_model(BAM_KP_FW, vin, max_current)
        m, d, self.motor, names = load_mujoco_with_bam(
            xml, self.bam, timestep, vin_drop_gain, BAM_VIN_MIN)
        self.model, self.data, self.names = m, d, names
        self.decimation = decimation
        self.control_dt = decimation * timestep

        self.act = {n: i for i, n in enumerate(self.names)}
        jid = m.actuator_trnid[:, 0]
        self.qidx = m.jnt_qposadr[jid]
        self.vidx = m.jnt_dofadr[jid]
        self.lo, self.hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]

        free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.root = int(m.jnt_qposadr[free])
        self.stand = np.array(DEFAULT_POSE[:len(names)], dtype=float)
        d.qpos[self.root:self.root + 7] = [0, 0, START_Z, 1, 0, 0, 0]
        self.set_pose(self.stand)

        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FEET]
        self.foot_sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
                           for n in ("left_foot", "right_foot")]
        self.mass = float(m.body_subtreemass[self.trunk])
        # Sole mesh vertices (local frame), for the true lowest point of each foot.
        self.sole_verts = []
        for g in self.feet:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            self.sole_verts.append(m.mesh_vert[a:a + n].astype(float))
        self.t = 0.0
        self.log = []
        self.on_substep = None   # called after every physics substep (video)

    # ---- poses -------------------------------------------------------------
    def leg_pose(self, hip, knee, ankle, base=None):
        """A 14-joint target with the given LEFT leg angles, mirrored to the right."""
        q = (self.stand if base is None else base).copy()
        for name, v in zip(LEFT_LEG, (hip, knee, ankle)):
            q[self.act[name]] = v
        for name, v in zip(RIGHT_LEG, (hip, knee, ankle)):
            q[self.act[name]] = -v
        return q

    def crouch(self, depth):
        """Leg pose with the hips `depth` metres lower than in the stand pose.

        Trunk upright, soles flat, each ankle straight under where it is in
        the stand pose. Negative depth = longer legs than standing. Solved on
        a scratch copy of the model, so the running sim is not touched.
        Returns the 14-joint target.
        """
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[self.root:self.root + 7] = [0, 0, START_Z, 1, 0, 0, 0]
        hip_j, knee_j, ankle_j = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_LEG)

        def ankle_rel(h, k):
            d.qpos[self.qidx] = self.leg_pose(h, k, k - h)   # k - h keeps the sole flat
            mujoco.mj_kinematics(m, d)
            r = d.xanchor[ankle_j] - d.xanchor[hip_j]
            return np.array([r[0], r[2]])

        h0, k0, _ = self.left_leg(self.stand)
        goal = ankle_rel(h0, k0) + np.array([0.0, depth])
        x = np.array([h0, k0])
        for _ in range(50):
            f = ankle_rel(*x) - goal
            if np.linalg.norm(f) < 1e-7:
                break
            J = np.column_stack([(ankle_rel(*(x + e)) - ankle_rel(*x)) / 1e-6 for e in np.eye(2) * 1e-6])
            x = x - np.linalg.lstsq(J, f, rcond=None)[0]
        err = np.linalg.norm(ankle_rel(*x) - goal)
        if err > 1e-5:
            raise ValueError(f"no leg pose for depth {depth:.3f} m (error {err*1000:.2f} mm)")
        return self.leg_pose(x[0], x[1], x[1] - x[0])

    def left_leg(self, q):
        return np.array([q[self.act[n]] for n in LEFT_LEG])

    def set_pose(self, q, z=None):
        """Place her in pose q before the run starts (only used at t = 0)."""
        d = self.data
        d.qpos[self.qidx] = q
        d.qvel[:] = 0
        if z is not None:
            d.qpos[self.root + 2] = z
        self.motor.reset(d.qpos)
        self.motor.q_target[:] = q
        mujoco.mj_forward(self.model, d)

    # ---- stepping ----------------------------------------------------------
    def step(self, target):
        """One control step (50 Hz): hold `target`, run the physics substeps."""
        m, d = self.model, self.data
        target = np.clip(np.asarray(target, dtype=float), self.lo, self.hi)
        if not np.isfinite(target).all():
            raise RuntimeError("non-finite target")
        self.motor.q_target[:] = target
        for _ in range(self.decimation):
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
                raise RuntimeError("external force present")
            self.motor.update()
            mujoco.mj_step(m, d)
            self._record()
            if self.on_substep is not None:
                self.on_substep(self)
        if not np.isfinite(d.qpos).all():
            raise RuntimeError("non-finite state")

    def hold(self, target, seconds):
        for _ in range(int(round(seconds / self.control_dt))):
            self.step(target)

    # ---- measures (every physics substep) ---------------------------------
    def touching(self):
        """Robot geoms touching the floor: (feet set, other-geoms set)."""
        d = self.data
        feet, other = set(), set()
        for c in d.contact[:d.ncon]:
            if self.floor not in (c.geom1, c.geom2):
                continue
            g = c.geom2 if c.geom1 == self.floor else c.geom1
            (feet if g in self.feet else other).add(int(g))
        return feet, other

    def sole_heights(self):
        """Lowest point of each sole mesh above the floor (m): (left, right)."""
        d = self.data
        return tuple(float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
                     for g, v in zip(self.feet, self.sole_verts))

    def tilt_deg(self):
        return float(np.degrees(np.arccos(np.clip(self.data.xmat[self.trunk][8], -1, 1))))

    def _record(self):
        d = self.data
        feet, other = self.touching()
        com = d.subtree_com[self.trunk]
        # subtree linear momentum / mass = CoM velocity. mj_subtreeVel fills cvel-based fields.
        mujoco.mj_subtreeVel(self.model, d)
        vz = float(d.subtree_linvel[self.trunk][2])
        self.log.append((
            float(d.time), float(com[2]), vz, len(feet), len(other),
            float(d.xpos[self.trunk][2]), self.tilt_deg(),
            float(min(d.site_xpos[s][2] for s in self.foot_sites)),
            float(np.max(np.abs(d.actuator_force))),
            min(self.sole_heights()),
        ))


LOG_COLUMNS = ("t", "com_z", "com_vz", "n_feet", "n_other", "trunk_z", "tilt", "foot_z_min", "max_torque",
               "sole_z_min")


def flight_report(log, stand_com_z, stand_foot_z, t0=0.0):
    """Measure the jump in a substep log (rows as in LOG_COLUMNS).

    Flight = substeps with no robot geom touching the floor. The longest
    contiguous flight after t0 counts.
    """
    a = np.array(log)
    a = a[a[:, 0] >= t0]
    air = (a[:, 3] == 0) & (a[:, 4] == 0)
    best, start, run_start = 0, None, None
    for i, f in enumerate(air):
        if f and run_start is None:
            run_start = i
        if (not f or i == len(air) - 1) and run_start is not None:
            end = i if not f else i + 1
            if end - run_start > best:
                best, start = end - run_start, run_start
            run_start = None
    dt = a[1, 0] - a[0, 0] if len(a) > 1 else 0.0
    out = {
        "flight_s": best * dt,
        "peak_com_vz": float(a[:, 2].max()),
        "com_rise_mm": float((a[:, 1].max() - stand_com_z) * 1000),
        "foot_lift_mm": 0.0,
        "other_contact": bool(a[:, 4].any()),
        "max_tilt": float(a[:, 6].max()),
        "max_torque": float(a[:, 8].max()),
    }
    if best:
        seg = a[start:start + best]
        out["foot_lift_mm"] = float((seg[:, 7].max() - stand_foot_z) * 1000)
        # true clearance: lowest point of either sole, at its highest in flight
        out["sole_clear_mm"] = float(seg[:, 9].max() * 1000)
        out["takeoff_t"] = float(seg[0, 0])
        out["takeoff_vz"] = float(seg[0, 2])
        # CoM rise during the flight itself (ballistic part), and above standing
        out["flight_rise_mm"] = float((seg[:, 1].max() - seg[0, 1]) * 1000)
        out["flight_com_over_stand_mm"] = float((seg[:, 1].max() - stand_com_z) * 1000)
    return out


def tilt_angles(quat):
    """(roll, pitch) of the trunk, the projected-gravity way (what the IMU gives).

    roll > 0: leans left. pitch > 0: nose down. Same convention as
    tasks/balance_board/board.py.
    """
    neg, g = np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(neg, np.asarray(quat, dtype=np.float64))
    mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), neg)
    return float(np.arctan2(g[1], -g[2])), float(np.arctan2(g[0], -g[2]))


class Upright:
    """Keep the trunk upright with IMU feedback on top of any leg pose.

    The bare servos cannot hold a pose on this robot: the XL330 P loop
    (0.55 Nm/rad) is softer than gravity about the ankles, so she tips over
    in about 1 s. Same structure and gains as the balance-board upright PD:
        pitch -> ankles  (left +u, right -u), u = kp*pitch + kd*d(pitch)/dt
        roll  -> both hip rolls (-v),         v = kp*roll  + kd*d(roll)/dt
    """

    def __init__(self, world, pitch=(5.0, 0.3), roll=(2.0, 0.1)):
        self.w = world
        self.pitch, self.roll = pitch, roll
        self.prev = None

    def __call__(self, pose):
        w = self.w
        r, p = tilt_angles(w.data.qpos[w.root + 3:w.root + 7])
        if self.prev is None:
            self.prev = (r, p)
        dr, dp = (r - self.prev[0]) / w.control_dt, (p - self.prev[1]) / w.control_dt
        self.prev = (r, p)
        u = self.pitch[0] * p + self.pitch[1] * dp
        v = self.roll[0] * r + self.roll[1] * dr
        q = np.array(pose, dtype=float).copy()
        q[w.act["left_ankle"]] += u
        q[w.act["right_ankle"]] -= u
        q[w.act["left_hip_roll"]] -= v
        q[w.act["right_hip_roll"]] -= v
        return q
