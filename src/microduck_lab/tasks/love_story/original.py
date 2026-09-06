#!/usr/bin/env python3
"""Two ducks, the pretrained policies, and a love story: closed loop, like the
kick and pick tasks. Nothing is posed. Every joint angle in the film comes out
of an ONNX policy running in the real simulator on the real servo gains.

    1. He walks up to her                    walking policy, drive-to-target
    2. He bows and offers the ring            ground-pick policy (beak to floor)
    3. She nods yes; the ring goes on her neck   head-pose command; weld swap
    4. They kiss                              standing policy, body + head
                                              commands leaned in until the
                                              beak tips meet
    5. She walks to the nest, sits, and lays two eggs   walking + sitstand
    6. He kisses the top of her head          standing policy, leaned in until
                                              his head touches hers
    7. She backs onto the eggs and broods; they swap; he broods
                                              walking (reverse) + sitstand
    8. The eggs rock, crack, and two ducklings come out

The two ducks are the same MJCF attached twice into one world
(`film/stage.py`), each with its own `PolicyInference` pointed at its own
prefixed sensors and actuators -- the same walking / standing / sitstand /
ground-pick policies `robotd` runs, on the same 50 Hz / decimation-4 cadence,
with the XL330 current limit clamping torque as `duck_sim.load_scene` does.
The ducks see each other only through contact.

The ring is welded to his jaw and later to her neck, the way `task_pick_up.py`
welds the cube: there is no gripper on this robot. The eggs are free bodies
that drop into the nest. The ducklings are the same MJCF at a fifth scale; no
policy exists at that scale, so their servos simply hold a pose and physics
does the rest.

The director is a state machine that reads the world each tick -- where each
duck actually is, whether the beaks actually touch -- and issues commands.
Every phase has a time-out, because the policies drift and cannot take a
small step (see HANDOFF.md).

    uv run python -m microduck_lab.tasks.love_story.original --dry                 # no GL: simulate, log the beats
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \\
        src/microduck_lab/tasks/love_story/original.py --video videos/love_story/duck_love_story.mp4
"""

import argparse
import contextlib
import io
import os

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.film import stage
from microduck_lab import paths
from microduck_lab.sim.duck_sim import CONTROL_DT, CONTROL_TIMESTEP, DECIMATION, POLICY_JOINTS  # noqa: E402
from microduck_lab.sim.upstream import PolicyInference  # noqa: E402

# `paths` already resolves this: the POLICIES env var if set, otherwise a
# `microduck` checkout beside this one.
POLICIES = str(paths.POLICIES)

# Same numbers as task_pick_up.py: the alpha policies turn at ~0.4 rad/s on a
# 1.0 command, walk 0.13 m/s on 0.3, and do nothing below ~0.25.
CRUISE = 0.3
TURN_GAIN = 2.5
AIM_TOL = 0.35
# Below about 1.0 the turn command does nothing (1.0 commands 0.38 rad/s); a
# proportional turn parks the duck in that deadband, pointing almost the right
# way forever. Outside the tolerance the turn is at least this.
TURN_MIN = 1.0


def turn_cmd(err, tol):
    """Bang-bang heading command with a floor the policy actually answers to."""
    if abs(err) <= tol:
        return 0.0
    return float(np.sign(err) * np.clip(TURN_GAIN * abs(err), TURN_MIN, 1.5))
MOUTH_SHUT, MOUTH_OPEN = 0.0, 0.45

quiet = lambda: contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------------
# One duck: a policy, its own sensors and actuators, and a few closed loops
# ---------------------------------------------------------------------------


class Agent:
    """A prefixed duck driven by `PolicyInference`.

    `PolicyInference` looks its sensors up by bare name and sizes itself from
    `model.nu`, which in a four-duck world is 60. So it is built against the
    shared model and then pointed at THIS duck: its trunk, its gyro, its 14
    servo actuators, its 14 joints -- the same trim `duck_sim.make_policy`
    does for the mouth, taken one step further.
    """

    def __init__(self, model, data, prefix, x, y, yaw, has_policy=True):
        self.model, self.data, self.prefix = model, data, prefix
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{n}")
        self.trunk, self.head_body, self.jaw, self.neck = (bid("trunk_base"), bid("jaw_soft"),
                                                      bid("mouth_jaw"), bid("neck_pitch"))
        self.tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_mouth_tip")
        first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_left_hip_yaw")
        self.servo = list(range(first, first + POLICY_JOINTS))
        self.mouth = first + POLICY_JOINTS
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_trunk_base_freejoint")
        self.adr = int(model.jnt_qposadr[jid])
        self.dofadr = int(model.jnt_dofadr[jid])
        self.qpos_idx = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in self.servo]
        self.qvel_idx = [int(model.jnt_dofadr[model.actuator_trnid[a, 0]]) for a in self.servo]
        self.head_geoms = {g for g in range(model.ngeom)
                           if model.geom_bodyid[g] in (self.head_body, self.jaw) and model.geom_contype[g]}
        # Just the jaw: the moving lower beak, and the only part of a duck that
        # can reach another duck's beak. `head_geoms` includes the shells, which
        # is what a beak has to get past.
        self.beak_geoms = {g for g in range(model.ngeom)
                           if model.geom_bodyid[g] == self.jaw and model.geom_contype[g]}
        self.mouth_cmd = MOUTH_SHUT
        self.pose_target = None            # for a duck with no policy (a duckling)
        self.scale = stage.CAST[prefix][2]
        self.clock = 0.0                   # story time, kept by the director
        self.v_filt = np.zeros(2)          # trunk velocity, ~0.5 s low-pass: the gait swings the raw one +-0.3 m/s
        self._best = (None, 0.0)           # (best distance so far, when it improved)
        self._unstick_until = -1.0
        self.say = print

        self.policy = None
        if has_policy:
            with quiet():
                pol = PolicyInference(
                    model, data,
                    walking_onnx_path=f"{POLICIES}/alpha_walking.onnx",
                    standing_onnx_path=f"{POLICIES}/alpha_stand.onnx",
                    sitstand_onnx_path=f"{POLICIES}/alpha_sitstand.onnx",
                    ground_pick_onnx_path=f"{POLICIES}/alpha_ground_pick.onnx",
                    new_cmd_obs=True, use_projected_gravity=True)
            pol.n_joints = POLICY_JOINTS
            pol.joint_qpos_indices = self.qpos_idx
            pol.joint_qvel_indices = self.qvel_idx
            pol.default_pose = pol.default_pose[:POLICY_JOINTS]
            pol.last_action = np.zeros(POLICY_JOINTS, dtype=np.float32)
            pol.trunk_base_id = self.trunk
            pol.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                                   f"{prefix}_imu_ang_vel")
            assert pol.imu_ang_vel_id >= 0 and self.trunk > 0
            pol.vel_max_x, pol.vel_min_x = 0.3, -0.3
            pol.vel_max_y, pol.vel_min_y = 0.2, -0.2
            pol.vel_max_ang = 1.5
            self.policy = pol
            self.default_pose = pol.default_pose
        else:
            from microduck_lab.sim.upstream import DEFAULT_POSE
            self.default_pose = np.array(DEFAULT_POSE[:POLICY_JOINTS], dtype=np.float32)

        # Stand it where the story starts, in the policy's default pose.
        data.qpos[self.adr:self.adr + 3] = (x, y, 0.125 * self.scale)
        data.qpos[self.adr + 3:self.adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        for i, q in enumerate(self.qpos_idx):
            data.qpos[q] = self.default_pose[i]
        data.ctrl[self.servo] = self.default_pose
        data.ctrl[self.mouth] = MOUTH_SHUT

    # -- reading the world --------------------------------------------------

    def xy(self):
        return self.data.qpos[self.adr:self.adr + 2].copy()

    def yaw(self):
        qw, qx, qy, qz = self.data.qpos[self.adr + 3:self.adr + 7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def z(self):
        return float(self.data.qpos[self.adr + 2])

    def up(self):
        return float(self.data.xmat[self.trunk].reshape(3, 3)[2, 2])

    def beak(self):
        return self.data.site_xpos[self.tip].copy()

    def to_local(self, world_xy):
        d = np.asarray(world_xy) - self.xy()
        c, s = np.cos(-self.yaw()), np.sin(-self.yaw())
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    # -- commands ----------------------------------------------------------

    def vel(self, vx=0.0, vy=0.0, wz=0.0):
        with quiet():
            self.policy.set_vel_cmd(vx, vy, wz)

    def stop(self):
        self.vel(0.0, 0.0, 0.0)

    def stalled(self, dist, patience=5.0):
        """No progress towards the target for `patience` seconds.

        The policies stall on things a person would step over -- a foot
        against a body, a 6 mm edge. When it happens, say what the duck is
        touching, then back off for a moment and try again."""
        best, since = self._best
        if best is None or dist < best - 0.01:
            self._best = (dist, self.clock)
            return False
        if self.clock - since > patience:
            self._best = (dist, self.clock)
            names = set()
            m, d = self.model, self.data
            mine = {m.geom_bodyid[g] for g in range(m.ngeom)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "").startswith(self.prefix + "_")}
            for i in range(d.ncon):
                c = d.contact[i]
                for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if m.geom_bodyid[g] in mine:
                        names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)
                                  or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[other]) or "?")
            self.say(f"t={self.clock:6.1f}s  {self.prefix} STALLED {dist*1000:.0f} mm from its mark, "
                     f"touching: {sorted(names)} -- backing off")
            self._unstick_until = self.clock + 1.5
            return True
        return False

    def reset_progress(self):
        self._best = (None, 0.0)

    def drive_to(self, target_xy, stop_r=0.03, cruise=CRUISE, ahead=0.0):
        """Steer at a point the way the pick task does: turn to face it, walk
        when facing it, stop inside the window. Returns True when there.

        The coast after a stop command measured 1-2 cm, so the window is the
        window. (Two things were tried and made every stop worse: predicting
        the coast from the trunk velocity, which the gait swings +-0.3 m/s,
        and calling the stop 4 cm early, which left the duck 8 cm short.)"""
        rel = self.to_local(target_xy)
        rel[0] -= ahead
        dist = float(np.hypot(*rel))
        if dist < stop_r:
            self.stop()
            self.reset_progress()
            return True
        if self.clock < self._unstick_until:
            self.vel(-cruise, 0.0, 0.0)
            return False
        if self.stalled(dist) and dist < 0.12:
            # Hovering a hand's width from the mark and not closing: that is
            # the small step this policy cannot take. Backing off blindly is
            # worse -- from a skewed heading it reversed a duck into the nest
            # and onto an egg. Stop here; it is close enough for every beat.
            self._unstick_until = -1.0
            self.stop()
            self.reset_progress()
            self.say(f"t={self.clock:6.1f}s  {self.prefix} settles {dist*1000:.0f} mm from its mark")
            return True
        aim = float(np.arctan2(rel[1], rel[0]))
        # `loose_aim` ducks tolerate more heading error the closer they are.
        # The aim angle to a nearby point is mostly noise: 40 mm of lateral
        # error at 150 mm out reads as 0.4 rad and sends the duck into a turn,
        # the turn shuffles it, and it never arrives (measured: 20 s of
        # oscillation two hand-widths from the mark). Walking forward still
        # closes the gap; only the last centimetres need the heading right.
        # OFF by default, so the earlier films keep the approach they were
        # tuned and recorded with.
        tol = 1.0 if (getattr(self, "loose_aim", False) and dist <= 0.18) else AIM_TOL
        if abs(aim) > tol:
            self.vel(0.0, 0.0, turn_cmd(aim, 0.0))          # turn on the spot first
        else:
            self.vel(cruise, 0.0, float(np.clip(TURN_GAIN * aim, -1.0, 1.0)))
        return False

    def face(self, yaw_target, tol=0.20):
        """Turn on the spot until facing `yaw_target`."""
        err = float(np.arctan2(np.sin(yaw_target - self.yaw()), np.cos(yaw_target - self.yaw())))
        if abs(err) < tol:
            self.stop()
            return True
        self.vel(0.0, 0.0, turn_cmd(err, 0.0))
        return False

    def back_to(self, target_xy, heading, tol=0.035):
        """Reverse until the trunk is over `target_xy`, holding `heading`.

        Reverse walking yaws the duck; left alone it drifted 50 degrees in
        two seconds and sat down a nest-width off. So the heading is held with
        the same bang-bang turn while it backs."""
        rel = self.to_local(target_xy)
        err = float(np.arctan2(np.sin(heading - self.yaw()), np.cos(heading - self.yaw())))
        if np.hypot(*rel) < tol:
            self.stop()
            self.reset_progress()
            return True
        if self.clock < self._unstick_until:
            self.vel(CRUISE, 0.0, 0.0)          # unstick a reverse by going forward
            return False
        self.stalled(float(np.hypot(*rel)), patience=10.0)   # reverse is slow, 2-3 cm/s
        if abs(err) > 0.35:
            self.vel(0.0, 0.0, turn_cmd(err, 0.0))
            return False
        self.vel(-CRUISE if rel[0] < 0 else CRUISE, 0.0, 0.0)   # a turn mixed in stalls it
        return False

    def head(self, neck=0.0, pitch=0.0, yaw=0.0, roll=0.0):
        self.policy.head_offset[:] = np.clip((neck, pitch, yaw, roll), -1.1, 1.1)
        self.policy._update_command()

    def look_at(self, world_pt, pitch_bias=0.0, gain=1.0):
        """Aim the head at a point with the head-pose command."""
        head = self.data.xpos[self.head_body]
        v = np.asarray(world_pt) - head
        yaw = float(np.arctan2(v[1], v[0]) - self.yaw())
        yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
        pitch = -float(np.arctan2(v[2], np.hypot(v[0], v[1])))
        self.head(0.0, np.clip(gain * pitch + pitch_bias, -1.1, 1.1), np.clip(gain * yaw, -1.1, 1.1))

    def lean(self, x=0.0, z=0.0, pitch=0.0, neck=0.0, head_pitch=0.0):
        """Body-pose command for the standing policy plus a head command."""
        self.policy.body_cmd[:] = (np.clip(x, -0.02, 0.02), 0.0, np.clip(z, -0.03, 0.03),
                                   0.0, np.clip(pitch, -0.52, 0.52), 0.0)
        self.head(neck, head_pitch, 0.0, 0.0)

    def relax(self):
        self.policy.body_cmd[:] = 0.0
        self.head()

    def sit(self, on):
        if bool(self.policy.sit_mode) != bool(on):
            with quiet():
                self.policy.toggle_sit()

    def bow(self):
        with quiet():
            self.policy.trigger_ground_pick()

    @property
    def bowing(self):
        return self.policy.ground_pick_mode

    # -- one control tick --------------------------------------------------

    def tick(self):
        d = self.data
        a = CONTROL_DT / 0.5
        self.v_filt += a * (d.qvel[self.dofadr:self.dofadr + 2] - self.v_filt)
        if self.policy is not None:
            with quiet():
                self.policy.update_ground_pick_phase(CONTROL_DT)
                self.policy.update_behavior(CONTROL_DT)
                self.policy._update_command()
            action = self.policy.infer()
            d.ctrl[self.servo] = self.policy.default_pose + action * self.policy.action_scale
        elif self.pose_target is not None:
            d.ctrl[self.servo] = self.pose_target
        d.ctrl[self.mouth] = self.mouth_cmd


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


def add_ring_welds(world):
    """The ring is carried the way the cube is in task_pick_up.py: a weld to the
    jaw, and later a weld to her neck. Both exist from compile time, inactive."""
    for name, body in (("ring_beak", "he_mouth_jaw"), ("ring_neck", "she_neck_pitch")):
        eq = world.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.name, eq.name1, eq.name2 = name, body, "ring"
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.active = False
        eq.solref = (0.02, 1.0)


def load_world():
    """`duck_sim.load_scene`'s overrides, on the four-duck world."""
    model = stage.build_model(hook=add_ring_welds)
    for prefix, (_, _, scale) in stage.CAST.items():
        if scale < 1.0:
            stage.scale_actuation(model, prefix, scale)
    model.opt.timestep = CONTROL_TIMESTEP
    from bam.model import load_model
    kt = load_model(motor_name="xl330", model="m6").kt.value
    limit = kt * 1.75
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if name.startswith(("she_", "he_")):
            model.actuator_forcerange[a] = (-limit, limit)
    model.actuator_forcelimited[:] = 1
    return model, mujoco.MjData(model)


def weld(model, data, eq_id, parent, child, child_q, seat=None, quat=None):
    """Activate a weld with the child where it is (or at `seat`), so the
    constraint starts satisfied and nothing snaps -- task_pick_up.grip()."""
    if seat is not None:
        data.qpos[child_q:child_q + 3] = seat
        data.qpos[child_q + 3:child_q + 7] = data.xquat[parent] if quat is None else quat
        mujoco.mj_forward(model, data)
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, data.xquat[parent])
    relpos = np.zeros(3)
    mujoco.mju_rotVecQuat(relpos, data.xpos[child] - data.xpos[parent], neg)
    relquat = np.zeros(4)
    mujoco.mju_mulQuat(relquat, neg, data.xquat[child])
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = relpos
    model.eq_data[eq_id, 6:10] = relquat
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1


# ---------------------------------------------------------------------------
# The story
# ---------------------------------------------------------------------------

STAGE_Y = stage.STAGE_Y
NEST = np.array(stage.NEST)
AXIS = np.array([np.cos(stage.BROOD_YAW), np.sin(stage.BROOD_YAW)])   # she faces the camera
LAY = NEST + 0.120 * AXIS        # where she sits to lay; the eggs land 5 cm behind her tail
LAY_WAY = NEST - 0.28 * AXIS     # she comes THROUGH the empty nest, so she arrives facing the camera
BROOD = NEST + 0.02 * AXIS       # where a parent sits to brood: the eggs are 10 cm behind its trunk, against its tail
NEST_IN = BROOD + 0.12 * AXIS    # where a parent turns before reversing on; reverse walking drifts ~40% sideways
NEST_OUT = NEST + 0.34 * AXIS
SIDE_R = np.array([np.sin(stage.BROOD_YAW), -np.cos(stage.BROOD_YAW)])   # her right, when she faces BROOD_YAW
HER_SIDE = LAY + 0.06 * AXIS + 0.42 * SIDE_R   # he waits beside her head, on her right
WATCH = {"she": np.array([0.24, 0.20]), "he": np.array([-0.22, 0.21])}


class Story:
    def __init__(self, verbose=True):
        self.model, self.data = load_world()
        m, d = self.model, self.data
        self.say = print if verbose else (lambda *a, **k: None)

        self.she = Agent(m, d, "she", 0.16, STAGE_Y, np.pi)
        self.he = Agent(m, d, "he", -0.62, STAGE_Y, 0.0)
        self.kids = {}
        for k, egg_i in (("kidb", 1), ("kidp", 0)):       # blue on his side, pink on hers
            xy = stage.egg_spots()[egg_i]
            kid = Agent(m, d, k, xy[0], xy[1], stage.BROOD_YAW, has_policy=False)
            kid.egg = egg_i
            kid.pose_target = stage.CURL.copy()[:14] * 0 + self.curl_pose()
            d.qpos[kid.adr + 2] = 0.03
            for i, q in enumerate(kid.qpos_idx):
                d.qpos[q] = kid.pose_target[i]
            self.kids[k] = kid
        self.adults = (self.she, self.he)

        # collisions: a duckling is nobody's business until it hatches
        self.kid_bits = {}
        for kid in self.kids.values():
            for g in range(m.ngeom):
                if m.geom_bodyid[g] in {m.geom_bodyid[gg] for gg in range(m.ngeom)
                                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gg]) or "").startswith(kid.prefix + "_")} \
                        and m.geom_contype[g]:
                    self.kid_bits[g] = (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
                    m.geom_contype[g] = m.geom_conaffinity[g] = 0

        eqid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        self.eq_beak, self.eq_neck = eqid("ring_beak"), eqid("ring_neck")
        self.ring = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ring")
        self.ring_q = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ring_free")])
        self.eggs = []
        for i in (0, 1):
            e = {}
            for half in ("bot", "top"):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"egg{i}_{half}_free")
                e[half] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"egg{i}_{half}")
                e[half + "_q"], e[half + "_v"] = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
            e["eq"] = eqid(f"egg{i}_weld")
            e["laid"] = e["popped"] = False
            e["rock_from"] = None
            self.eggs.append(e)
        self.props = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("bow", "bowtie")}
        self.hearts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"heart{i}") for i in range(stage.N_HEARTS)]
        self.heart_bursts = []      # (t0, t1, kind)

        mujoco.mj_forward(m, d)
        # the ring starts in his beak
        tip = self.he.beak()
        fwd = tip - d.xpos[self.he.head_body]
        fwd /= np.linalg.norm(fwd)
        weld(m, d, self.eq_beak, self.he.jaw, self.ring, self.ring_q,
             seat=tip - fwd * 0.004, quat=self.ring_quat(fwd))
        mujoco.mj_forward(m, d)

        self.t = 0.0
        self.phase_i = 0
        self.phase_t0 = 0.0
        self.mark = {}
        self.log = []
        self.cam_target = [np.array([-0.2, STAGE_Y, 0.15]), 1.2, 95.0, -12.0]
        self.cam_now = [np.array([-0.2, STAGE_Y, 0.15]), 1.2, 95.0, -12.0]
        self.phases = self.script()

    @staticmethod
    def curl_pose():
        """A duckling's servo targets: sitting, head down.

        Not the SIT keyframe. A passive duck in that pose balances on its two
        shins and topples in three seconds; the sit-stand policy holds the
        adult there, nothing holds a duckling. This pose came out of a search
        over seated leg angles for the one a passive duckling stays upright
        in for eight seconds with its head bobbing: knees fully bent, ankles
        turned so the feet take weight -- a chick squatting on its heels."""
        q = stage.pose((0.0, 0.0, -0.35, 1.50, 0.40), neck=(0.50, 0.55, 0.0, 0.0))
        return q[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]].astype(np.float32)

    @staticmethod
    def stand_pose():
        return stage.STAND[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]].astype(np.float32)

    @staticmethod
    def ring_quat(axis):
        R = stage.frame_from_z(axis)
        return stage.mat_quat(R)

    # -- helpers the phases use ---------------------------------------------

    def touching(self, a, b):
        """Any contact between duck a's head geoms and duck b's head geoms."""
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in a.head_geoms and c.geom2 in b.head_geoms) or \
               (c.geom2 in a.head_geoms and c.geom1 in b.head_geoms):
                return True
        return False

    def contact_between(self, a, b):
        d, m = self.data, self.model
        ba = {m.geom_bodyid[g] for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "").startswith(a.prefix + "_")}
        bb = {m.geom_bodyid[g] for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "").startswith(b.prefix + "_")}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (g1 in ba and g2 in bb) or (g1 in bb and g2 in ba):
                return True
        return False

    def note(self, msg):
        self.say(f"t={self.t:6.1f}s  {msg}")
        self.log.append((self.t, msg))

    def hearts_burst(self, seconds, kind="pair"):
        self.heart_bursts.append((self.t, self.t + seconds, kind))

    def camera(self, look, dist, az, elev):
        self.cam_target = [np.asarray(look, float), dist, az, elev]

    def settle_on_nest(self, a, S, key, t):
        """Bring duck `a` onto the eggs and sit it down: to the mark in front
        of the nest, turn to the brooding heading, reverse onto the nest with
        the heading held, sit. Reverse walking drifts sideways (the policy
        cannot strafe), so if it ends up wide it goes round again. Returns True
        once sitting. State lives in S under `key`."""
        st = S.setdefault(key, {"stage": "in", "t0": t, "tries": 0})
        stage_, age = st["stage"], t - st["t0"]
        if stage_ == "in":
            if a.drive_to(NEST_IN, stop_r=0.05) or age > 20:
                st.update(stage="turn", t0=t)
        elif stage_ == "turn":
            if a.face(stage.BROOD_YAW, tol=0.25) or age > 8:
                st.update(stage="back", t0=t)
        elif stage_ == "back":
            if a.back_to(BROOD, stage.BROOD_YAW, tol=0.06) or age > 22:
                off = float(np.linalg.norm(a.xy() - BROOD))
                if off > 0.09 and st["tries"] < 2:
                    st.update(stage="in", t0=t, tries=st["tries"] + 1)
                    self.note(f"{a.prefix} is {off*1000:.0f} mm off the nest -- going round again")
                else:
                    st.update(stage="sit", t0=t)
                    a.stop()
        elif stage_ == "sit":
            if age > 0.8:
                a.sit(True)
                st.update(stage="done", t0=t)
                self.note(f"{a.prefix} broods ({1000*np.linalg.norm(a.xy() - BROOD):.0f} mm off its mark)")
        return st["stage"] == "done"

    # -- the script: (name, timeout, update) --------------------------------

    def script(self):
        she, he = self.she, self.he
        S = {}

        def meet(t):
            he_done = he.drive_to(she.xy(), stop_r=0.07, ahead=0.36)
            she.look_at(self.data.xpos[he.head_body] + [0, 0, 0.02])
            if he_done:
                he.stop()
                he.look_at(self.data.xpos[she.head_body])
                S.setdefault("settled", t)
                return t - S["settled"] > 2.0
            return False

        def propose(t):
            she.look_at(self.data.xpos[he.head_body] + [0, 0, 0.01])
            if "bowed" not in S:
                he.head()
                he.bow()
                S["bowed"] = t
                self.note("he bows, ring in his beak")
                return False
            if he.bowing:
                she.look_at(self.data.xpos[self.ring])
                he.mouth_cmd = MOUTH_OPEN if t - S["bowed"] > 1.2 else MOUTH_SHUT
                return False
            he.mouth_cmd = MOUTH_SHUT
            S.setdefault("up", t)
            # she says yes: three nods
            dt = t - S["up"]
            if dt < 3.0:
                she.head(0.0, 0.55 * max(0.0, np.sin(2 * np.pi * dt)), 0.0, 0.0)
                he.look_at(self.data.xpos[she.head_body])
                if dt > 0.5 and "hearts" not in S:
                    S["hearts"] = True
                    self.hearts_burst(6.0)
                    self.note("she nods -- yes")
                return False
            # the ring moves to her neck
            neck = 0.5 * (self.data.xpos[she.head_body] + self.data.xpos[she.neck])
            axis = self.data.xpos[she.head_body] - self.data.xpos[she.trunk]
            self.data.eq_active[self.eq_beak] = 0
            weld(self.model, self.data, self.eq_neck, she.neck, self.ring, self.ring_q,
                 seat=neck, quat=self.ring_quat(axis))
            self.note("the ring is on her neck")
            she.head()
            return True

        def kiss(t):
            # close the gap first: he walks in until the trunks are ~0.20 m apart
            if "close" not in S:
                she_sq = she.face(float(np.arctan2(*(he.xy() - she.xy())[::-1])), tol=0.15)
                if he.drive_to(she.xy(), stop_r=0.03, ahead=0.16, cruise=0.3) and she_sq:
                    S["close"] = t
                    self.note("close enough to kiss (trunks %.0f mm apart)" % (1000 * np.linalg.norm(he.xy() - she.xy())))
                return False
            # both lean in with the standing policy until the beak tips meet
            k = min(1.0, (t - S["close"] - 1.0) / 3.0)
            if k < 0:
                he.stop(); she.stop()
                return False
            gap = float(np.linalg.norm(he.beak() - she.beak()))
            met = gap < 0.014 or self.touching(he, she)
            if met:
                S.setdefault("met", t)
                if "kissed" not in S:
                    S["kissed"] = True
                    self.note(f"beaks touch (gap {gap*1000:.0f} mm)")
                    self.hearts_burst(6.0)
            lam = S.setdefault("lam", 0.0)
            if not met:
                lam = min(1.0, lam + CONTROL_DT / 4.0)      # keep leaning until they meet
            S["lam"] = lam
            for a in (he, she):
                a.lean(x=0.02 * lam, z=0.0, pitch=-0.20 * lam, neck=-0.35 * lam,
                       head_pitch=0.20 * lam)
            hold = 4.5
            done = "met" in S and t - S["met"] > hold
            if not done and t - S["close"] > 14.0:
                self.note(f"kiss timed out (gap {gap*1000:.0f} mm)")
                done = True
            if done:
                he.relax(); she.relax()
            return done

        def to_nest(t):
            if "she_way" not in S:
                if she.drive_to(LAY_WAY, stop_r=0.08):
                    S["she_way"] = t
                she_done = False
            else:
                she_done = she.drive_to(LAY, stop_r=0.03) if "she_at" not in S else True
            if she_done and "she_at" not in S:
                off = float(np.linalg.norm(she.xy() - LAY))
                clear = float(np.linalg.norm(she.xy() - NEST))
                if (off > 0.07 or clear < 0.075) and S.get("retries", 0) < 4:
                    S["retries"] = S.get("retries", 0) + 1
                    S.pop("she_way", None)
                    self.note("she is %.0f mm off her mark -- going round again" % (1000 * off))
                    return False
                S["she_at"] = t
                self.note("she is at the nest, %.0f mm off her mark, facing %+.2f (wants %+.2f)"
                          % (1000 * off, she.yaw(), stage.BROOD_YAW))
            if "she_at" in S:
                if she.face(stage.BROOD_YAW, tol=0.30) or t - S["she_at"] > 5.0:
                    S.setdefault("she_faced", t)
            he_done = he.drive_to(HER_SIDE, stop_r=0.06) if "he_at" not in S else True
            if he_done and "he_at" not in S:
                S["he_at"] = t
            if "he_at" in S:
                he.face(float(np.arctan2(*(self.data.xpos[she.head_body][:2] - he.xy())[::-1])), tol=0.25)
            if "she_faced" in S and t - S["she_faced"] > 1.0:
                she.sit(True)
                S.setdefault("sat", t)
            return "sat" in S and "he_at" in S and t - S["sat"] > 2.5

        def lay(t):
            he.look_at(self.data.xpos[she.head_body])
            if not (she.policy.sit_mode and she.z() < 0.09):
                she.stop()
                she.sit(True)
                return False
            for i, e in enumerate(self.eggs):
                if not e["laid"] and t - S.setdefault("t0", t) > 1.5 + 3.0 * i:
                    self.lay_egg(i)
                    self.note(f"egg {i} laid")
            return all(e["laid"] for e in self.eggs) and t - S["t0"] > 8.0

        def head_kiss(t):
            # he walks to her side, then leans his head down onto hers
            head = self.data.xpos[she.head_body][:2]
            if "at" not in S:
                # straight in along her side from where he waited, so he
                # arrives facing her head with no turn to drift on
                if he.drive_to(head + 0.11 * SIDE_R, stop_r=0.03):
                    S["at"] = t
                    self.note("at her side, %.0f mm from her head" % (1000 * np.linalg.norm(he.xy() - head)))
                return False
            if "faced" not in S:
                if he.face(float(np.arctan2(*(self.data.xpos[she.head_body][:2] - he.xy())[::-1]))) \
                        or t - S["at"] > 6.0:
                    S["faced"] = t
                    he.stop()
                return False
            k = (t - S["faced"] - 1.0) / 3.0
            if k < 0:
                return False
            touched = self.touching(he, she)
            if touched and "touch" not in S:
                S["touch"] = t
                self.note("his head rests on hers")
                self.hearts_burst(5.0)
            lam = S.setdefault("lam", 0.0)
            if not touched:
                lam = min(1.0, lam + CONTROL_DT / 4.0)
            S["lam"] = lam
            he.lean(x=0.02 * lam, z=-0.03 * lam, pitch=-0.35 * lam, neck=0.2 * lam,
                    head_pitch=1.0 * lam)
            done = ("touch" in S and t - S["touch"] > 3.5) or (t - S["faced"] > 14.0)
            if done:
                he.relax()
                if "touch" not in S:
                    self.note("head kiss timed out")
            return done

        def step_back(t):
            # he backs off her and goes to wait; she stands, backs onto the eggs
            if t - S.setdefault("t0", t) < 1.5:
                he.vel(-CRUISE, 0, 0)
                return False
            he_done = he.drive_to(HER_SIDE + np.array([0.0, -0.10]), stop_r=0.04) if "he_away" not in S else True
            if he_done:
                S.setdefault("he_away", t)
                he.face(float(np.arctan2(*(NEST - he.xy())[::-1])))
            if "stood" not in S:
                she.sit(False)
                S["stood"] = t
                self.note("she stands to settle onto the eggs")
            if t - S["stood"] < 2.5:
                return False
            if self.settle_on_nest(she, S, "she_nest", t):
                S.setdefault("brood", t)
            return "brood" in S and "he_away" in S and t - S["brood"] > 7.0

        def swap(t):
            # she leaves forwards; he comes to the front, turns, backs on, sits
            if "she_up" not in S:
                she.sit(False)
                S["she_up"] = t
            if t - S["she_up"] < 2.5:
                return False
            if "she_out" not in S:
                if she.drive_to(NEST_OUT, stop_r=0.04):
                    S["she_out"] = t
                return False
            if "she_watch" not in S:
                if she.drive_to(WATCH["she"], stop_r=0.04):
                    S["she_watch"] = t
            else:
                she.face(float(np.arctan2(*(NEST - she.xy())[::-1])))
                she.look_at([*NEST, 0.05])
            if "she_out" in S:
                if self.settle_on_nest(he, S, "he_nest", t):
                    S.setdefault("brood", t)
            return "brood" in S and "she_watch" in S and t - S["brood"] > 7.0

        def hatch(t):
            if "he_up" not in S:
                he.sit(False)
                S["he_up"] = t
            if t - S["he_up"] < 2.5:
                return False
            if "he_out" not in S:
                if he.drive_to(NEST_OUT + np.array([-0.05, 0]), stop_r=0.05) or he.up() < 0.5:
                    S["he_out"] = t
                return False
            if "he_watch" not in S:
                if he.drive_to(WATCH["he"], stop_r=0.05) or he.up() < 0.5:
                    S["he_watch"] = t
                    self.note("both watch the eggs")
                    for i, e in enumerate(self.eggs):
                        e["rock_from"] = t + 1.0 + 1.5 * i
                return False
            he.face(float(np.arctan2(*(NEST - he.xy())[::-1])))
            he.look_at([*NEST, 0.05])
            she.look_at([*NEST, 0.05])
            for i, e in enumerate(self.eggs):
                if not e["popped"] and t > e["rock_from"] + 4.0:
                    self.pop_egg(i)
                    self.note(f"egg {i} hatches")
                    if all(x["popped"] for x in self.eggs):
                        self.hearts_burst(8.0, "family")
            return all(e["popped"] for e in self.eggs) and t > self.eggs[1]["rock_from"] + 16.0

        return [("meet", 40.0, meet), ("propose", 25.0, propose), ("kiss", 40.0, kiss),
                ("to the nest", 60.0, to_nest), ("laying", 14.0, lay),
                ("head kiss", 45.0, head_kiss), ("brooding", 60.0, step_back),
                ("taking turns", 90.0, swap), ("hatching", 60.0, hatch)]

    # -- eggs and ducklings ---------------------------------------------------

    def lay_egg(self, i):
        d, e = self.data, self.eggs[i]
        xy = stage.egg_spots()[i]
        for half in ("bot", "top"):
            q = e[half + "_q"]
            d.qpos[q:q + 3] = (xy[0], xy[1], stage.NEST_LINING_Z + stage.EGG_SEMI[1] + 0.012)
            d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
            d.qvel[e[half + "_v"]:e[half + "_v"] + 6] = 0.0
        e["laid"] = True

    def pop_egg(self, i):
        m, d, e = self.model, self.data, self.eggs[i]
        d.eq_active[e["eq"]] = 0
        v = e["top_v"]
        side = -1.0 if i == 0 else 1.0
        d.qvel[v:v + 3] = (side * 0.45, -0.30, 1.05)
        d.qvel[v + 3:v + 6] = (4.0 * side, 6.0, 2.0)
        for kid in self.kids.values():
            if kid.egg == i:
                kid.hatched_at = self.t
                for g, (ct, ca) in self.kid_bits.items():
                    if m.geom_bodyid[g] in {m.geom_bodyid[gg] for gg in range(m.ngeom)
                                            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gg]) or "").startswith(kid.prefix + "_")}:
                        m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        e["popped"] = True

    def eggs_tick(self):
        d = self.data  # noqa
        for i, e in enumerate(self.eggs):
            d.xfrc_applied[e["bot"]] = 0.0
            if e["laid"] and not e["popped"] and e["rock_from"] is not None and self.t > e["rock_from"]:
                w = min(1.0, (self.t - e["rock_from"]) / 4.0)
                d.xfrc_applied[e["bot"], 3] = w * 4e-5 * np.sin(2 * np.pi * 2.1 * self.t + 2.1 * i)
                d.xfrc_applied[e["bot"], 4] = w * 3e-5 * np.sin(2 * np.pi * 1.4 * self.t + i)
        for kid in self.kids.values():
            geoms = [g for g in range(self.model.ngeom)
                     if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, self.model.geom_dataid[g]) or "").startswith(kid.prefix + "_")]
            if not hasattr(kid, "hatched_at"):
                alpha = 0.0
                # nothing touches it yet, so nothing holds it up: keep it in its
                # egg, at the height a sitting duckling actually has (the adult
                # sits with its trunk at 59 mm; a fifth of that) -- held any
                # higher it drops when released and tips onto its back
                xy = stage.egg_spots()[kid.egg]
                d.qpos[kid.adr:kid.adr + 3] = (xy[0], xy[1], stage.NEST_LINING_Z + 0.059 * kid.scale)
                d.qpos[kid.adr + 3:kid.adr + 7] = (np.cos(stage.BROOD_YAW / 2), 0, 0, np.sin(stage.BROOD_YAW / 2))
                d.qvel[kid.dofadr:kid.dofadr + 6] = 0.0
            else:
                dt = self.t - kid.hatched_at
                alpha = float(np.clip((dt - 0.25) / 0.6, 0.0, 1.0))
                # A newborn stays sitting in its shell -- there is no policy at
                # a fifth scale. It keeps its head DOWN and forward: a seated
                # duck that lifts its head puts its centre of mass behind its
                # tail, and with nothing balancing it, it goes over on its back
                # (a standing ramp and a neck lift were both tried; both did).
                # It looks left and right a little and opens its beak to peep.
                pose = self.curl_pose().copy()
                w = float(np.clip((dt - 1.5) / 2.0, 0.0, 1.0))
                pose[6] = 0.55 + 0.10 * w * np.sin(2 * np.pi * 0.30 * dt)
                pose[7] = 0.30 * w * np.sin(2 * np.pi * 0.17 * dt + kid.egg)
                kid.pose_target = pose
                kid.mouth_cmd = 0.25 * max(0.0, np.sin(2 * np.pi * 0.6 * dt)) if dt > 3 else 0.0
            for g in geoms:
                self.model.geom_rgba[g, 3] = alpha

    # -- props: bow, bow tie, hearts ------------------------------------------

    def props_tick(self, cam_az):
        m, d = self.model, self.data
        mid = lambda b: int(m.body_mocapid[b])
        hR = d.xmat[self.she.head_body].reshape(3, 3)
        top = d.xpos[self.she.head_body] + hR @ np.array([0.0, 0.0, 0.0])
        # the bow sits on the crown: highest point of the head shell
        d.mocap_pos[mid(self.props["bow"])] = d.xpos[self.she.head_body] + np.array([0, 0, 0.055]) + hR @ np.array([-0.01, 0, 0])
        d.mocap_quat[mid(self.props["bow"])] = d.xquat[self.she.head_body]
        tR = d.xmat[self.he.trunk].reshape(3, 3)
        d.mocap_pos[mid(self.props["bowtie"])] = d.xpos[self.he.trunk] + tR @ np.array([0.045, 0.0, 0.030])
        d.mocap_quat[mid(self.props["bowtie"])] = d.xquat[self.he.trunk]

        rng = np.random.default_rng(5)
        offs = rng.uniform(-1, 1, (stage.N_HEARTS, 3)) * np.array([0.10, 0.06, 0.03])
        life, rise = 2.8, 0.20
        for i, b in enumerate(self.hearts):
            shown = False
            for t0, t1, kind in self.heart_bursts:
                stagger = t0 + (i * 0.31) % max(0.9, (t1 - t0) - life)
                if not (stagger <= self.t <= min(stagger + life, t1 + life)):
                    continue
                u = (self.t - stagger) / life
                if u > 1:
                    continue
                if kind == "pair":
                    src = 0.5 * (d.xpos[self.she.head_body] + d.xpos[self.he.head_body]) + [0, 0, 0.08]
                else:
                    src = np.array([*NEST, 0.16])
                pos = src + offs[i] + np.array([0.05 * np.sin(3 * u + i), 0, rise * u])
                d.mocap_pos[mid(b)] = pos
                d.mocap_quat[mid(b)] = stage.rpy_quat(0, 0, np.radians(cam_az + 90))
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == b:
                        m.geom_rgba[g, 3] = min(1.0, 3.5 * u, 3.0 * (1 - u))
                shown = True
                break
            if not shown:
                d.mocap_pos[mid(b)] = (0, 0, -1)

    # -- camera: follows the phase, smoothly ---------------------------------

    def camera_tick(self):
        name = self.phases[self.phase_i][0] if self.phase_i < len(self.phases) else "end"
        she, he = self.she, self.he
        mid = 0.5 * (she.xy() + he.xy())
        if name == "meet":
            self.camera([*mid, 0.15], 0.85, 92, -10)
        elif name == "propose":
            self.camera([*mid, 0.12], 0.66, 86, -7)
        elif name == "kiss":
            self.camera([*mid, 0.17], 0.50, 90, -5)
        elif name == "to the nest":
            self.camera([*(0.5 * (mid + NEST)), 0.13], 0.80, 92, -13)
        elif name in ("laying", "head kiss"):
            self.camera([*(she.xy() + 0.3 * (he.xy() - she.xy())), 0.12], 0.60, 84, -10)
        elif name in ("brooding", "taking turns"):
            self.camera([*NEST, 0.10], 0.60, 92, -13)
        elif name == "hatching":
            self.camera([*NEST, 0.07], 0.48, 90, -10)
        else:
            self.camera([*NEST, 0.10], 0.80, 70, -13)
        # ease towards the target
        a = 1.0 - np.exp(-CONTROL_DT / 1.2)
        self.cam_now[0] = self.cam_now[0] + a * (self.cam_target[0] - self.cam_now[0])
        for k in (1, 2, 3):
            self.cam_now[k] += a * (self.cam_target[k] - self.cam_now[k])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.cam_now[0]
        cam.distance, cam.azimuth, cam.elevation = self.cam_now[1], self.cam_now[2], self.cam_now[3]
        return cam

    # -- one control tick ------------------------------------------------------

    def step(self):
        """Advance one 50 Hz control tick. Returns False when the story is over."""
        if self.phase_i >= len(self.phases):
            return False
        name, timeout, fn = self.phases[self.phase_i]
        if self.t - self.phase_t0 == 0.0:
            self.note(f"--- {name} ---")
        done = fn(self.t - self.phase_t0 + 0.0) if False else fn(self.t)
        if done or self.t - self.phase_t0 > timeout:
            if not done:
                self.note(f"{name}: timed out")
            self.phase_i += 1
            self.phase_t0 = self.t
            self.phases_state_reset()
        self.eggs_tick()
        for a in (*self.adults, *self.kids.values()):
            a.clock = self.t
            a.tick()
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self.t += CONTROL_DT
        return True

    def phases_state_reset(self):
        # each phase closure keeps its scratch in S (shared); clear it between phases
        for _, _, fn in self.phases:
            pass
        self.script_state_clear()

    def script_state_clear(self):
        # the closures share one dict S captured in script(); rebuild the list
        # of closures with a fresh S but keep the phase index
        self.phases = self.script()


def stage_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------


def run(args):
    story = Story(verbose=True)
    writer = renderer = None
    frames = 0
    every = 2                                   # 50 Hz control -> 25 fps, like duck_sim.Recorder
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(story.model, height=args.height, width=args.width)
        writer = imageio.get_writer(args.video, fps=1.0 / (every * CONTROL_DT), quality=8,
                                    macro_block_size=8)
    i = 0
    fell = None
    while story.step() and story.t < args.seconds:
        cam = story.camera_tick()
        story.props_tick(story.cam_now[2])
        if writer is not None and i % every == 0:
            renderer.update_scene(story.data, camera=cam)
            writer.append_data(renderer.render())
            frames += 1
        for a in story.adults:
            if a.up() < 0.3 and fell is None:
                fell = (a.prefix, story.t)
                story.note(f"{a.prefix} FELL OVER -- stopping")
        if fell:
            break
        if args.trace and i % 50 == 0:
            she, he = story.she, story.he
            name = story.phases[story.phase_i][0] if story.phase_i < len(story.phases) else "end"
            gap = np.linalg.norm(he.beak() - she.beak())
            print("  %6.1f %-12s she %s yaw %+.2f z %.3f up %.2f %-8s | he %s yaw %+.2f z %.3f up %.2f %-8s | gap %.0f mm"
                  % (story.t, name, np.round(she.xy(), 2), she.yaw(), she.z(), she.up(), she.policy.current_policy,
                     np.round(he.xy(), 2), he.yaw(), he.z(), he.up(), he.policy.current_policy, 1000 * gap), flush=True)
        i += 1
    if writer is not None:
        writer.close()
        print(f"wrote {args.video} ({frames} frames)")
    print()
    print(f"story ran {story.t:.0f} s;  fell: {fell or 'nobody'}")
    for j, e in enumerate(story.eggs):
        print(f"  egg {j} bottom at {np.round(story.data.xpos[e['bot']], 3)}  (cradle {np.round(stage.egg_spots()[j], 3)})")
    for k, kid in story.kids.items():
        print(f"  {k}: hatched {'yes' if hasattr(kid, 'hatched_at') else 'no'}, z {kid.z()*1000:.0f} mm, up {kid.up():.2f}")
    for t, msg in story.log:
        if "brood" in msg or "kiss" in msg or "touch" in msg or "side" in msg:
            print(f"  [{t:6.1f}] {msg}")
    return 0 if fell is None else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--dry", action="store_true", help="no rendering; log the beats")
    ap.add_argument("--trace", action="store_true", help="one line a second: where everyone is")
    ap.add_argument("--seconds", type=float, default=400.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
