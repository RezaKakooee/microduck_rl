#!/usr/bin/env python3
"""Rocker balance board: the duck stands with both feet on a plank that rests
on a free cylinder lying on the floor (a "rola bola"), and must keep either
end of the plank off the floor.

Two orientations:
  A  plank long axis along the duck's LATERAL axis (world y), cylinder axis
     along x. The board tilts in ROLL. The controller drives both hip_rolls
     (same sign) to shift weight between the feet.
  B  plank long axis along the duck's FORWARD axis (world x), cylinder axis
     along y. The board tilts in PITCH. The controller drives both ankles.

The scene is written from the template scene_board.xml into
scene_board_<tag>.xml so the geometry can be varied from the constants below
or the CLI. Setup: the cylinder and plank are settled on the floor first, the
duck is then rested on the plank by a contact-free binary search on its
trunk height, and it settles for SETTLE_S with the board HELD level. At t = 0
the board is released and the clock starts.

Failure = any plank-floor contact, a foot with no plank contact for
FOOT_GRACE_S, any non-foot robot geom touching anything, or the trunk under
FALL_HEIGHT above the plank. The run reports seconds held, the largest plank
tilt while held, and whether a --push impulse was survived.

    uv run python -m microduck_lab.tasks.balance_board.board --orientation A --controller open
    uv run python -m microduck_lab.tasks.balance_board.board --orientation B --kp-b 2 --kd-b 0.2 --push 0.15
    uv run python -m microduck_lab.tasks.balance_board.board --grid A            # gain search
    uv run python -m microduck_lab.tasks.balance_board.board --compare           # the report table
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \\
        src/microduck_lab/tasks/balance_board/board.py --orientation A --video videos/balance_board/board_A.mp4 --azimuth 180

--video needs a GPU (EGL); everything else runs on CPU.
"""

import argparse
import contextlib
import io
import itertools
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

import mujoco  # noqa: E402
from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION  # noqa: E402
from microduck_lab.tasks.skating.ice_experts.harness import (  # noqa: E402
    HOME, L_HIP_ROLL, R_HIP_ROLL, L_ANKLE, R_ANKLE, L_HIP_PITCH, R_HIP_PITCH,
)

# The board scenes and the board-footed robot are ours; the plain robot is
# upstream. Generated scene_board_<tag>.xml files land beside the template.
ROBOT_DIR = str(paths.MODELS)
TEMPLATE = paths.model("scene_board.xml")
ROBOT_SRC = paths.robot("robot_allcollisions.xml")
ROBOT_VARIANT = paths.model("robot_allcollisions_boardfeet.xml")
FOOT_FLAT_TOL = 0.001      # sole-hull vertices this close to the bottom form the flat patch
FOOT_BOX_HALF_H = 0.003    # half-height of the generated foot box
# make_policy needs some 61->14 ONNX to build PolicyInference; it is never run.
ANY_ONNX = os.path.join(duck_sim.REPO, "spiral_v5.onnx")

# ---- board geometry (defaults; the CLI can override radius / length / thickness)
CYL_RADIUS = 0.030      # m
# The roller is a CAPSULE of the cylinder's radius: MuJoCo has a native
# capsule-box collider (2 contacts along the line, 0.1 mm dip under the duck),
# while cylinder-box goes through the convex collider, which under load drops
# to 0-1 contacts and let the plank sink 21 mm (measured). The plank (+/-60 mm)
# rests on the straight part (+/-90 mm), so it rolls exactly like a cylinder.
ROLLER = "capsule"      # "cylinder" reproduces the artefact
CYL_LENGTH = 0.18       # m, along its axis
CYL_MASS = 0.060        # kg
PLANK_LEN = 0.30        # m, long axis = the tilting direction
PLANK_WID = 0.12        # m
PLANK_THK = 0.012       # m
PLANK_MASS = 0.070      # kg
FRICTION = 1.5
# Contact stiffness for everything the board touches. MuJoCo's soft contact
# scales with the LIGHTER body's inverse mass, so with the defaults a 70 g
# plank loaded by a 737 g duck sank 13 mm into the cylinder and the cylinder
# 5 mm into the floor at release (measured). solref 0.01 s (2 x timestep) and
# impedance 0.99-0.999 make these contacts nearly rigid; priority 1 makes the
# board's values win over the floor's defaults.
CONTACT_SOL = 'solref="0.02 1" solimp="0.99 0.999 0.001"'
CONTACT = CONTACT_SOL + ' priority="1"'

# ---- protocol
SETTLE_S = 1.0          # duck settles on the HELD board this long before release
FOOT_GRACE_S = 0.10     # a foot may lose plank contact this long (contact flicker)
FALL_HEIGHT = duck_sim.FALL_HEIGHT   # trunk this far above the plank top = fallen
AFTER_FAIL_S = 1.0      # keep simulating after a failure so a video shows it
HIP_ROLL_LIMIT = 0.384
HOLD_MASS_SCALE = 1e4   # plank/cylinder mass factor while the board is held
# Upright PD on the trunk's projected-gravity roll/pitch (two_leg_glide.py
# structure), per axis (kp, kd), chosen on the HELD board with `--stand-grid`.
STAND = {"roll": (2.0, 0.1), "pitch": (5.0, 0.3)}

# Actuation sign, MEASURED with both feet planted (see --levers):
#   hip_rolls: +u on BOTH (same sign) rolls the TRUNK to the left (+roll); the
#     legs stay vertical on the planted feet and the trunk + head swing about
#     the hips, so the CoM goes LEFT. This is the opposite of the one-leg
#     harness convention, where +0.384 on both moved the body over the right
#     foot: there the free leg hung and the support leg was the lever.
#   ankles: L +a, R -a pitches the body BACKWARD (CoM to -x).
# The controllers below therefore command "u = CoM toward the -axis" and map
# it as hip_rolls -= u (A / roll) and L_ANKLE += u, R_ANKLE -= u (B / pitch).
# "tilt > 0" means the +axis side of the plank is down, so u = +k * tilt.

AXIS = {"A": 1, "B": 0}            # world axis the plank tilts along (index into xyz)


def ensure_robot_variant(force=False):
    """robot_allcollisions.xml plus one box geom per foot (`left_foot_box`,
    `right_foot_box`) that matches the FLAT patch of the sole's convex hull.

    Why: a mesh sole on a box plank goes through the general convex collider,
    which returns a cluster of contacts around the deepest point and lets the
    foot roll forward on the plank (measured: the duck pitched 10 deg in 0.4 s
    on a held, level plank). Box-box is a native face contact. The box has
    contype/conaffinity 0 and is only collided through explicit <pair>s with
    the plank; the sole mesh still collides with the floor, so the robot is
    unchanged on flat ground. Written once; delete the file to regenerate."""
    if os.path.exists(ROBOT_VARIANT) and not force:
        return ROBOT_VARIANT
    model, data = duck_sim.load_scene("walk")
    with contextlib.redirect_stdout(io.StringIO()):
        duck_sim.make_policy(model, data, walking_onnx_path=ANY_ONNX)   # HOME pose: soles level
    with open(ROBOT_SRC) as f:
        src = f.read()
    info = []
    for side in ("left", "right"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        b = int(model.geom_bodyid[g])
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        w = model.mesh_vert[va:va + vn] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        zmin = w[:, 2].min()
        flat = w[w[:, 2] < zmin + FOOT_FLAT_TOL]
        lo, hi = flat.min(axis=0), flat.max(axis=0)
        half = ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, FOOT_BOX_HALF_H)
        centre_w = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin + FOOT_BOX_HALF_H])
        R = data.xmat[b].reshape(3, 3)
        centre_b = R.T @ (centre_w - data.xpos[b])
        quat_b = np.zeros(4)
        mujoco.mju_mat2Quat(quat_b, R.T.flatten())        # world-aligned box, in the body frame
        line = (f'<geom name="{side}_foot_box" type="box" size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" '
                f'pos="{centre_b[0]:.5f} {centre_b[1]:.5f} {centre_b[2]:.5f}" '
                f'quat="{quat_b[0]:.6f} {quat_b[1]:.6f} {quat_b[2]:.6f} {quat_b[3]:.6f}" '
                f'contype="0" conaffinity="0" group="3" rgba="0.2 0.8 0.2 0.4"/>')
        anchor = f'name="{side}_foot_collision"'
        i = src.index(anchor)
        j = src.index("/>", i) + 2
        src = src[:j] + "\n                " + line + src[j:]
        info.append(f"{side}: flat patch {2000 * half[0]:.1f} x {2000 * half[1]:.1f} mm at world "
                    f"({1000 * centre_w[0]:+.1f}, {1000 * centre_w[1]:+.1f}) mm, sole bottom z={1000 * zmin:.1f} mm")
    src = src.replace("<mujoco", "<!-- GENERATED by src/microduck_lab/tasks/balance_board/board.py (ensure_robot_variant) from "
                      "robot_allcollisions.xml: adds left_foot_box / right_foot_box; do not edit. -->\n<mujoco", 1)
    tmp = f"{ROBOT_VARIANT}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, ROBOT_VARIANT)
    print("wrote", ROBOT_VARIANT, "|", "; ".join(info))
    return ROBOT_VARIANT


def write_scene(orientation, radius=CYL_RADIUS, cyl_len=CYL_LENGTH, cyl_mass=CYL_MASS,
                plank_len=PLANK_LEN, plank_wid=PLANK_WID, plank_thk=PLANK_THK,
                plank_mass=PLANK_MASS, friction=FRICTION, roller=None, fixed_roller=False):
    """Write scene_board_<tag>.xml next to the template with the asked geometry.
    `fixed_roller` drops the roller's freejoint: a rocker fixed to the floor
    (a wobble board) instead of the free rola-bola roller. Diagnostic only."""
    roller = ROLLER if roller is None else roller
    ensure_robot_variant()
    with open(TEMPLATE) as f:
        src = f.read()
    if orientation == "A":
        cyl_quat = "0.7071068 0 0.7071068 0"        # cylinder axis along x
        size = (plank_wid / 2, plank_len / 2, plank_thk / 2)
    elif orientation == "B":
        cyl_quat = "0.7071068 0.7071068 0 0"        # cylinder axis along y
        size = (plank_len / 2, plank_wid / 2, plank_thk / 2)
    else:
        sys.exit(f"orientation must be A or B, got {orientation!r}")
    zc = radius
    zp = 2 * radius + plank_thk / 2
    fr = f"{friction} 0.005 0.0001"
    cyl = (f'<body name="cylinder" pos="0 0 {zc:.4f}">' + ("" if fixed_roller else '<freejoint name="cylinder_freejoint" />')
           + f'<geom name="cylinder" type="{roller}" size="{radius:.4f} {cyl_len / 2:.4f}" quat="{cyl_quat}" '
           f'mass="{cyl_mass:.3f}" friction="{fr}" {CONTACT} rgba="0.55 0.55 0.60 1" /></body>')
    plank = (f'<body name="plank" pos="0 0 {zp:.4f}"><freejoint name="plank_freejoint" />'
             f'<geom name="plank" type="box" size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" '
             f'mass="{plank_mass:.3f}" friction="{fr}" {CONTACT} rgba="0.90 0.35 0.10 1" /></body>')
    contact = ('<contact>\n'
               '        <exclude body1="ankle_left" body2="plank" />\n'
               '        <exclude body1="ankle_right" body2="plank" />\n'
               f'        <pair geom1="left_foot_box" geom2="plank" friction="{friction} {friction} 0.005 0.0001 0.0001" {CONTACT_SOL} />\n'
               f'        <pair geom1="right_foot_box" geom2="plank" friction="{friction} {friction} 0.005 0.0001 0.0001" {CONTACT_SOL} />\n'
               '    </contact>')
    out, n1 = re.subn(r'<body name="cylinder".*?</body>', cyl, src, flags=re.S)
    out, n2 = re.subn(r'<body name="plank".*?</body>', plank, out, flags=re.S)
    out, n3 = re.subn(r'<contact>.*?</contact>', contact, out, flags=re.S)
    if n1 != 1 or n2 != 1 or n3 != 1:
        sys.exit(f"{TEMPLATE}: expected one cylinder body, one plank body and one contact block, found {n1}, {n2}, {n3}")
    key = ("0 0 0.12 1 0 0 0 " + "0 " * 14 + ("" if fixed_roller else f"0 0 {zc:.4f} 1 0 0 0 ") + f"0 0 {zp:.4f} 1 0 0 0").strip()
    out = re.sub(r'<key name="INIT" qpos="[^"]*"', f'<key name="INIT" qpos="{key}"', out)
    tag = (f"{orientation}_r{radius * 1000:.0f}_t{plank_thk * 1000:.0f}_l{plank_len * 1000:.0f}"
           + ("" if roller == "cylinder" else f"_{roller}") + ("_fixed" if fixed_roller else ""))
    out = out.replace('<mujoco model="scene_board">',
                      f'<mujoco model="scene_board_{tag}">\n'
                      f'    <!-- GENERATED by src/microduck_lab/tasks/balance_board/board.py from scene_board.xml; do not edit. -->', 1)
    path = os.path.join(ROBOT_DIR, f"scene_board_{tag}.xml")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)
    return path


def tilt_angles(quat):
    """(roll, pitch) of a body from its quaternion, the projected-gravity way:
    g = R(q)^T [0,0,-1]; roll = atan2(g_y, -g_z), pitch = atan2(g_x, -g_z).
    roll > 0: the body's +y (left) side is down / it leans left.
    pitch > 0: its +x (front) side is down / nose down."""
    neg = np.zeros(4)
    g = np.zeros(3)
    mujoco.mju_negQuat(neg, np.asarray(quat, dtype=np.float64))
    mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), neg)
    return float(np.arctan2(g[1], -g[2])), float(np.arctan2(g[0], -g[2]))


class Board:
    """The scene, the duck on the plank, and everything a controller may read."""

    def __init__(self, orientation, radius=CYL_RADIUS, plank_len=PLANK_LEN, plank_thk=PLANK_THK,
                 cyl_mass=CYL_MASS, plank_mass=PLANK_MASS, friction=FRICTION, fixed_roller=False):
        self.orientation = orientation
        self.axis = AXIS[orientation]
        self.radius, self.plank_len, self.plank_thk = radius, plank_len, plank_thk
        self.fixed_roller = fixed_roller
        self.xml = write_scene(orientation, radius=radius, plank_len=plank_len, plank_thk=plank_thk,
                               cyl_mass=cyl_mass, plank_mass=plank_mass, friction=friction,
                               fixed_roller=fixed_roller)
        model, data = duck_sim.load_scene(self.xml)
        with contextlib.redirect_stdout(io.StringIO()):
            self.policy, self.adr = duck_sim.make_policy(model, data, walking_onnx_path=ANY_ONNX)
        self.model, self.data = model, data
        m = model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.padr = int(m.jnt_qposadr[jid("plank_freejoint")])
        self.pvadr = int(m.jnt_dofadr[jid("plank_freejoint")])
        if fixed_roller:      # no roller joint: alias its "qpos" to a scratch copy
            self._cyl_q = np.array([0, 0, radius, 1, 0, 0, 0], dtype=np.float64)
            self.cadr = self.cvadr = None
        else:
            self.cadr = int(m.jnt_qposadr[jid("cylinder_freejoint")])
            self.cvadr = int(m.jnt_dofadr[jid("cylinder_freejoint")])
        self.vadr = int(m.jnt_dofadr[jid("trunk_base_freejoint")])
        self.floor, self.plank_g, self.cyl_g = gid("floor"), gid("plank"), gid("cylinder")
        self.feet = {gid("left_foot_collision"): "left", gid("right_foot_collision"): "right",
                     gid("left_foot_box"): "left", gid("right_foot_box"): "right"}
        self.trunk = bid("trunk_base")
        self.plank_b, self.cyl_b = bid("plank"), bid("cylinder")
        passive = {0, self.plank_b, self.cyl_b}
        self.robot_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in passive}
        self.robot_bodies = [b for b in range(m.nbody) if b not in passive]
        self.qi = list(self.policy.joint_qpos_indices)
        self.held_q = None

    # ---- setup -------------------------------------------------------------
    def _freeze_duck(self, q, steps):
        """Step the passive bodies with the duck parked at `q` (out of contact)."""
        d = self.data
        for _ in range(steps):
            d.qpos[self.adr:self.adr + 7] = q
            d.qvel[self.vadr:self.vadr + 6] = 0.0
            mujoco.mj_step(self.model, d)
        d.qpos[self.adr:self.adr + 7] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def robot_contacts(self):
        d = self.data
        return sum(1 for k in range(d.ncon)
                   if d.contact[k].geom1 in self.robot_geoms or d.contact[k].geom2 in self.robot_geoms)

    def ground(self, clearance=0.001, lo=0.05, hi=0.40):
        """Lowest trunk height with no robot contact (harness `_ground` pattern,
        but the plank/cylinder contacts are ignored), plus a small clearance."""
        d = self.data
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            d.qpos[self.adr + 2] = mid
            mujoco.mj_forward(self.model, d)
            if self.robot_contacts() > 0:
                lo = mid
            else:
                hi = mid
        d.qpos[self.adr + 2] = hi + clearance
        mujoco.mj_forward(self.model, d)
        return hi

    def com(self, include_plank=True):
        m, d = self.model, self.data
        bodies = self.robot_bodies + ([self.plank_b] if include_plank else [])
        w = m.body_mass[bodies]
        return (w[:, None] * d.xipos[bodies]).sum(axis=0) / w.sum()

    def place(self, pose=HOME):
        """Settle the passive bodies, then rest the duck on the plank with the
        combined CoM (duck + plank) over the cylinder axis."""
        m, d = self.model, self.data
        if not self.fixed_roller:
            d.qpos[self.cadr:self.cadr + 7] = [0, 0, self.radius, 1, 0, 0, 0]
        d.qpos[self.padr:self.padr + 7] = [0, 0, 2 * self.radius + self.plank_thk / 2, 1, 0, 0, 0]
        for i, qi in enumerate(self.qi):
            d.qpos[qi] = pose[i]
        d.ctrl[:14] = pose
        parked = np.array([0, 0, 1.0, 1, 0, 0, 0], dtype=np.float64)
        self._freeze_duck(parked, 300)                 # 1.5 s: plank on cylinder on floor
        cyl_q = self._cyl_q if self.fixed_roller else d.qpos[self.cadr:self.cadr + 7].copy()
        self.held_q = (cyl_q, d.qpos[self.padr:self.padr + 7].copy())
        cyl = cyl_q[:3].copy()
        # put the trunk over the cylinder, then move it so the CoM is over the axis
        d.qpos[self.adr + 0:self.adr + 2] = cyl[:2]
        d.qpos[self.adr + 3:self.adr + 7] = [1, 0, 0, 0]
        for _ in range(3):
            self.ground()
            off = self.com()[:2] - cyl[:2]
            d.qpos[self.adr + 0:self.adr + 2] -= off
        z_rest = self.ground()
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        return z_rest

    def set_held(self, held):
        """While held, the cylinder and plank are made 1e4 x heavier so the
        duck's weight cannot move them within a physics step (MuJoCo's contact
        softness is mass-scaled, so they do not sink either). Restoring their
        positions alone is not enough: a 70 g plank accelerates away under a
        7 N load inside one step and the duck sinks through it."""
        m = self.model
        if not hasattr(self, "_mass0"):
            self._mass0 = (m.body_mass[[self.cyl_b, self.plank_b]].copy(),
                           m.body_inertia[[self.cyl_b, self.plank_b]].copy())
        f = HOLD_MASS_SCALE if held else 1.0
        m.body_mass[[self.cyl_b, self.plank_b]] = self._mass0[0] * f
        m.body_inertia[[self.cyl_b, self.plank_b]] = self._mass0[1] * f
        # invweight0 (used to regularise contacts) is compiled from the mass;
        # without this the heavy plank gets a 70 g contact and falls through.
        # mj_setConst overwrites the qpos of the data it is given, so use a scratch.
        mujoco.mj_setConst(m, mujoco.MjData(m))

    def hold_board(self):
        """Pin the cylinder and plank where they settled (used while settling)."""
        d = self.data
        d.qpos[self.padr:self.padr + 7] = self.held_q[1]
        d.qvel[self.pvadr:self.pvadr + 6] = 0.0
        if not self.fixed_roller:
            d.qpos[self.cadr:self.cadr + 7] = self.held_q[0]
            d.qvel[self.cvadr:self.cvadr + 6] = 0.0

    # ---- observation -------------------------------------------------------
    def state(self):
        m, d = self.model, self.data
        troll, tpitch = tilt_angles(d.qpos[self.adr + 3:self.adr + 7])
        proll, ppitch = tilt_angles(d.qpos[self.padr + 3:self.padr + 7])
        R = d.xmat[self.plank_b].reshape(3, 3)
        long_axis = R[:, self.axis]                       # plank's long axis, world
        rel = d.xpos[self.cyl_b] - d.xpos[self.plank_b]
        s = float(rel @ long_axis)                        # cylinder offset along the plank
        return {
            "trunk_tilt": troll if self.orientation == "A" else tpitch,
            "trunk_other": tpitch if self.orientation == "A" else troll,
            "plank_tilt": proll if self.orientation == "A" else ppitch,
            "plank_other": ppitch if self.orientation == "A" else proll,
            "s": s,
            "cyl_pos": d.xpos[self.cyl_b].copy(),
            "plank_pos": d.xpos[self.plank_b].copy(),
            "trunk_pos": d.xpos[self.trunk].copy(),
            "com": self.com(),
            "q": np.array([d.qpos[i] for i in self.qi]),
        }

    def contacts(self):
        """Classify this step's contacts: plank-floor, per-foot plank contacts,
        and any non-foot robot geom touching floor/plank/cylinder."""
        d = self.data
        plank_floor = 0
        foot_plank = {"left": 0, "right": 0}
        foot_floor = {"left": 0, "right": 0}
        bad = None
        for k in range(d.ncon):
            g1, g2 = d.contact[k].geom1, d.contact[k].geom2
            pair = {g1, g2}
            if pair == {self.plank_g, self.floor}:
                plank_floor += 1
                continue
            for g in (g1, g2):
                if g in self.feet:
                    other = g2 if g == g1 else g1
                    if other == self.plank_g:
                        foot_plank[self.feet[g]] += 1
                    elif other == self.floor:
                        foot_floor[self.feet[g]] += 1
                elif g in self.robot_geoms and bad is None:
                    other = g2 if g == g1 else g1
                    if other in (self.floor, self.plank_g, self.cyl_g):
                        bad = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])),
                               mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other))
        return plank_floor, foot_plank, foot_floor, bad


# ---- controllers ------------------------------------------------------------
class OpenLoop:
    name = "open"

    def __call__(self, s, dt):
        return HOME.copy(), 0.0


class PD:
    """Hold HOME, keep the trunk upright, and add board feedback.

    Upright (both axes, always on; the bare servos cannot hold HOME even on
    the floor, measured: falls at ~0.8 s):
        pitch: ankles (L +, R -)   <- kp_s * trunk_pitch + kd_s * d(trunk_pitch)/dt
        roll:  hip_rolls (both -)  <- kp_s * trunk_roll  + kd_s * d(trunk_roll)/dt
    Board (the tilt axis only):
        u = kp_b*tilt + kd_b*d(tilt)/dt - ks*s - kv*ds/dt
    tilt = plank tilt (rad, + = +axis side down), s = cylinder offset from
    the plank centre along the plank (m, + = toward +axis). u > 0 always
    means "move the CoM toward the -axis". kp_t/kd_t add to the stand gains
    on the tilt axis. With every board gain 0 this is the "upright only, no
    board feedback" baseline. `stand` = {"roll": (kp, kd), "pitch": (kp, kd)}."""
    name = "pd"

    def __init__(self, orientation, kp_b=0.0, kd_b=0.0, kp_t=0.0, kd_t=0.0, ks=0.0, kv=0.0,
                 stand=None, u_max=None):
        self.o = orientation
        self.g = dict(kp_b=kp_b, kd_b=kd_b, kp_t=kp_t, kd_t=kd_t, ks=ks, kv=kv)
        self.stand = dict(STAND) if stand is None else dict(stand)
        self.u_max = u_max if u_max is not None else (HIP_ROLL_LIMIT - abs(HOME[L_HIP_ROLL]) if orientation == "A" else 1.0)
        self.prev = None
        self.name = "pd" if any(v != 0 for v in self.g.values()) else "upright"

    def __call__(self, s, dt):
        g = self.g
        if self.prev is None:
            self.prev = (s["plank_tilt"], s["trunk_tilt"], s["s"], s["trunk_other"])
        d_tilt = (s["plank_tilt"] - self.prev[0]) / dt
        d_trunk = (s["trunk_tilt"] - self.prev[1]) / dt
        d_s = (s["s"] - self.prev[2]) / dt
        d_other = (s["trunk_other"] - self.prev[3]) / dt
        self.prev = (s["plank_tilt"], s["trunk_tilt"], s["s"], s["trunk_other"])
        tilt_axis, other_axis = ("roll", "pitch") if self.o == "A" else ("pitch", "roll")
        kp_s, kd_s = self.stand[tilt_axis]
        kp_o, kd_o = self.stand[other_axis]
        if not s.get("released", True):
            # board still held (static): plain stand PD, no board terms. A
            # board law with negative trunk gain would fall off a fixed floor.
            g = dict(kp_b=0.0, kd_b=0.0, kp_t=0.0, kd_t=0.0, ks=0.0, kv=0.0)
        # tilt axis: stand PD on the trunk + board terms
        u = ((kp_s + g["kp_t"]) * s["trunk_tilt"] + (kd_s + g["kd_t"]) * d_trunk
             + g["kp_b"] * s["plank_tilt"] + g["kd_b"] * d_tilt
             - g["ks"] * s["s"] - g["kv"] * d_s)
        u = float(np.clip(u, -self.u_max, self.u_max))
        # other axis: stand PD only
        v = kp_o * s["trunk_other"] + kd_o * d_other
        c = HOME.copy()
        if self.o == "A":
            c[L_HIP_ROLL] -= u
            c[R_HIP_ROLL] -= u
            c[L_ANKLE] += v
            c[R_ANKLE] -= v
        else:
            c[L_ANKLE] += u
            c[R_ANKLE] -= u
            c[L_HIP_ROLL] -= v
            c[R_HIP_ROLL] -= v
        return c, u


# ---- one run ---------------------------------------------------------------
def run(orientation, controller, seconds=10.0, push=0.0, push_at=2.0, radius=CYL_RADIUS,
        plank_len=PLANK_LEN, plank_thk=PLANK_THK, fixed_roller=False, video=None, azimuth=None,
        cam_distance=0.8, elevation=-12.0, fps=30, log_every=0.0, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    b = Board(orientation, radius=radius, plank_len=plank_len, plank_thk=plank_thk, fixed_roller=fixed_roller)
    m, d = b.model, b.data
    z_rest = b.place()
    s0 = b.state()
    tilt_max = np.degrees(np.arcsin(min(1.0, 2 * radius / (plank_len / 2))))
    say(f"board {orientation}: {'FIXED' if fixed_roller else 'free'} roller r={radius * 1000:.0f} mm, plank {plank_len * 1000:.0f} x "
        f"{PLANK_WID * 1000:.0f} x {plank_thk * 1000:.0f} mm; plank top at z={2 * radius + plank_thk:.3f} m; "
        f"an end touches the floor at about {tilt_max:.1f} deg")
    say(f"duck: trunk rests at z={z_rest:.4f} m ({(z_rest - 2 * radius - plank_thk) * 1000:.1f} mm above the plank top); "
        f"CoM {1000 * (s0['com'][b.axis] - s0['cyl_pos'][b.axis]):+.1f} mm from the cylinder axis, "
        f"{1000 * (s0['com'][2] - 2 * radius - plank_thk):.0f} mm above the plank top; "
        f"cylinder offset on plank s={1000 * s0['s']:+.1f} mm")

    if azimuth is None:
        azimuth = 180.0 if orientation == "A" else 270.0
    rec = duck_sim.Recorder(m, fps=fps, distance=cam_distance, azimuth=azimuth, elevation=elevation) if video else None

    res = {"orientation": orientation, "controller": controller.name, "held": 0.0, "fail": None,
           "fail_t": None, "max_tilt": 0.0, "max_trunk": 0.0, "max_s": 0.0, "max_u": 0.0,
           "push": push, "push_survived": None, "seconds": seconds, "radius": radius,
           "plank_len": plank_len, "plank_thk": plank_thk, "fixed_roller": fixed_roller}
    settle_steps = int(round(SETTLE_S / CONTROL_DT))
    steps = int(round(seconds / CONTROL_DT))
    no_contact = {"left": 0, "right": 0}
    grace = int(round(FOOT_GRACE_S / CONTROL_DT))
    pushed = False
    stop_at = None
    last_log = -1e9
    frame = 0

    b.set_held(True)
    for i in range(-settle_steps, steps):
        t = i * CONTROL_DT
        held_phase = i < 0
        if i == 0:
            b.set_held(False)
        s = b.state()
        s["released"] = not held_phase
        ctrl, u = controller(s, CONTROL_DT)
        d.ctrl[:14] = ctrl
        if push and not pushed and t >= push_at and res["fail"] is None:
            d.qvel[b.vadr + b.axis] += push
            pushed = True
            say(f"t={t:4.2f}s  PUSH {push:+.2f} m/s on the trunk along {'xyz'[b.axis]}")
        for _ in range(DECIMATION):
            if held_phase:
                b.hold_board()
            mujoco.mj_step(m, d)
        if held_phase:
            b.hold_board()
            mujoco.mj_forward(m, d)
        if rec:
            rec.maybe_capture(frame, d)
            frame += 1
        if held_phase:
            continue

        plank_floor, foot_plank, foot_floor, bad = b.contacts()
        for side in ("left", "right"):
            no_contact[side] = 0 if foot_plank[side] > 0 else no_contact[side] + 1
        trunk_above = float(d.xpos[b.trunk][2] - (d.xpos[b.plank_b][2] + plank_thk / 2))
        tilt = float(s["plank_tilt"])
        if res["fail"] is None:
            fail = None
            if plank_floor > 0:
                side = "+" if tilt > 0 else "-"
                fail = (f"plank end touched the floor on the {side}{'xyz'[b.axis]} side "
                        f"(tilt {np.degrees(tilt):+.1f} deg, cylinder offset s={1000 * s['s']:+.0f} mm)")
            elif any(no_contact[k] > grace for k in no_contact):
                foot = [k for k in no_contact if no_contact[k] > grace][0]
                fail = f"{foot} foot left the plank (tilt {np.degrees(tilt):+.1f} deg)"
            elif bad is not None:
                fail = f"robot body '{bad[0]}' touched '{bad[1]}' (tilt {np.degrees(tilt):+.1f} deg)"
            elif trunk_above < FALL_HEIGHT:
                fail = f"trunk dropped to {trunk_above * 1000:.0f} mm above the plank"
            if fail is None:
                res["held"] = t + CONTROL_DT
                res["max_tilt"] = max(res["max_tilt"], abs(tilt))
                res["max_trunk"] = max(res["max_trunk"], abs(float(s["trunk_tilt"])))
                res["max_s"] = max(res["max_s"], abs(float(s["s"])))
                res["max_u"] = max(res["max_u"], abs(u))
            else:
                res["fail"], res["fail_t"] = fail, t
                say(f"t={t:4.2f}s  FAILED: {fail}")
                stop_at = t + AFTER_FAIL_S
                if not rec:
                    break
        if log_every and t - last_log >= log_every - 1e-9 and res["fail"] is None:
            last_log = t
            say(f"t={t:4.2f}s  plank {np.degrees(tilt):+6.2f} deg  trunk {np.degrees(s['trunk_tilt']):+6.2f} deg  "
                f"s={1000 * s['s']:+5.1f} mm  u={u:+.3f}  contacts L={foot_plank['left']} R={foot_plank['right']}  "
                f"cyl {'xyz'[b.axis]}={1000 * s['cyl_pos'][b.axis]:+.0f} mm")
        if stop_at is not None and t >= stop_at:
            break

    res["held"] = min(res["held"], seconds)
    if push:
        res["push_survived"] = res["fail"] is None
    say("")
    say(f"result:  {orientation} / {controller.name}  held {res['held']:.2f} s of {seconds:.0f}"
        + ("" if res["fail"] is None else f"  ({res['fail']})"))
    say(f"  max plank tilt while held: {np.degrees(res['max_tilt']):.1f} deg; max trunk lean {np.degrees(res['max_trunk']):.1f} deg; "
        f"cylinder wandered up to {1000 * res['max_s']:.0f} mm along the plank; max |u| {res['max_u']:.2f} rad")
    if push:
        say(f"  push {push:+.2f} m/s at t={push_at:.1f} s: {'SURVIVED' if res['push_survived'] else 'NOT survived'}")
    if rec:
        n = rec.write(video)
        say(f"  video: {video} ({n} frames)")
    return res


# ---- diagnostics -----------------------------------------------------------
def measure_levers(orientation="A", delta=0.15, hold_s=1.0):
    """Trunk lean and CoM shift per rad of the balance joints, both feet
    planted on the HELD board. The upright PD is switched OFF on the measured
    axis (kept on the other one) so the loop cannot cancel the offset: settle
    1 s, add the offset for `hold_s`, read the change."""
    out = {}
    for name, axis in (("hip_roll_both", "roll"), ("ankle_mirrored", "pitch")):
        b = Board(orientation)
        b.place()
        b.set_held(True)
        stand = dict(STAND)
        stand[axis] = (0.0, 0.0)
        ctrl = PD(orientation, stand=stand)
        key = "trunk_tilt" if (axis == "roll") == (orientation == "A") else "trunk_other"
        com0 = lean0 = None
        n0, n1 = 200, 200 + int(hold_s / CONTROL_DT)
        for i in range(n1):
            st = b.state()
            c, _ = ctrl(st, CONTROL_DT)
            if i == n0:
                com0, lean0 = b.com(include_plank=False).copy(), st[key]
            if i >= n0:
                if name == "hip_roll_both":
                    c[L_HIP_ROLL] += delta
                    c[R_HIP_ROLL] += delta
                else:
                    c[L_ANKLE] += delta
                    c[R_ANKLE] -= delta
            b.data.ctrl[:14] = c
            for _ in range(DECIMATION):
                b.hold_board()
                mujoco.mj_step(b.model, b.data)
        st = b.state()
        shift = b.com(include_plank=False) - com0
        lean = st[key] - lean0
        out[name] = (shift, lean)
        ax = "y" if axis == "roll" else "x"
        print(f"{name}: +{delta:.2f} rad for {hold_s:.1f} s, upright PD off on {axis} -> trunk {axis} "
              f"{np.degrees(lean):+.1f} deg ({lean / delta:+.2f} rad/rad), CoM {ax} {1000 * shift[AXIS['A'] if axis == 'roll' else AXIS['B']]:+.1f} mm "
              f"({1000 * shift[AXIS['A'] if axis == 'roll' else AXIS['B']] / delta:+.0f} mm/rad); z {1000 * shift[2]:+.1f} mm")
    return out


def stand_grid(seconds=5.0):
    """Upright-PD gains that keep the duck standing on the HELD (static) board,
    one axis at a time (the other axis keeps STAND). Prints the worst lean on
    the swept axis and the lean it settles to."""
    for axis in ("pitch", "roll"):
        for kp in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0):
            for kd in (0.0, 0.1, 0.3):
                b = Board("A")
                b.place()
                b.set_held(True)
                stand = dict(STAND)
                stand[axis] = (kp, kd)
                ctrl = PD("A", stand=stand)
                key = "trunk_tilt" if axis == "roll" else "trunk_other"
                lean = []
                for i in range(int(seconds / CONTROL_DT)):
                    st = b.state()
                    c, _ = ctrl(st, CONTROL_DT)
                    b.data.ctrl[:14] = c
                    for _ in range(DECIMATION):
                        b.hold_board()
                        mujoco.mj_step(b.model, b.data)
                    lean.append(st[key])
                    if abs(st[key]) > np.radians(30):
                        break
                lean = np.degrees(np.array(lean))
                print(f"  stand {axis} kp={kp:g} kd={kd:g}: worst {abs(lean).max():5.1f} deg, "
                      f"last second {lean[-50:].min():+.1f}..{lean[-50:].max():+.1f} deg"
                      + ("" if len(lean) == int(seconds / CONTROL_DT) else f"  [stopped at {len(lean) * CONTROL_DT:.2f} s]"), flush=True)


def grid(orientation, seconds=10.0, push=0.15, **kw):
    """Coarse gain search in three stages (each keeps the best of the last):
    1. kp_t (added to the stand gain on the tilt axis; negative = less trunk
       feedback), kp_b, kd_b;  2. kd_t;  3. ks, kv (both signs: the linear
       model says the CoM may have to move AWAY from the roller to steer it).
    Score = seconds held, then smaller max tilt. Prints every row."""
    stage1 = dict(kp_t=[-4, -3, -2, 0], kp_b=[1, 2, 4, 6, 8], kd_b=[0.1, 0.3, 0.6, 1.0])
    stage2 = dict(kd_t=[-0.2, 0, 0.3, 0.6])
    stage3 = dict(ks=[-40, -20, -10, 0, 10, 20], kv=[-2, -1, 0, 1, 2])
    best = dict(kp_b=0.0, kd_b=0.0, kp_t=0.0, kd_t=0.0, ks=0.0, kv=0.0)
    best_score = (-1.0, 0.0)
    rows = []

    for stage in (stage1, stage2, stage3):
        keys = list(stage)
        stage_best, stage_score = None, best_score
        for vals in itertools.product(*stage.values()):
            g = dict(best)
            g.update(dict(zip(keys, vals)))
            r = run(orientation, PD(orientation, **g), seconds=seconds, verbose=False, **kw)
            sc = (r["held"], -r["max_tilt"])
            rows.append((dict(g), r))
            print(f"  {orientation} " + " ".join(f"{k}={v:g}" for k, v in g.items())
                  + f" -> held {r['held']:.2f} s, max tilt {np.degrees(r['max_tilt']):.1f} deg"
                  + ("" if r["fail"] is None else f"  [{r['fail'].split(' (')[0]}]"), flush=True)
            if sc > stage_score:
                stage_best, stage_score = g, sc
        if stage_best is not None:
            best, best_score = stage_best, stage_score
        print(f"  stage best: {best} -> held {best_score[0]:.2f} s, max tilt {np.degrees(-best_score[1]):.1f} deg", flush=True)
    return best, best_score, rows


def compare(gains, seconds=10.0, push=0.15, **kw):
    """The report table: orientation x controller, with and without a push."""
    rows = []
    for o in ("A", "B"):
        for ctrl_name in ("open", "upright", "pd"):
            for p in (0.0, push):
                c = OpenLoop() if ctrl_name == "open" else (PD(o) if ctrl_name == "upright" else PD(o, **gains[o]))
                r = run(o, c, seconds=seconds, push=p, verbose=False, **kw)
                rows.append(r)
                print(f"  {o} {ctrl_name:4s} push {p:.2f}: held {r['held']:.2f} s, max tilt {np.degrees(r['max_tilt']):.1f} deg"
                      + ("" if r["fail"] is None else f"  [{r['fail']}]"), flush=True)
    print()
    print("| orientation | controller | push (m/s) | seconds held (of %.0f) | max plank tilt (deg) | push survived | how it ended |" % seconds)
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        how = "held to the end" if r["fail"] is None else r["fail"]
        ps = "-" if not r["push"] else ("yes" if r["push_survived"] else "no")
        print(f"| {r['orientation']} | {r['controller']} | {r['push']:.2f} | {r['held']:.2f} | "
              f"{np.degrees(r['max_tilt']):.1f} | {ps} | {how} |")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orientation", choices=["A", "B"], default="A")
    p.add_argument("--controller", choices=["open", "pd"], default="pd")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--push", type=float, default=0.0, help="velocity impulse on the trunk (m/s) at --push-at, along the tilt axis")
    p.add_argument("--push-at", type=float, default=2.0)
    p.add_argument("--kp-b", type=float, default=None, help="rad of joint per rad of plank tilt")
    p.add_argument("--kd-b", type=float, default=None, help="rad per rad/s of plank tilt rate")
    p.add_argument("--kp-t", type=float, default=None, help="rad per rad of trunk lean")
    p.add_argument("--kd-t", type=float, default=None, help="rad per rad/s of trunk lean rate")
    p.add_argument("--ks", type=float, default=None, help="rad per m of cylinder offset along the plank")
    p.add_argument("--kv", type=float, default=None, help="rad per m/s of cylinder offset rate")
    p.add_argument("--radius", type=float, default=CYL_RADIUS)
    p.add_argument("--plank-len", type=float, default=PLANK_LEN)
    p.add_argument("--plank-thk", type=float, default=PLANK_THK)
    p.add_argument("--fixed-roller", action="store_true",
                   help="roller fixed to the floor (a wobble board) instead of free (rola bola); diagnostic")
    p.add_argument("--video", default=None)
    p.add_argument("--azimuth", type=float, default=None, help="camera azimuth; default 180 (behind) for A, 270 (right side) for B")
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--cam-distance", type=float, default=0.8)
    p.add_argument("--log-every", type=float, default=0.5)
    p.add_argument("--levers", action="store_true", help="measure the CoM lever of the balance joints and exit")
    p.add_argument("--stand-grid", action="store_true", help="grid the upright PD gains on the held board and exit")
    p.add_argument("--grid", choices=["A", "B"], default=None, help="run the gain search for one orientation")
    p.add_argument("--compare", action="store_true", help="print the orientation x controller table with the built-in best gains")
    args = p.parse_args()

    if args.levers:
        measure_levers(args.orientation)
        return
    if args.stand_grid:
        stand_grid(seconds=args.seconds)
        return
    geo = dict(radius=args.radius, plank_len=args.plank_len, plank_thk=args.plank_thk, fixed_roller=args.fixed_roller)
    if args.grid:
        best, sc, _ = grid(args.grid, seconds=args.seconds, **geo)
        print(f"\nbest gains for {args.grid}: {best} (held {sc[0]:.2f} s)")
        r = run(args.grid, PD(args.grid, **best), seconds=args.seconds, push=0.15, verbose=False, **geo)
        print(f"with a 0.15 m/s push: held {r['held']:.2f} s, {'survived' if r['push_survived'] else 'not survived'}")
        return
    gains = {o: dict(BEST_GAINS[o]) for o in BEST_GAINS}
    for k in ("kp_b", "kd_b", "kp_t", "kd_t", "ks", "kv"):
        v = getattr(args, k)
        if v is not None:
            gains[args.orientation][k] = v
    if args.compare:
        compare(gains, seconds=args.seconds, push=0.15, **geo)
        return
    ctrl = OpenLoop() if args.controller == "open" else PD(args.orientation, **gains[args.orientation])
    if args.controller == "pd":
        print(f"gains: {gains[args.orientation]}")
    run(args.orientation, ctrl, seconds=args.seconds, push=args.push, push_at=args.push_at,
        video=args.video, azimuth=args.azimuth, cam_distance=args.cam_distance, elevation=args.elevation,
        log_every=args.log_every, **geo)


# Best of the grid searches on the FREE roller, r = 30 mm (see the doc). Neither
# holds: A 1.98 s, B ~1.0 s. Negative kp_t/kd_t mean LESS trunk feedback than
# the stand PD (the board is a cart-pole; see docs/tasks/balance_board.md).
BEST_GAINS = {
    "A": dict(kp_b=0.0, kd_b=0.0, kp_t=-4.0, kd_t=-0.7, ks=0.0, kv=0.0),
    "B": dict(kp_b=0.0, kd_b=0.0, kp_t=-5.0, kd_t=-0.3, ks=-10.0, kv=0.0),
}


if __name__ == "__main__":
    main()
