"""Trampoline world: one Microduck on a spring bed, BAM XL330 motors.

The bed is a rigid plate on a vertical slide joint with a linear spring and a
damper. Two numbers set it, and both can be measured on a real bed:

- `sag_mm`: how far the bed sinks when the robot stands still on it. This
  sets the spring;
- `rebound`: drop a rigid weight of the robot's mass from `rebound_drop_mm`
  onto the bed; `rebound` = its first bounce height / drop height. The
  damper is fitted to this in a small 1-D sim (`fit_damping`).

The rebound is not set by the damper alone. The weight also loses energy
when it hits the 30 g bed, and gravity keeps it on the bed for more than
half a spring cycle. So the textbook restitution formula does not fit here.

`rigid=True` locks the bed: the same bed surface and contact, but no spring.
It stands in for a hard floor.

The plate carries no weight of its own (gravcomp), so at rest it sits at
BED_TOP. Only motor targets move the robot, as in the jump world: `step`
refuses to run if anything writes an external force.

The feet touch the bed through the flat foot boxes of
robot_allcollisions_boardfeet.xml (box on box). A mesh sole on a box rolls
forward, as the balance-board task measured. The sole meshes still touch
the floor.

The physics step is 1 ms here, not the usual 5 ms (the control rate is
still 50 Hz). There are two reasons. First, the stiff foot-bed contact (see
CONTACT) needs a step of 1 ms or less. Second, measured on a 20 mm drop with
a 30 g bed and the earlier, softer contact (solref 0.01):

    step   bed jumps   result
    5 ms   15          feet pass through the bed, shins land on it
    2 ms    1          stands, but one bad step
    1 ms    0          stands

At 5 ms the foot-bed contact is lost for one step when its depth is near
zero. The light bed, pushed by about 12 N of spring, then moves up 7 mm in
that one step and passes through the foot. `bounce_report` gives the
deepest foot-box corner under the bed top (`max_foot_depth_mm`), so a bad
run shows in the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.tasks.jump.world import JumpWorld

G = 9.81
ROBOT_HEIGHT = 0.272   # standing, sole to top of head (measured in this scene)


@dataclass(frozen=True)
class Frame:
    """Trampoline size. All in metres."""
    name: str
    top: float          # bed top at rest, above the floor
    half: float         # collision box half size
    radius: float       # visual bed disc
    ring: float         # frame ring
    range: tuple        # slide limits (the bed bottoms out at range[0])
    ruler: bool = False  # a pole with a mark at every robot height above the bed


# The first bed: 20 mm sag under the duck, hops of a few cm.
SMALL = Frame("small", top=0.10, half=0.13, radius=0.17, ring=0.22, range=(-0.08, 0.05))
# For bounces of several robot heights: a deep bed (0.6 m of travel) and a
# wide one (the duck drifts more in a long flight).
BIG = Frame("big", top=0.65, half=0.30, radius=0.34, ring=0.40, range=(-0.60, 0.10), ruler=True)
# For flips: a flip turns about the feet on the bed, so it travels forward
# (about 0.1 m/s per rad/s of spin; 0.45 m for a full flip). Twice as wide.
WIDE = Frame("wide", top=0.65, half=0.60, radius=0.64, ring=0.70, range=(-0.60, 0.10), ruler=True)
# A check only: is the flip's landing good once there is room to travel after it?
HUGE = Frame("huge", top=0.65, half=1.00, radius=1.04, ring=1.10, range=(-0.60, 0.10), ruler=True)
FRAMES = {f.name: f for f in (SMALL, BIG, WIDE, HUGE)}

# Kept for the small bed's callers.
BED_TOP, BED_HALF, BED_RADIUS, RING_RADIUS, BED_RANGE = SMALL.top, SMALL.half, SMALL.radius, SMALL.ring, SMALL.range
BED_THICK = 0.015    # collision box half thickness, m: thick, so a deep foot cannot pass through
BED_MASS = 0.03      # moving mass of the bed, kg
N_SPRINGS = 24

# Foot-bed contact. It must be stiff: a soft contact on top of the bed spring
# ADDS energy. A rigid weight of the duck's mass dropped 20 mm on the bed with
# no damper should come back to 92% (it loses 8% hitting the 30 g bed). Measured:
#     solref 0.02 -> 127%,  0.01 -> 102%,  0.002 -> 93%
# 0.002 is the stiffest MuJoCo allows at a 1 ms step (2 steps).
CONTACT = 'solref="0.002 1" solimp="0.99 0.999 0.001"'


def scene_path(frame=SMALL):
    return paths.model("scene_trampoline.xml" if frame is SMALL else f"scene_trampoline_{frame.name}.xml")


def scene_xml(frame=SMALL):
    """The scene as a string. `write_scene` puts it in models/."""
    f = frame
    ring, legs, sites, edge, springs, ruler = [], [], [], [], [], []
    for i in range(N_SPRINGS):
        a, b = 2 * np.pi * i / N_SPRINGS, 2 * np.pi * (i + 1) / N_SPRINGS
        ring.append(f'<geom type="capsule" fromto="{f.ring * np.cos(a):.4f} {f.ring * np.sin(a):.4f} {f.top} '
                    f'{f.ring * np.cos(b):.4f} {f.ring * np.sin(b):.4f} {f.top}" size="0.007" '
                    f'contype="0" conaffinity="0" rgba="0.10 0.35 0.70 1"/>')
        r = f.ring - 0.006
        sites.append(f'<site name="ring_{i}" pos="{r * np.cos(a):.4f} {r * np.sin(a):.4f} {f.top}" size="0.001"/>')
        r = f.radius - 0.002
        edge.append(f'<site name="edge_{i}" pos="{r * np.cos(a):.4f} {r * np.sin(a):.4f} 0" size="0.001"/>')
        springs.append(f'<spatial width="0.0012" rgba="0.75 0.75 0.78 1"><site site="ring_{i}"/><site site="edge_{i}"/></spatial>')
    for i in range(6):
        a = 2 * np.pi * (i + 0.5) / 6
        legs.append(f'<geom type="capsule" fromto="{f.ring * np.cos(a):.4f} {f.ring * np.sin(a):.4f} {f.top} '
                    f'{(f.ring + 0.03) * np.cos(a):.4f} {(f.ring + 0.03) * np.sin(a):.4f} 0.004" size="0.006" '
                    f'contype="0" conaffinity="0" rgba="0.10 0.35 0.70 1"/>')
    if f.ruler:
        # A pole behind the bed (seen from the side camera), with a band at
        # every robot height above the bed's rest level.
        x, y, top = 0.0, f.ring + 0.10, f.top + 4.3 * ROBOT_HEIGHT
        ruler.append(f'<geom type="cylinder" fromto="{x} {y} 0 {x} {y} {top:.3f}" size="0.006" '
                     f'contype="0" conaffinity="0" rgba="0.85 0.85 0.85 1"/>')
        for n in range(1, 5):
            z = f.top + n * ROBOT_HEIGHT
            ruler.append(f'<geom type="box" pos="{x} {y} {z:.3f}" size="0.05 0.008 0.004" '
                         f'contype="0" conaffinity="0" rgba="0.95 0.55 0.10 1"/>')
    nl = "\n        "
    return f'''<mujoco model="scene_trampoline">
    <!-- GENERATED by src/microduck_lab/tasks/trampoline/world.py (write_scene); do not edit.
         Stiffness and damping of bed_slide are set at load time from sag_mm and bounce. -->
    <include file="robot_allcollisions_boardfeet.xml" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />
        {nl.join(ring)}
        {nl.join(legs)}
        {nl.join(sites)}
        {nl.join(ruler)}
        <body name="bed" pos="0 0 {f.top}" gravcomp="1">
            <joint name="bed_slide" type="slide" axis="0 0 1" limited="true" range="{f.range[0]} {f.range[1]}" />
            <inertial pos="0 0 -0.004" mass="{BED_MASS}" diaginertia="2e-4 2e-4 4e-4" />
            <geom name="bed" type="box" size="{f.half} {f.half} {BED_THICK}" pos="0 0 -{BED_THICK}" group="3"
                friction="1 0.005 0.0001" {CONTACT} priority="1" />
            <geom name="bed_visual" type="cylinder" size="{f.radius} 0.002" pos="0 0 -0.002"
                contype="0" conaffinity="0" rgba="0.08 0.08 0.10 1" />
            {nl.join(edge)}
        </body>
    </worldbody>

    <tendon>
        {nl.join(springs)}
    </tendon>

    <contact>
        <exclude body1="world" body2="bed" />
        <exclude body1="ankle_left" body2="bed" />
        <exclude body1="ankle_right" body2="bed" />
        <pair geom1="left_foot_box" geom2="bed" friction="1 1 0.005 0.0001 0.0001" {CONTACT} />
        <pair geom1="right_foot_box" geom2="bed" friction="1 1 0.005 0.0001 0.0001" {CONTACT} />
    </contact>
</mujoco>
'''


def write_scene(frame=SMALL):
    """Write the frame's scene XML in models/ if it is missing or out of date."""
    path, xml = scene_path(frame), scene_xml(frame)
    try:
        with open(path) as fh:
            if fh.read() == xml:
                return path
    except FileNotFoundError:
        pass
    with open(path, "w") as fh:
        fh.write(xml)
    return path


def rigid_rebound(k, c, mass, bed_mass=BED_MASS, drop_mm=20.0, timestep=0.001):
    """First bounce height / drop height of a rigid weight dropped on the bed.

    Same bed as the scene (spring k, damper c, mass, box, contact), with a
    rigid box of `mass` instead of the robot.
    """
    xml = f'''<mujoco><option timestep="{timestep}"/><worldbody>
        <body name="bed" pos="0 0 0" gravcomp="1">
            <joint type="slide" axis="0 0 1" stiffness="{k}" damping="{c}"/>
            <inertial pos="0 0 -0.004" mass="{bed_mass}" diaginertia="2e-4 2e-4 4e-4"/>
            <geom type="box" size="{BED_HALF} {BED_HALF} {BED_THICK}" pos="0 0 -{BED_THICK}" {CONTACT} priority="1"/>
        </body>
        <body name="weight" pos="0 0 {drop_mm / 1000 + 0.01}">
            <joint type="slide" axis="0 0 1"/>
            <geom type="box" size="0.03 0.02 0.01" mass="{mass}"/>
        </body>
    </worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    # The first time in the air after landing that lasts 5 ms or more. Shorter
    # gaps are contact chatter during the impact, not a bounce.
    min_air = max(1, int(round(0.005 / timestep)))
    landed, air, peak = False, 0, 0.0
    for _ in range(int(3.0 / timestep)):
        mujoco.mj_step(m, d)
        bottom = d.xpos[2][2] - 0.01
        if d.ncon:
            if air >= min_air:
                break
            landed, air, peak = True, 0, 0.0
        elif landed:
            air += 1
            peak = max(peak, bottom)
    return (peak * 1000 / drop_mm) if air >= min_air else 0.0


def fit_damping(k, mass, rebound, bed_mass=BED_MASS, drop_mm=20.0, timestep=0.001):
    """The bed damper (N s/m) that gives `rigid_rebound` == rebound."""
    top = rigid_rebound(k, 0.0, mass, bed_mass, drop_mm, timestep)
    if not 0 < rebound < top:
        raise ValueError(f"rebound must be between 0 and {top:.2f} for this bed (no damper)")
    lo, hi = 0.0, 2 * np.sqrt(k * (mass + bed_mass))
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rigid_rebound(k, mid, mass, bed_mass, drop_mm, timestep) > rebound else (lo, mid)
    return (lo + hi) / 2


class TrampolineWorld(JumpWorld):
    def __init__(self, sag_mm=20.0, rebound=0.6, rebound_drop_mm=20.0, bed_mass=BED_MASS, rigid=False,
                 frame="small", timestep=0.001, decimation=20, **kw):
        self.frame = FRAMES[frame] if isinstance(frame, str) else frame
        super().__init__(scene=write_scene(self.frame), timestep=timestep, decimation=decimation, **kw)
        m = self.model
        self.bed_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bed")
        scale = bed_mass / m.body_mass[self.bed_body]
        m.body_mass[self.bed_body] = bed_mass
        m.body_inertia[self.bed_body] *= scale
        mujoco.mj_setConst(m, self.data)
        self.bed_mass = bed_mass
        self.bed_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bed")
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bed_slide")
        self.bed_q, self.bed_v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.foot_boxes = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_box")
                           for s in ("left", "right")]
        self.robot_geoms = set(np.flatnonzero(m.body_rootid[m.geom_bodyid] == self.trunk).tolist())
        self.rigid = rigid
        self.sag_mm, self.rebound = (0.0, 0.0) if rigid else (sag_mm, rebound)
        m.qpos_spring[self.bed_q] = 0.0
        if rigid:
            # Lock the bed at rest with a stiff limit (time constant 2 steps).
            self.k = self.c = self.bed_hz = 0.0
            m.jnt_range[j] = (0.0, 0.0)
            m.jnt_solref[j] = (2 * timestep, 1.0)
            return

        # Spring from the static sag, damper from the rebound, both with the
        # robot's mass: those are what a drop test on the real bed gives.
        self.k = self.mass * G / (sag_mm / 1000)
        # The empty bed is the fastest part of the scene. Its spring is
        # explicit, so the step must stay well inside one of its periods.
        if np.sqrt(self.k / bed_mass) * timestep > 0.5:
            raise ValueError(f"bed too stiff for a {timestep * 1000:g} ms step: use a smaller step "
                             f"or a larger sag_mm")
        self.c = fit_damping(self.k, self.mass, rebound, bed_mass, rebound_drop_mm, timestep)
        m.jnt_stiffness[j] = self.k
        m.dof_damping[self.bed_v] = self.c
        self.bed_hz = float(np.sqrt(self.k / (self.mass + bed_mass)) / (2 * np.pi))

    # ---- placing -----------------------------------------------------------
    def place(self, pose, drop_mm=0.0, settled_bed=None):
        """Put the robot upright in `pose`, lowest sole `drop_mm` above the bed.

        `settled_bed=True` starts the bed at its static sag, for a robot that
        starts standing on it. Default: sagged only when drop_mm == 0.
        """
        if settled_bed is None:
            settled_bed = drop_mm == 0
        d = self.data
        d.qpos[self.bed_q] = -self.sag_mm / 1000 if settled_bed else 0.0
        d.qpos[self.root + 3:self.root + 7] = [1, 0, 0, 0]
        self.set_pose(pose)
        gap = min(self.sole_heights())
        self.set_pose(pose, z=d.qpos[self.root + 2] - gap + drop_mm / 1000)

    # ---- measures ----------------------------------------------------------
    def bed_top(self):
        return float(self.data.xpos[self.bed_body][2])

    def touching(self):
        """Robot geoms touching anything that is not the robot: (feet, other).

        feet = the foot boxes on the bed. Anything else counts as other: a
        body part on the bed, or any part (a sole too) on the floor.
        """
        d = self.data
        feet, other = set(), set()
        for c in d.contact[:d.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            ra, rb = a in self.robot_geoms, b in self.robot_geoms
            if ra == rb:
                continue
            g, env = (a, b) if ra else (b, a)
            if g in self.foot_boxes and env == self.bed_geom:
                feet.add(g)
            else:
                other.add(g)
        return feet, other

    def sole_heights(self):
        """Lowest point of each sole mesh above the bed top (m): (left, right)."""
        top = self.bed_top()
        return tuple(h - top for h in super().sole_heights())

    def foot_box_gap(self):
        """Lowest corner of either foot box above the bed top (m). Negative = inside the bed."""
        d, m = self.data, self.model
        low = min(float((d.geom_xmat[g].reshape(3, 3) @ (CORNERS * m.geom_size[g]).T)[2].min() + d.geom_xpos[g][2])
                  for g in self.foot_boxes)
        return low - self.bed_top()

    def _record(self):
        super()._record()
        d = self.data
        self.log[-1] = self.log[-1] + (float(d.qpos[self.bed_q]), float(d.qvel[self.bed_v]), self.foot_box_gap())


CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


LOG_COLUMNS = ("t", "com_z", "com_vz", "n_feet", "n_other", "trunk_z", "tilt", "foot_z_min", "max_torque",
               "sole_gap", "bed_z", "bed_vz", "box_gap")
COL = {n: i for i, n in enumerate(LOG_COLUMNS)}


def bounce_report(log, t0=0.0, fall_tilt=45.0, min_flight_s=0.010):
    """Measure the bounces in a substep log (rows as in LOG_COLUMNS).

    A flight is a run of substeps where no robot part touches anything, at
    least `min_flight_s` long (shorter gaps are contact chatter). A fall is
    the first body contact, or tilt above `fall_tilt` degrees.
    """
    a = np.array(log)
    a = a[a[:, 0] >= t0]
    dt = a[1, 0] - a[0, 0]
    air = (a[:, COL["n_feet"]] == 0) & (a[:, COL["n_other"]] == 0)
    bad = (a[:, COL["n_other"]] > 0) | (a[:, COL["tilt"]] > fall_tilt)
    fall_i = int(np.argmax(bad)) if bad.any() else None
    end = len(a) if fall_i is None else fall_i

    flights, i = [], 0
    while i < end:
        if air[i]:
            j = i
            while j < end and air[j]:
                j += 1
            if (j - i) * dt >= min_flight_s:
                seg = a[i:j]
                flights.append({
                    "t": round(float(seg[0, 0]), 3),
                    "flight_ms": round((j - i) * dt * 1000, 1),
                    "sole_gap_mm": round(float(seg[:, COL["sole_gap"]].max() * 1000), 1),
                    "com_rise_mm": round(float((seg[:, COL["com_z"]].max() - seg[0, COL["com_z"]]) * 1000), 1),
                })
            i = j
        else:
            i += 1

    alive = a[:end]
    return {
        "survived_s": round(float(alive[-1, 0] - a[0, 0]) if len(alive) else 0.0, 3),
        "fell": fall_i is not None,
        "fall_reason": None if fall_i is None else (
            "body contact" if a[fall_i, COL["n_other"]] > 0 else f"tilt > {fall_tilt:g} deg"),
        "bounces": len(flights),
        "max_bed_sag_mm": round(float(-alive[:, COL["bed_z"]].min() * 1000), 1) if len(alive) else 0.0,
        "max_tilt_deg": round(float(alive[:, COL["tilt"]].max()), 1) if len(alive) else 0.0,
        "max_foot_depth_mm": round(float(-alive[:, COL["box_gap"]].min() * 1000), 2) if len(alive) else 0.0,
        "end_tilt_deg": round(float(alive[-1, COL["tilt"]]), 1) if len(alive) else 0.0,
        "flights": flights,
    }
