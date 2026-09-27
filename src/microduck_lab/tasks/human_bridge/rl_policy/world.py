"""The set and the two ducks for the human bridge.

Two real microducks in one MuJoCo world, both on BAM XL330 M6 motors, as in
training and in `scripts/infer_policy.py`:

* `he`  -- the big brother (blue). He becomes the bridge.
* `she` -- the little sister (pink). She crosses on his back.

The prefixes are the love-story ones, so `film.stage.paint` colours them.

Only motor commands move anything after `World()` returns. No external
forces, no welds, no teleports. `World.step` checks this every call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController

from microduck_lab.film import stage

# The policies run at 50 Hz. Physics runs at 2.5 ms: duck-on-duck contact
# between two meshes is stiffer than a sole on a plane.
TIMESTEP = 0.0025
CONTROL_DT = 0.02
SUBSTEPS = int(round(CONTROL_DT / TIMESTEP))

# BAM settings copied from scripts/infer_policy.py (its defaults).
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_MIN = 6.0
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

SERVOS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")

# Collision bits. Duck i is 1 << i. Loose props are 8, the set is 16.
BIT = {"he": 1, "she": 2}
BIT_PROP = 8
BIT_SET = 16
ALL = 31

# The visible meshes that also collide. The robot MJCF only collides its
# group-3 geoms, which leave out the thighs and the trunk shells. She stands
# on exactly those parts of him, so they must be solid.
# The servos (xl330, all 15), the thigh plates and the ankle brackets are hard
# parts too. When they were only drawn, her feet sank 17 mm into his shin
# servos on video. MuJoCo collides each mesh as its convex hull: 1.2x the
# servo's volume, about 3x for a bracket (the hull fills the U round its servo).
SOLID_MESHES = ("left_shell", "right_shell", "trunk_base", "upper_leg_left",
                "upper_leg_right", "foot_left", "foot_right", "neck",
                "xl330", "upper_leg_rigidity_plate", "ankle_left", "ankle_right")

WOOD = (0.55, 0.40, 0.27, 1)
WOOD_DARK = (0.40, 0.28, 0.18, 1)
STONE = (0.72, 0.70, 0.66, 1)


@dataclass
class Box:
    """A static box of the set, by its min and max corners."""
    name: str
    lo: tuple
    hi: tuple
    rgba: tuple = STONE
    friction: float = 1.0

    def xml(self):
        lo, hi = np.array(self.lo, float), np.array(self.hi, float)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        return (f'<geom name="{self.name}" type="box" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
                f'size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" rgba="{" ".join(map(str, self.rgba))}" '
                f'friction="{self.friction} .005 .0001"/>')


@dataclass
class SetDesign:
    """Boxes of the set. The floor is always there, at z = 0."""
    boxes: list = field(default_factory=list)
    extra: str = ""          # more worldbody XML (tests only: probes, loads)


def world_xml(design: SetDesign):
    boxes = "\n".join(b.xml() for b in design.boxes)
    return f'''<mujoco model="human_bridge">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" iterations="100" ls_iterations="50"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/>
    <headlight ambient=".30 .30 .30" diffuse=".50 .50 .50" specular=".1 .1 .1"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .74 .92" rgb2=".95 .97 1" width="512" height="512"/>
    <texture name="yard" type="2d" builtin="checker" rgb1=".36 .38 .40" rgb2=".40 .42 .44" width="512" height="512"/>
    <material name="yard" texture="yard" texrepeat="12 12" reflectance="0"/>
  </asset>
  <default><geom friction="1 .005 .0001" solref=".008 1" solimp=".98 .999 .001"/></default>
  <worldbody>
    <light pos="1 -2 3" dir="-.3 .5 -1" directional="true" ambient=".2 .2 .2" diffuse=".8 .8 .8"/>
    <light pos="-1 1 2" diffuse=".3 .3 .3" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 .1" material="yard" rgba="1 1 1 1"/>
    {boxes}
    {design.extra}
  </worldbody>
</mujoco>'''


class Duck:
    """One prefixed duck in the shared world: indices, motors and readings."""

    def __init__(self, world, prefix):
        m = world.model
        self.world, self.prefix = world, prefix
        self.model, self.data = m, world.data
        name = lambda kind, n: mujoco.mj_name2id(m, getattr(mujoco.mjtObj, "mjOBJ_" + kind), f"{prefix}_{n}")
        self.trunk = name("BODY", "trunk_base")
        self.head = name("BODY", "jaw_soft")
        jid = name("JOINT", "trunk_base_freejoint")
        self.adr = int(m.jnt_qposadr[jid])
        self.dof = int(m.jnt_dofadr[jid])
        self.acts = [name("ACTUATOR", j) for j in SERVOS]
        self.mouth = name("ACTUATOR", "mouth")
        joints = [int(m.actuator_trnid[a, 0]) for a in self.acts]
        self.qidx = np.array([m.jnt_qposadr[j] for j in joints])
        self.vidx = np.array([m.jnt_dofadr[j] for j in joints])
        self.lo = m.jnt_range[joints, 0].copy()
        self.hi = m.jnt_range[joints, 1].copy()
        self.foot_sites = (name("SITE", "left_foot"), name("SITE", "right_foot"))
        self.bodies = {b for b in range(m.nbody)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix + "_")}
        self.geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.bodies]
        self.solid = [g for g in self.geoms if m.geom_contype[g]]
        self.sole_geoms = (name("GEOM", "left_foot_collision"), name("GEOM", "right_foot_collision"))
        # Each duck owns its BAM model: the controller writes the voltage sag
        # into it every step, so a shared one would couple the two batteries.
        bam = load_model(motor_name="xl330", model="m6")
        bam.actuator.kp = BAM_KP_FW
        bam.actuator.vin = BAM_VIN
        bam.actuator.max_current = None
        self.motor = MujocoController(bam, [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self.acts],
                                      m, world.data, vin_drop_gain=BAM_VIN_DROP_GAIN, vin_min=BAM_VIN_MIN)
        self.target = np.zeros(len(SERVOS))

    # -- readings ----------------------------------------------------------

    @property
    def q(self):
        return self.data.qpos[self.qidx].copy()

    def pos(self):
        return self.data.qpos[self.adr:self.adr + 3].copy()

    def R(self):
        return self.data.xmat[self.trunk].reshape(3, 3).copy()

    def up(self):
        return float(self.data.xmat[self.trunk][8])

    def yaw(self):
        qw, qx, qy, qz = self.data.qpos[self.adr + 3:self.adr + 7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def com(self):
        return self.data.subtree_com[self.trunk].copy()

    def mass(self):
        return float(self.model.body_subtreemass[self.trunk])

    def feet(self):
        return [self.data.site_xpos[s].copy() for s in self.foot_sites]

    def torque(self):
        return self.data.actuator_force[self.acts].copy()

    # -- commands ----------------------------------------------------------

    def hold(self, target):
        """Joint position targets for the firmware position loop (BAM)."""
        target = np.asarray(target, float)
        if not np.isfinite(target).all():
            raise RuntimeError(f"{self.prefix}: non-finite joint target")
        self.target = np.clip(target, self.lo, self.hi)
        self.motor.q_target[:] = self.target

    def points(self, geoms):
        """World positions of the mesh vertices of `geoms`."""
        m, d = self.model, self.data
        out = []
        for g in geoms:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            out.append(m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(out)

    def stand_on(self, x_front, y, z, q, yaw=0.0):
        """Set him down in pose `q`, soles touching height `z`, the front of
        his feet at `x_front` (for yaw 0). Setup only."""
        self.place((0.0, y, z + 0.2), yaw=yaw, q=q)
        soles = self.points(self.sole_geoms)
        p = self.pos()
        p[0] += x_front - soles[:, 0].max()
        p[2] += z + 0.0005 - soles[:, 2].min()
        self.place(p, yaw=yaw, q=q)

    def place(self, xyz, yaw=0.0, pitch=0.0, q=None):
        """Initial placement only. `World` refuses this once the story runs."""
        if self.world.running:
            raise RuntimeError("place() is for setup; the story may only move motors")
        d = self.data
        d.qpos[self.adr:self.adr + 3] = xyz
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        d.qpos[self.adr + 3:self.adr + 7] = (cy * cp, -sy * sp, cy * sp, sy * cp)
        d.qvel[self.dof:self.dof + 6] = 0
        if q is not None:
            d.qpos[self.qidx] = q
            self.hold(q)
        mujoco.mj_forward(self.model, d)


class Brain:
    """The pretrained walking and standing policies, pointed at one duck.

    `PolicyInference` looks its sensors up by bare name and sizes itself from
    `model.nu`. In a two-duck world both are wrong, so it is built on the
    shared model and then aimed at this duck's trunk, gyro and 14 servos, as
    `love_story.original.Agent` does. Its targets go to this duck's BAM
    controller, exactly as in `scripts/infer_policy.py`.
    """

    def __init__(self, duck, stand_only=False):
        import contextlib
        import io
        from microduck_lab import paths
        from microduck_lab.sim.upstream import PolicyInference

        m = duck.model
        with contextlib.redirect_stdout(io.StringIO()):
            pol = PolicyInference(
                m, duck.data,
                walking_onnx_path=None if stand_only else str(paths.WALKING),
                standing_onnx_path=str(paths.STAND),
                bam_ctrl=duck.motor, new_cmd_obs=True, use_projected_gravity=True)
        pol.n_joints = len(SERVOS)
        pol.joint_qpos_indices = list(duck.qidx)
        pol.joint_qvel_indices = list(duck.vidx)
        pol.default_pose = pol.default_pose[:len(SERVOS)]
        pol.last_action = np.zeros(len(SERVOS), dtype=np.float32)
        pol.trunk_base_id = duck.trunk
        pol.imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{duck.prefix}_imu_ang_vel")
        assert pol.imu_ang_vel_id >= 0
        pol.vel_max_x, pol.vel_min_x = 0.3, -0.3
        pol.vel_max_y, pol.vel_min_y = 0.2, -0.2
        pol.vel_max_ang = 1.5
        self.pol, self.duck = pol, duck
        self.quiet = lambda: contextlib.redirect_stdout(io.StringIO())

    def command(self, vx=0.0, wz=0.0):
        with self.quiet():
            self.pol.set_vel_cmd(vx, 0.0, wz)

    def act(self):
        """One 50 Hz policy step: infer, then send the targets to the motors."""
        with self.quiet():
            self.pol.apply_action(self.pol.infer())
        self.duck.target = self.duck.motor.q_target.copy()

    @property
    def default_pose(self):
        return np.asarray(self.pol.default_pose, float)


def build(design: SetDesign, cast=("he", "she"), edit=None):
    """Compile the set with one duck per prefix. Returns the model.

    `edit(spec)` may add more static set pieces before compile (test benches:
    a fixed copy of his body). They collide as part of the set."""
    spec = mujoco.MjSpec.from_string(world_xml(design))
    for prefix in cast:
        duck = mujoco.MjSpec.from_file(stage.DUCK_XML)
        spec.attach(duck, prefix=prefix + "_", frame=spec.worldbody.add_frame())
    if edit is not None:
        edit(spec)
    arm(spec, cast)
    # BAM owns the 14 servos: position actuators become torque motors bounded
    # by the stall torque, and the joints lose the XML damping and friction
    # that BAM recomputes every step (bam.mjlab.BamActuator.edit_spec).
    bam = load_model(motor_name="xl330", model="m6")
    limit = BAM_VIN * bam.kt.value / bam.R.value
    servos = {f"{p}_{j}" for p in cast for j in SERVOS}
    for act in spec.actuators:
        if act.name in servos:
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-limit, limit)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
    for j in spec.joints:
        if j.name in servos:
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = STIFF_SOLREF_FRICTION
            j.solimp_friction = STIFF_SOLIMP_FRICTION
    model = spec.compile()
    stage.paint(model)
    check_armed(model)
    return model


def arm(spec, cast):
    """Who touches whom. A duck never touches itself; its CAD parts overlap
    at every joint. Ducks touch the set and each other.

    This must run on the spec, BEFORE compile. MuJoCo also filters contacts
    per body (body_contype / body_conaffinity, the OR of the body's geoms),
    and the compiler sets those. Until 2026-09-26 the bits were written into
    the compiled model instead: every body with no group-3 geom kept body
    bits 0/0, so its geoms never collided. His neck rested 16.8 mm inside
    the far shelf, and her thighs could pass through him unseen.
    """
    for body in spec.bodies:
        prefix = body.name.split("_")[0]
        for geom in body.geoms:
            geom.margin = 0.0
            if prefix in cast:
                # The head shells, shins and jaw already have a group-3 copy.
                mesh = geom.meshname.partition("_")[2] if geom.type == mujoco.mjtGeom.mjGEOM_MESH else ""
                solid = geom.group == 3 or mesh in SOLID_MESHES
                bit = BIT[prefix]
                geom.contype = bit if solid else 0
                geom.conaffinity = (ALL ^ bit) if solid else 0
            elif body.name == "world":
                geom.contype = BIT_SET
                geom.conaffinity = ALL ^ BIT_SET
            else:
                geom.contype = BIT_PROP
                geom.conaffinity = ALL


def check_armed(model):
    """Every geom meant to collide sits on a body that can collide."""
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        b = model.geom_bodyid[g]
        if (model.body_contype[b] & model.geom_contype[g]) != model.geom_contype[g] or \
                (model.body_conaffinity[b] & model.geom_conaffinity[g]) != model.geom_conaffinity[g]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            raise RuntimeError(f"body {name} cannot collide, but its geom {g} should")


class World:
    """The model, its data, and the ducks. `step` advances one 50 Hz tick."""

    GUARDED = ("geom_contype", "geom_conaffinity", "body_contype", "body_conaffinity",
               "body_gravcomp", "body_mass", "geom_pos", "geom_size", "geom_friction", "geom_solref")

    def __init__(self, design: SetDesign, cast=("he", "she"), edit=None):
        self.model = build(design, cast, edit)
        self.data = mujoco.MjData(self.model)
        self.running = False
        self.ducks = {p: Duck(self, p) for p in cast}
        self.t = 0.0
        self.guard = None
        mujoco.mj_forward(self.model, self.data)

    def start(self):
        """From here on only motor targets may change the world."""
        for duck in self.ducks.values():
            duck.motor.reset(self.data.qpos)
            duck.motor.q_target[:] = duck.target
        self.guard = {k: getattr(self.model, k).copy() for k in self.GUARDED}
        self.running = True

    def step(self):
        m, d = self.model, self.data
        q, v = d.qpos.copy(), d.qvel.copy()
        for _ in range(SUBSTEPS):
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
                raise RuntimeError("external force applied")
            for duck in self.ducks.values():
                duck.motor.update()
            mujoco.mj_step(m, d)
        if self.guard is not None:
            for k, v0 in self.guard.items():
                if not np.array_equal(getattr(m, k), v0):
                    raise RuntimeError(f"model field {k} changed during the story")
        if not np.isfinite(d.qpos).all():
            raise RuntimeError("simulation went non-finite")
        self.t += CONTROL_DT
        return q, v

    def snapshot(self):
        """Everything needed to resume the story here, motors included."""
        d = self.data
        return dict(t=self.t, time=d.time, qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                    act=d.act.copy(), ctrl=d.ctrl.copy(), warm=d.qacc_warmstart.copy(),
                    motors={p: (dk.motor.q_target.copy(), dk.motor.last_ts,
                                dk.motor._prev_motor_torque.copy(), dk.target.copy())
                            for p, dk in self.ducks.items()})

    def restore(self, s):
        d = self.data
        self.t, d.time = s["t"], s["time"]
        d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:] = s["qpos"], s["qvel"], s["act"], s["ctrl"]
        d.qacc_warmstart[:] = s["warm"]
        for p, (qt, ts, tq, tg) in s["motors"].items():
            dk = self.ducks[p]
            dk.motor.q_target[:] = qt
            dk.motor.last_ts = ts
            dk.motor._prev_motor_torque[:] = tq
            dk.target = tg.copy()
        mujoco.mj_forward(self.model, d)

    def contacts(self, a, b):
        """Contacts between geom sets `a` and `b`: list of (pos, normal, force)."""
        m, d = self.model, self.data
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in a and c.geom2 in b) or (c.geom2 in a and c.geom1 in b):
                mujoco.mj_contactForce(m, d, i, f6)
                out.append((c.pos.copy(), c.frame[:3].copy(), float(f6[0])))
        return out
