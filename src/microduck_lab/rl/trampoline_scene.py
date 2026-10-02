"""The trampoline for mjlab: a bed entity, foot boxes on the robot, contact pairs.

The same bed as the CPU world in tasks/trampoline/world.py (the WIDE frame:
bed top 0.65 m above the floor, 0.6 m half width, 0.6 m of travel), checked
against it in MuJoCo Warp (local_storage/hb_dev/trampoline/warp_check.py:
a rigid weight rebounds 0.950 on both).

- Bed: a box on a vertical slide joint with a spring and a damper, weightless
  (gravcomp), 10 g of moving mass. The spring gives 80 mm of sag under the
  robot; the damper is fitted so a rigid weight of the robot's mass dropped
  500 mm comes back to 95% of it.
- Robot: the ground-contact model (the roulade's) plus one flat box per foot,
  on the flat patch of the sole. The feet touch the bed ONLY through these
  boxes (explicit pairs): a mesh sole on a box rolls forward (measured in the
  balance-board task). The soles are excluded from the bed; every other robot
  part still collides with it.
- Contact: stiff (solref 0.002). A soft contact on top of the bed spring ADDS
  energy (solref 0.01 gave 102% rebound, 0.02 gave 127%).
- Step: 1 ms. At 5 ms the light bed passed through the feet. BAM's command
  delay is counted in physics steps, so its lag is scaled 5x (15-30 steps =
  the same 15-30 ms as the other policies).
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    _BAM_ACTUATOR_KWARGS,
    get_standup_spec,
)
from microduck_lab.tasks.trampoline.world import BED_THICK, WIDE, fit_damping

TIMESTEP = 0.001
DECIMATION = 20               # 50 Hz policy, like every other microduck policy
BAM_LAG_STEPS = (15, 30)      # = (3, 6) steps at 5 ms

BED_TOP = WIDE.top            # m above the floor, at rest
BED_HALF = WIDE.half
BED_RANGE = WIDE.range
BED_MASS = 0.01               # kg
ROBOT_MASS = 0.7372           # kg, measured in the CPU scene (same export)
SAG_MM = 80.0
REBOUND = 0.95
REBOUND_DROP_MM = 500.0
BED_K = ROBOT_MASS * 9.81 / (SAG_MM / 1000)
BED_C = fit_damping(BED_K, ROBOT_MASS, REBOUND, BED_MASS, REBOUND_DROP_MM)

SOLREF = [0.002, 1.0]
SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]

# Flat patch of each sole, in the ankle body frame (from the balance-board
# task's robot_allcollisions_boardfeet.xml; the ankle frames of
# robot_groundcontact.xml are the same rotation).
FOOT_BOXES = {
    "left": dict(body="ankle_left", size=(0.0228, 0.0170, 0.0030),
                 pos=(-0.00699, -0.02088, -0.01430), quat=(0.0, 0.0, 0.737289, 0.675578)),
    "right": dict(body="ankle_right", size=(0.0228, 0.0169, 0.0030),
                  pos=(0.00699, -0.02087, -0.01424), quat=(0.675578, -0.737289, 0.0, 0.0)),
}
FOOT_BOX_NAMES = tuple(f"{s}_foot_box" for s in FOOT_BOXES)


def get_trampoline_robot_spec() -> mujoco.MjSpec:
    spec = get_standup_spec()
    for side, f in FOOT_BOXES.items():
        spec.body(f["body"]).add_geom(
            name=f"{side}_foot_box", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(f["size"]), pos=list(f["pos"]), quat=list(f["quat"]),
            contype=0, conaffinity=0, group=3, condim=3, density=0.0,
            friction=[1.0, 0.005, 0.0001], rgba=[0.2, 0.8, 0.2, 0.4],
        )
    return spec


def get_bed_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "bed"
    world = spec.worldbody
    ring = WIDE.ring
    for i in range(24):
        a, b = 2 * np.pi * i / 24, 2 * np.pi * (i + 1) / 24
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.007, 0, 0],
                       fromto=[ring * np.cos(a), ring * np.sin(a), BED_TOP, ring * np.cos(b), ring * np.sin(b), BED_TOP],
                       contype=0, conaffinity=0, density=0.0, rgba=[0.10, 0.35, 0.70, 1])
    for i in range(6):
        a = 2 * np.pi * (i + 0.5) / 6
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0],
                       fromto=[ring * np.cos(a), ring * np.sin(a), BED_TOP,
                               (ring + 0.03) * np.cos(a), (ring + 0.03) * np.sin(a), 0.004],
                       contype=0, conaffinity=0, density=0.0, rgba=[0.10, 0.35, 0.70, 1])
    bed = world.add_body(name="bed", pos=[0, 0, BED_TOP], gravcomp=1.0,
                         mass=BED_MASS, ipos=[0, 0, -0.004], inertia=[2e-4, 2e-4, 4e-4], explicitinertial=True)
    bed.add_joint(name="bed_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, range=list(BED_RANGE),
                  stiffness=[BED_K, 0, 0], damping=[BED_C, 0, 0])
    bed.add_geom(name="bed", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BED_HALF, BED_HALF, BED_THICK],
                 pos=[0, 0, -BED_THICK], group=3, priority=1, density=0.0,
                 friction=[1.0, 0.005, 0.0001], solref=SOLREF, solimp=SOLIMP)
    bed.add_geom(name="bed_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[WIDE.radius, 0.002, 0],
                 pos=[0, 0, -0.002], contype=0, conaffinity=0, density=0.0, rgba=[0.08, 0.08, 0.10, 1])
    return spec


def add_bed_contacts(spec: mujoco.MjSpec, robot: str = "robot", bed: str = "bed", terrain: str = "terrain") -> None:
    """Scene spec_fn: foot-box/bed pairs; soles, floor and bed kept apart."""
    names = {g.name for g in spec.geoms}
    bed_geom = f"{bed}/bed"
    if bed_geom not in names:
        raise ValueError(f"[trampoline] bed geom '{bed_geom}' not in the scene")
    for side, f in FOOT_BOXES.items():
        box = f"{robot}/{side}_foot_box"
        if box not in names:
            raise ValueError(f"[trampoline] foot box '{box}' not in the scene")
        spec.add_pair(name=f"{side}_foot_bed", geomname1=box, geomname2=bed_geom, condim=3,
                      friction=[1.0, 1.0, 0.005, 0.0001, 0.0001], solref=SOLREF, solimp=SOLIMP)
        spec.add_exclude(bodyname1=f"{robot}/{f['body']}", bodyname2=f"{bed}/bed")
    spec.add_exclude(bodyname1=terrain, bodyname2=f"{bed}/bed")


BAM_1MS = FrictionDRBamActuatorCfg(**dict(_BAM_ACTUATOR_KWARGS,
                                          delay_min_lag=BAM_LAG_STEPS[0], delay_max_lag=BAM_LAG_STEPS[1]))

TRAMPOLINE_ROBOT_CFG = EntityCfg(
    spec_fn=get_trampoline_robot_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(actuators=(BAM_1MS,), soft_joint_pos_limit_factor=0.9),
)

BED_CFG = EntityCfg(
    spec_fn=get_bed_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    articulation=EntityArticulationInfoCfg(),
)
