"""Microduck velocity environment — skate-blade variant.

Same task as the walking recipe (track a commanded twist; head/body pose slots
zero-padded into the shared 61D obs), but the feet touch the floor through
ANISOTROPIC contact pairs: slippery along the sole's long axis, grippy across
it. That is what a real skate blade gives a skater — something to push against
sideways while gliding forward. The isotropic ice task
(`microduck_velocity_ice_env_cfg.py`) showed that with flat rubber soles on
low friction there is nothing to push against in any direction, and the
policy converged to a non-propulsive shuffle three runs out of three.

Built on `make_microduck_velocity_env_cfg` so DR / obs / noise / delays stay
in sync for free. What changes:

1. **Contact pairs.** A `spec_fn` adds one `<pair>` per foot against the
   terrain plane, `condim=4`, two sliding coefficients. Pairs override geom
   friction, so the base `foot_friction` geom DR is deleted: it would no
   longer reach the physics.

2. **The pair frame is world-fixed, so the blade is re-projected every step.**
   Measured: for any contact with the +Z plane, tangent1 = world +Y and
   tangent2 = world -X, regardless of foot yaw (`mdp_blades.py` has the
   numbers). A static pair is a blade only while the foot points along X. So
   a step-mode event (`project_blade_friction`) rewrites the per-env
   `pair_friction` from each foot's yaw and load: a loaded foot gets the true
   blade ellipse's friction radius along the direction it is pushing (grip
   perpendicular to the blade, glide along it, ~0.09 for a straight-back
   shove with a 30-degree turned-out foot), an unloaded foot gets the
   ellipse's world-axis intercepts. The bounding-box alternative
   (BLADE_RULE = "bbox") hands a turned-out walking gait mu_x 0.5-0.6 for
   free, measured on the pretrained walker, so a policy never has to skate.

3. **Heading hold.** mjlab's `heading_command` drives the yaw-rate slot from
   a heading error toward a target sampled near 0, and the reset yaw range
   is narrowed to match. Turn-in-place practice is off. The obs contract is
   unchanged: slot 2 is still a yaw-rate command, just a small corrective one.

4. **Curriculum from walking to blades.** Stage 0 is isotropic walking grip
   (along = across = 0.7-1.3): the first 1000 iterations ARE the walking
   task. Then `mu_along` ramps down to ice while `mu_across` stays grippy.
   The ice runs taught that dropping straight onto a slippery floor makes
   "stand still" the argmax.

5. **`foot_slip` = 0.** Slip along the blade is the medium, not an error.
   `air_time` keeps its walking weight with the ice window (a glide holds a
   foot down longer than a stride).

Judge a run by `Metrics/twist/error_vel_xy` and a rollout, never by
`air_time` — the ice journey doc explains why.
"""

from copy import deepcopy

import mujoco as _mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg

from microduck_lab.rl import mdp_blades
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# ── Blade pairs ──────────────────────────────────────────────────────────────
# mjlab attaches the robot with prefix "robot/"; the plane geom is "terrain".
ROBOT_ENTITY = "robot"
FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
BLADE_PAIR_NAMES = ("blade_left", "blade_right")
TERRAIN_BODY = "terrain"

# Torsional / rolling terms of the pair (indices 2, 3, 4). Same as the
# MuJoCo geom defaults so only the two sliding terms differ from walking.
BLADE_SPIN_MU = 0.005
BLADE_ROLL_MU = 0.0001

# How the yawed blade ellipse is squeezed into MuJoCo's axis-aligned pair
# friction each step: "load" (true radius along the foot's current push
# direction) or "bbox" (bounding box; exploitable by a turned-out walk).
BLADE_RULE = "load"
# The foot contact sensor the "load" rule reads. Its primary order is LEFT,
# RIGHT — the same order as FOOT_GEOM_NAMES / BLADE_PAIR_NAMES.
FOOT_CONTACT_SENSOR = "feet_ground_contact"

# ── Blade friction curriculum ────────────────────────────────────────────────
# Stage steps are env steps: iteration * NUM_STEPS_PER_ENV (24).
# `along` is the coefficient on the sole's long axis, `across` perpendicular.
# Stage 0 is isotropic walking grip; only `along` ramps.
BLADE_ALONG_FINAL = (0.04, 0.15)   # real steel-on-ice is ~0.005-0.02; rubber sole
BLADE_ACROSS_FINAL = (0.80, 1.30)  # the blade edge bites
BLADE_STAGES = [
    {"step":    0 * NUM_STEPS_PER_ENV, "along": (0.70, 1.30), "across": (0.70, 1.30)},  # walking
    {"step": 1000 * NUM_STEPS_PER_ENV, "along": (0.40, 0.90), "across": (0.70, 1.30)},
    {"step": 2000 * NUM_STEPS_PER_ENV, "along": (0.25, 0.60), "across": (0.80, 1.30)},
    {"step": 3000 * NUM_STEPS_PER_ENV, "along": (0.15, 0.40), "across": BLADE_ACROSS_FINAL},
    {"step": 4000 * NUM_STEPS_PER_ENV, "along": (0.08, 0.25), "across": BLADE_ACROSS_FINAL},
    {"step": 5000 * NUM_STEPS_PER_ENV, "along": BLADE_ALONG_FINAL, "across": BLADE_ACROSS_FINAL},
]

# ── Heading hold ─────────────────────────────────────────────────────────────
# The projection is exact at heading 0; keep the duck facing +X.
RESET_YAW_RANGE = (-0.2, 0.2)
HEADING_TARGET_RANGE = (-0.2, 0.2)
HEADING_STIFFNESS = 2.0            # yaw-rate cmd = stiffness * heading error
ANG_VEL_Z_CLIP = (-1.0, 1.0)       # clip on that yaw-rate command
LIN_VEL_X_RANGE = (-0.3, 0.5)      # forward-biased: skating backwards is rare
LIN_VEL_Y_RANGE = (-0.1, 0.1)      # small but alive (obs slot must not die)
TURN_IN_PLACE_FRACTION = 0.0

# ── Reward adjustments ───────────────────────────────────────────────────────
FOOT_SLIP_WEIGHT = 0.0
AIR_TIME_WEIGHT = 3.0
AIR_TIME_WINDOW = (0.10, 0.40)


def _add_blade_pairs(spec: _mujoco.MjSpec) -> None:
    """Add one anisotropic contact pair per foot against every terrain geom.

    Works on the scene spec after the robot is attached (geoms are
    ``robot/<name>``) and on a bare spec with unprefixed names (tests, CPU
    playback). Initial sliding terms are isotropic 1.0; the DR event and the
    per-step projection own them from the first reset on.
    """
    geom_names = {g.name for g in spec.geoms}
    terrain_geoms = [g.name for g in spec.body(TERRAIN_BODY).geoms if g.name]
    if not terrain_geoms:
        raise ValueError("[blades] terrain body has no named geoms to pair against")
    if len(terrain_geoms) > 1:
        raise NotImplementedError(
            "[blades] one pair per foot assumes a single terrain geom (plane); "
            f"got {len(terrain_geoms)} — rough terrain is not supported"
        )
    count = 0
    for foot, pair_name in zip(FOOT_GEOM_NAMES, BLADE_PAIR_NAMES):
        full = f"{ROBOT_ENTITY}/{foot}" if f"{ROBOT_ENTITY}/{foot}" in geom_names else foot
        if full not in geom_names:
            raise ValueError(f"[blades] foot geom '{foot}' not found in spec")
        spec.add_pair(
            name=pair_name,
            geomname1=full,
            geomname2=terrain_geoms[0],
            condim=4,
            friction=[1.0, 1.0, BLADE_SPIN_MU, BLADE_ROLL_MU, BLADE_ROLL_MU],
        )
        count += 1
    print(f"[blades] spec_fn: added {count} anisotropic foot-terrain pair(s) "
          f"(condim=4) -> {list(BLADE_PAIR_NAMES)}")


def make_microduck_velocity_blades_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Velocity tracking on skate blades (anisotropic foot-floor friction)."""
    if rough:
        raise NotImplementedError(
            "Blades pairs are built against the single terrain plane; "
            "rough terrain would need one pair per foot per box."
        )
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    previous_spec_fn = cfg.scene.spec_fn
    if previous_spec_fn is None:
        cfg.scene.spec_fn = _add_blade_pairs
    else:

        def _chain(spec: _mujoco.MjSpec) -> None:
            previous_spec_fn(spec)
            _add_blade_pairs(spec)

        cfg.scene.spec_fn = _chain

    # Pairs override geom friction: the base foot geom DR no longer reaches
    # the floor contact. Remove it rather than leave a misleading no-op.
    del cfg.events["foot_friction"]

    sensor_names = {s.name for s in cfg.scene.sensors}
    if FOOT_CONTACT_SENSOR not in sensor_names:
        raise ValueError(f"[blades] '{FOOT_CONTACT_SENSOR}' sensor missing; have {sensor_names}")
    blade_params = {
        "pair_names": BLADE_PAIR_NAMES,
        "rule": BLADE_RULE,
        "sensor_name": FOOT_CONTACT_SENSOR,
    }
    stage0 = BLADE_STAGES[0]
    cfg.events["blade_friction"] = EventTermCfg(
        func=mdp_blades.randomize_blade_friction,
        mode="reset",
        params={
            "along_range": stage0["along"],
            "across_range": stage0["across"],
            **blade_params,
        },
    )
    cfg.events["blade_projection"] = EventTermCfg(
        func=mdp_blades.project_blade_friction,
        mode="step",
        params=dict(blade_params),
    )
    cfg.curriculum["blade_friction"] = CurriculumTermCfg(
        func=mdp_blades.blade_friction_curriculum,
        params={"event_name": "blade_friction", "stages": BLADE_STAGES},
    )

    # Heading hold (see module docstring, item 3).
    twist = cfg.commands["twist"]
    twist.heading_command = True
    twist.heading_control_stiffness = HEADING_STIFFNESS
    twist.rel_heading_envs = 1.0
    twist.ranges.heading = HEADING_TARGET_RANGE
    twist.ranges.lin_vel_x = LIN_VEL_X_RANGE
    twist.ranges.lin_vel_y = LIN_VEL_Y_RANGE
    twist.ranges.ang_vel_z = ANG_VEL_Z_CLIP
    twist.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    cfg.events["reset_base"].params["pose_range"]["yaw"] = RESET_YAW_RANGE

    # Rewards (module docstring, item 5).
    cfg.rewards["foot_slip"].weight = FOOT_SLIP_WEIGHT
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_WINDOW[0]
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_WINDOW[1]

    return cfg


MicroduckBladesRlCfg = deepcopy(MicroduckRlCfg)
MicroduckBladesRlCfg.experiment_name = "velocity_blades"
