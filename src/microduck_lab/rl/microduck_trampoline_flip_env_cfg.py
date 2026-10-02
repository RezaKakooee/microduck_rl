"""Trampoline front flip: bounce, turn a full circle in the air, land on the feet, keep bouncing.

Episodic trick, run 1. What the scripted controller (tasks/trampoline/flip.py)
taught before this env existed:

- A front flip is this robot's natural direction: with its upright
  controllers off, the bed's push tips it forward at the soft hips.
- A tuck (knees, hips, head in) halves the pitch inertia, so it spins twice
  as fast in the air.
- A flip turns about the feet on the bed, so it travels forward (about 0.1 m/s
  per rad/s of spin). The bed is the wide one (1.2 m across).
- Coming in short of a full turn (about 320 deg) and upright-ish lands best.
  The hand-tuned landing passed 5 of 11 robustness variants: the reason for RL.

Scene and physics: see trampoline_scene.py (bed entity, foot boxes, 1 ms step,
BAM lag in physics steps scaled 5x).

Rewards (AGENTS.md lessons):
- flip_progress: potential-based, the forward rotation counted only while
  airborne (a flip leaves the bed; tipping over on the bed pays nothing),
  paid once up to 360 deg.
- height_progress: potential-based, the highest point above the spawn, paid
  once up to 0.5 m: the bounce that a flip needs.
- upright_after_flip: per step, gated on the rotation frontier passing
  300-340 deg. The landing and the bouncing after it earn this; nothing
  pre-flip can farm it.
- Terminations: any robot part other than the feet on the bed, or any part
  on the floor. No always-on upright term (it would oppose the flip).

Spawns: dropped from 0.05-0.5 m onto the bed, or in the air in the middle of
a flip (60-330 deg, spinning 4-12 rad/s, tucked), the reverse curriculum that
gives the landing on-policy data from the start.

The actor keeps the 61-D observation contract (so the runtime can run it);
the critic also sees the bed, the height above it and the flip state.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

from microduck_lab.rl import mdp_trampoline as tmdp
from microduck_lab.rl.trampoline_scene import (
    BED_CFG,
    DECIMATION,
    TIMESTEP,
    TRAMPOLINE_ROBOT_CFG,
    add_bed_contacts,
)

# ── Domain randomisation (the roulade's set, for sim2real parity) ────────────
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# The bed (run 2). Run 1 trained on one bed and lost a lot on a softer one
# (iter 1500: 99% flip-and-stay-up at 80 mm sag, 90% at 60 mm, 69% at 100 mm);
# the bed's damping hardly mattered (99% at rebound 0.85). A real bed will
# differ, so its spring, damper and moving mass are randomised per reset.
ENABLE_BED_RANDOMIZATION = True
BED_STIFFNESS_SCALE = (0.7, 1.6)   # sag 115..50 mm
BED_DAMPING_SCALE = (0.5, 4.0)
BED_MASS_SCALE = (0.5, 3.0)        # 5..30 g of moving mat

EPISODE_LENGTH_S = 6.0

# Pump-flip variant (pump_first=True). The flip policy above learned to flip
# at the first chance: from standing still it succeeded in 1% of episodes
# (92% crashed trying to flip off a tiny bounce). The user wants it to start
# standing, bounce a few times, then flip. So: most episodes start standing on
# the settled bed, the airborne rotation only counts after MIN_BOUNCES
# bounces, and each of those bounces pays once.
MIN_BOUNCES = 3
PUMP_EPISODE_LENGTH_S = 10.0

# Pump-flip curriculum (curriculum=True). Runs 3-6 started mostly standing and
# never found a real flip: run 4 farmed the counter with small tilts, runs 5-6
# bounced safely or crashed. The flip policy (run 2) flips 98% from a 5-15 cm
# drop but 1% from standing: it needs the drop's energy. So start where it
# works (drops, flip after 1 bounce) and take the free energy away in steps,
# until it starts standing and must bounce 3 times first.
PUMP_CURRICULUM = [
    {"step": 0, "params": dict(min_bounces=1, stand_prob=0.0, drop_prob=0.8, midflip_prob=0.2,
                               drop_height_range=(0.05, 0.5))},
    {"step": 600 * 24, "params": dict(min_bounces=2, stand_prob=0.2, drop_prob=0.6, midflip_prob=0.2,
                                      drop_height_range=(0.02, 0.3))},
    {"step": 1200 * 24, "params": dict(min_bounces=3, stand_prob=0.4, drop_prob=0.4, midflip_prob=0.2,
                                       drop_height_range=(0.0, 0.2))},
    {"step": 1800 * 24, "params": dict(min_bounces=3, stand_prob=0.6, drop_prob=0.2, midflip_prob=0.2,
                                       drop_height_range=(0.0, 0.1))},
]

# The scripted tuck (flip.py): crouch 40 mm, hips +0.8, neck +0.8, head +0.6.
# Servo-index keyed (0-4 left leg, 5-8 neck/head, 9-13 right leg).
TUCK_OVERRIDES = {
    2: 1.06, 3: 1.30, 4: 1.04,       # left hip_pitch, knee, ankle
    5: 1.15, 6: 0.95,                # neck_pitch, head_pitch
    11: -1.06, 12: -1.30, 13: -1.04,  # right leg, mirrored
}

FLIP_GATE_LO = math.radians(300.0)
FLIP_GATE_HI = math.radians(340.0)


def make_microduck_trampoline_flip_env_cfg(play: bool = False, pump_first: bool = False,
                                           curriculum: bool = False, stretch: float = 1.0) -> ManagerBasedRlEnvCfg:
    # stretch: longer curriculum stages. Run 7b's flip reward fell 0.69 -> 0.09 at
    # stage 1 and was only back to 0.27 when stage 2 came (AGENTS.md: too fast).
    stages = [dict(step=int(st["step"] * stretch), params=st["params"]) for st in PUMP_CURRICULUM]
    episode_s = PUMP_EPISODE_LENGTH_S if pump_first else EPISODE_LENGTH_S
    feet_bed = ContactSensorCfg(
        name=tmdp.FEET_SENSOR,
        primary=ContactMatch(mode="geom", pattern=r"^(left|right)_foot_box$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="bed", entity="bed"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    body_bed = ContactSensorCfg(
        name=tmdp.BODY_SENSOR,
        primary=ContactMatch(mode="body", pattern=r"^(?!ankle_).*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="bed", entity="bed"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    floor = ContactSensorCfg(
        name=tmdp.FLOOR_SENSOR,
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()

    # ── Scene and physics ────────────────────────────────────────────────────
    # The robot must stay the first entity (resets write its root through the
    # entity API, but other terms assume robot qpos first).
    cfg.scene.entities = {"robot": TRAMPOLINE_ROBOT_CFG, "bed": BED_CFG}
    cfg.scene.sensors = (feet_bed, body_bed, floor, self_collision)
    cfg.scene.spec_fn = add_bed_contacts
    cfg.scene.env_spacing = 0.0            # every robot is above its own bed at the origin
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.viewer.body_name = "trunk_base"
    cfg.sim.mujoco.timestep = TIMESTEP
    cfg.decimation = DECIMATION
    cfg.sim.nconmax = 80
    cfg.sim.njmax = 800
    cfg.episode_length_s = episode_s

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards ──────────────────────────────────────────────────────────────
    for name in ["track_linear_velocity", "track_angular_velocity", "upright", "pose", "air_time",
                 "foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"]:
        cfg.rewards.pop(name, None)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = 0.0        # a flip IS angular velocity
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.05

    if pump_first:
        cfg.rewards["bounce_progress"] = RewardTermCfg(func=tmdp.bounce_progress, weight=1.0)
        # Run 3 (no fall cost) dove into a flip from standing and never bounced.
        # Run 5 (charged always) stopped trying the flip; run 6: only while pumping.
        cfg.rewards["fall"] = RewardTermCfg(func=tmdp.fall_cost, weight=-5.0, params={"only_before_flip": True})
    cfg.rewards["flip_progress"] = RewardTermCfg(
        func=tmdp.flip_progress, weight=8.0, params={"target_angle": 2 * math.pi, "max_paid_rate": 20.0})
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=tmdp.height_progress, weight=1.0, params={"cap": 0.5})
    cfg.rewards["upright_after_flip"] = RewardTermCfg(
        func=tmdp.upright_after_flip, weight=3.0,
        params={"gate_lo": FLIP_GATE_LO, "gate_hi": FLIP_GATE_HI, "std": 0.4})
    cfg.rewards["roll"] = RewardTermCfg(func=tmdp.roll_cost, weight=-1.0)
    cfg.rewards["lateral_velocity"] = RewardTermCfg(func=tmdp.lateral_velocity_cost, weight=-1.0)
    cfg.rewards["off_center"] = RewardTermCfg(func=tmdp.off_center_cost, weight=-0.5)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-0.05, params={"sensor_name": self_collision.name})

    # ── Observations (actor: the 61-D contract, as the roulade) ──────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    for grp in ("actor", "critic"):
        del cfg.observations[grp].terms["height_scan"]
    for term in ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"):
        cfg.observations["critic"].terms.pop(term, None)

    actor = cfg.observations["actor"].terms
    for term in ("projected_gravity", "base_ang_vel", "joint_vel"):
        actor[term] = deepcopy(actor[term])
    for term in ("base_ang_vel", "projected_gravity"):
        actor[term].delay_min_lag, actor[term].delay_max_lag, actor[term].delay_update_period = 0, 1, 64
    actor["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    actor["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    actor["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    actor["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        actor["base_ang_vel"].func = microduck_mdp.base_ang_vel_imu_misaligned
        actor["base_ang_vel"].params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        actor["projected_gravity"].func = microduck_mdp.projected_gravity_imu_misaligned
        actor["projected_gravity"].params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
    actor["joint_vel"].delay_min_lag, actor["joint_vel"].delay_max_lag, actor["joint_vel"].delay_update_period = 1, 1, 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[grp].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6})

    critic = cfg.observations["critic"].terms
    critic["bed_state"] = ObservationTermCfg(func=tmdp.bed_state)
    critic["height_above_bed"] = ObservationTermCfg(func=tmdp.height_above_bed)
    critic["flip_state"] = ObservationTermCfg(func=tmdp.flip_state)
    critic["feet_on_bed"] = ObservationTermCfg(func=tmdp.feet_on_bed)
    critic["root_xy"] = ObservationTermCfg(func=tmdp.root_xy)

    # Command: tiny noise around zero, kept for the obs layout (as the roulade).
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (episode_s, episode_s * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ─────────────────────────────────────────────────────────
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)
    cfg.terminations["body_on_bed"] = TerminationTermCfg(func=tmdp.body_on_bed, time_out=False)
    cfg.terminations["touches_floor"] = TerminationTermCfg(func=tmdp.touches_floor, time_out=False)

    # ── Events ───────────────────────────────────────────────────────────────
    cfg.events.pop("push_robot", None)
    cfg.events.pop("foot_friction", None)     # the foot/bed pairs set their own friction
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup")
    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    # After reset_base / reset_robot_joints (dict order): starts from the HOME joints.
    cfg.events["reset_trampoline_state"] = EventTermCfg(
        func=tmdp.reset_trampoline_state,
        mode="reset",
        params=dict(drop_prob=0.2, midflip_prob=0.2, stand_prob=0.6, min_bounces=MIN_BOUNCES,
                    per_flight=True, tuck_overrides=TUCK_OVERRIDES) if pump_first else
               dict(drop_prob=0.5, midflip_prob=0.5, tuck_overrides=TUCK_OVERRIDES),
    )
    if ENABLE_BED_RANDOMIZATION:
        bed_joint = SceneEntityCfg("bed", joint_names=("bed_slide",))
        cfg.events["randomize_bed_stiffness"] = EventTermCfg(
            func=dr.joint_stiffness, mode="reset",
            params={"asset_cfg": bed_joint, "operation": "scale", "ranges": BED_STIFFNESS_SCALE})
        cfg.events["randomize_bed_damping"] = EventTermCfg(
            func=dr.joint_damping, mode="reset",
            params={"asset_cfg": deepcopy(bed_joint), "operation": "scale", "ranges": BED_DAMPING_SCALE})
        cfg.events["randomize_bed_mass"] = EventTermCfg(
            func=dr.body_mass, mode="reset",
            params={"asset_cfg": SceneEntityCfg("bed", body_names=("bed",)), "operation": "scale",
                    "ranges": BED_MASS_SCALE})
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add",
                    "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE)})
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES), "operation": "add",
                    "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE)})
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale",
                    "ranges": ARMATURE_RANDOMIZATION_RANGE})
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                    "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)})
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE})

    # ── Curriculum ───────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    # Mid-flip spawns never go to zero: they keep the landing practised.
    if curriculum:
        cfg.events["reset_trampoline_state"].params.update(PUMP_CURRICULUM[0]["params"])
    cfg.curriculum["spawn_mix"] = CurriculumTermCfg(
        # Steps counted from the start of this run, not from the checkpoint's (see the func).
        func=tmdp.relative_event_param_curriculum if curriculum else microduck_mdp.event_param_curriculum,
        params={
            "event_name": "reset_trampoline_state",
            "param_stages": stages if curriculum else [
                {"step": 0, "params": {"stand_prob": 0.6, "drop_prob": 0.2, "midflip_prob": 0.2}},
                {"step": 1500 * 24, "params": {"stand_prob": 0.7, "drop_prob": 0.15, "midflip_prob": 0.15}},
            ] if pump_first else [
                {"step": 0, "params": {"drop_prob": 0.5, "midflip_prob": 0.5}},
                {"step": 2000 * 24, "params": {"drop_prob": 0.7, "midflip_prob": 0.3}},
                {"step": 4000 * 24, "params": {"drop_prob": 0.85, "midflip_prob": 0.15}},
            ],
        },
    )
    return cfg


MicroduckTrampolineFlipRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_trampoline_flip",
    run_name="microduck_trampoline_flip",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
)


# Same networks and PPO; its own experiment folder.
MicroduckTrampolinePumpFlipRlCfg = deepcopy(MicroduckTrampolineFlipRlCfg)
MicroduckTrampolinePumpFlipRlCfg.experiment_name = "microduck_trampoline_pumpflip"
MicroduckTrampolinePumpFlipRlCfg.run_name = "microduck_trampoline_pumpflip"
MicroduckTrampolinePumpFlipRlCfg.max_iterations = 3000


# ── Flip from any pose ───────────────────────────────────────────────────────
# Run 2 flips 97-99% from a drop in the standing pose, but not from the pose an
# RL bounce policy has at the top of its flight (neck ~0.4 rad, a knee ~0.5 rad
# off), and moving to the standing pose in the air tips the trunk ~20 deg. So
# the flip policy is trained to flip from wider start poses and tilts.
FLIP_START_POSE_NOISE = 0.3          # rad, per servo (run 2: 0.03)
FLIP_START_TILT_MAX = math.radians(8.0)


def make_microduck_trampoline_flip_anypose_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_trampoline_flip_env_cfg(play=play)
    cfg.events["reset_trampoline_state"].params.update(
        joint_noise_std=FLIP_START_POSE_NOISE, tilt_max=FLIP_START_TILT_MAX)
    return cfg


# ── Bounce (no flip) ─────────────────────────────────────────────────────────
# The pump-flip runs (3-7) never pumped from standing: a bounce paid 1 point
# (up to 3) and a fall cost 5, so standing still won. This task only bounces:
# every landing pays for the height of that flight (up to BOUNCE_TARGET_M),
# upright pays every step, and a fall always costs. It is the first half of a
# two-policy flip (bounce, then hand over to a flip policy), as the runtime
# swaps policies that share the observation.
BOUNCE_TARGET_M = 0.12
BOUNCE_POSE_STD = 2.0


def make_microduck_trampoline_bounce_env_cfg(play: bool = False, target: float = BOUNCE_TARGET_M,
                                             pose_std: float | None = None) -> ManagerBasedRlEnvCfg:
    """target: bounce height that pays in full. 0.12 m (bounce1) gave ~15 cm bounces;
    the flip policy needs a hand-over from >= ~20 cm (from 15 cm it turns 296 deg and falls).
    pose_std: the bounce also pays for the standing pose at the top of the flight (see
    tmdp.bounce_height). 2.0 rad: bounce2's top pose (neck 103 deg off) scores ~0.24."""
    cfg = make_microduck_trampoline_flip_env_cfg(play=play, pump_first=True)
    for name in ("flip_progress", "upright_after_flip", "bounce_progress", "height_progress"):
        cfg.rewards.pop(name, None)
    cfg.rewards["bounce_height"] = RewardTermCfg(
        func=tmdp.bounce_height, weight=1.0, params={"target": target, "pose_std": pose_std})
    cfg.rewards["upright"] = RewardTermCfg(func=tmdp.upright, weight=1.0, params={"std": 0.4})
    cfg.rewards["fall"] = RewardTermCfg(func=tmdp.fall_cost, weight=-5.0, params={"only_before_flip": False})
    cfg.events["reset_trampoline_state"].params.update(
        stand_prob=0.8, drop_prob=0.2, midflip_prob=0.0, drop_height_range=(0.0, 0.2),
        min_bounces=0, per_flight=False)
    cfg.curriculum.pop("spawn_mix", None)
    return cfg


MicroduckTrampolineBounceRlCfg = deepcopy(MicroduckTrampolineFlipRlCfg)
MicroduckTrampolineBounceRlCfg.experiment_name = "microduck_trampoline_bounce"
MicroduckTrampolineBounceRlCfg.run_name = "microduck_trampoline_bounce"
MicroduckTrampolineBounceRlCfg.max_iterations = 3000
