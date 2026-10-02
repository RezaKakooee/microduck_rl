"""Microduck: walk sideways along narrow beams with height changes.

Teaches the sideways walker (sideways v4) the skill the human bridge needs and
it lacks. Built on `make_microduck_velocity_sideways_env_cfg`, so actions, the
61D actor obs, DR, obs noise and delays are the walking recipe's.

What the walker cannot do today (local_storage/hb_dev/rl_policy/s1_envelope,
sideways_v4_iter1000 + FF5 steering, story.Judge rules, 3 seeds each):

* flat beams 64 -> 28 mm: 15/15 crossed;
* on 64-150 mm beams: steps up/down <= 6 mm, ramps up <= 15 mm per 190 mm,
  ramps down <= 20 mm per 190 mm;
* on beams <= 48 mm nearly any step or ramp fails; a 16 mm hump fails at
  every width.

His back, for a rigid 41 x 54 mm sole (local_storage/hb_dev/rl_policy/s1_set): a 35 mm
gap, +6.2 mm, +4.5 mm, 22-55 mm wide, then a 15 deg ramp down of 22 mm.

Design (numbers at the top of this file and in `beam_terrain` / `mdp_beam`):

1. Terrain (`beam_terrain`). mjlab's terrain generator, one beam per tile:
   300 mm start platform, 2.4 m beam, 300 mm end platform, no floor (a drop
   beside the beam). 10 columns: flat, steps, ramps, pits, bumps, mixed;
   narrow_steps, narrow_mixed, crowned (width follows the row but features
   stay <= 12 mm; crowned = a rounded top like his trunk); and "his_back"
   (his F1w back as her sole meets it: rails, holes, heights). 10 rows:
   difficulty d = row / 9 sets width 150 -> 28 mm and features up to d x
   (steps 25 mm, ramps 20 deg, gaps 40 mm, bumps 20 mm; 12 mm in the small-
   feature columns). Row 0 is flat 150 mm. Envs start on rows 0-2 (width
   >= 110 mm, steps <= 6 mm: inside the measured envelope). A row up after
   >= 0.8 m of new ground in an episode that did not fail; a row down after
   < 0.2 m (`mdp_beam.terrain_levels_beam`; a demotion also needs a failure
   or an episode >= 10 s). Spawns: 25 % on the start platform, the rest at
   a random flat spot along the beam, including right before each feature
   (reverse curriculum).
2. Command = the deployment steering law, every step (`mdp_beam.BeamSteeringCommand`):
   vy ~ U(0.08, 0.14); vx = clip(vx0 + 5 y_off, +-0.3); wz = clip(wz0 - 2 heading_err, +-0.4);
   vx0 = -0.055 +- 0.01, wz0 = 0.06 +- 0.02 per resample, +- 5 mm/s and +- 0.02 rad/s
   noise. Her forward is world -y, so y_off > 0 gives vx > 0 (checked in the
   tests). vx_max 0.3 (was 0.1, which clipped her only offset sensor at
   y < -9 mm). The command is never zero: the law runs from the first step
   on every spawn (the story's standing policy covers her wait).
   Resample every 10-20 s. Head command: the recipe's smallest ranges, no
   curriculum, weight 0.5 (gated): a small bonus; its 0.5 rad std is too wide
   to train the head inputs, the tiny ranges only keep them alive.
   Body-pose slots: the recipe's tiny ranges, weight 0. 61D actor obs.
   The law feeds vx0 = -0.055 because the v4 walker drifts forward at vx = 0.
   A policy that tracked vx exactly would settle 11 mm off the beam axis
   (0.055 / 5). So vx is NOT tracked; the policy is paid for the outcome
   (on the axis) and learns its own response to the law. Measured in this
   sim with the warm-start actor (s3_train/diag_law.py, flat 150 mm, 16 envs,
   30 s): the law settles her +7 mm off the axis (9 mm rms, heading +1.5 deg);
   with vx0 = wz0 = 0 it is -4 mm (8 mm rms).
3. Terminations as story.Judge: tilt > acos(0.9) = 25.8 deg; any of her
   35 non-foot story-solid parts touching the terrain (the walking model only
   collides its soles, so those parts are named in `beam_walk_spec` and
   collide with the terrain only); a foot 15 mm below the path surface under
   it (beside the beam or down a gap); trunk 60 mm below standing height.
   Reaching the end platform ends the episode as a time-out. mjlab's
   out_of_terrain_bounds is removed (it fires on the outer tiles). nconmax 100
   (measured peak use: 2.1 contacts per env).
4. Rewards, per second before mjlab's dt scaling (dt = 0.02 s):
   * progress: +50 per metre of NEW ground (ratchet on the best x): rocking
     or stepping back and forth pays 0.
   * gate: lateral speed, yaw-rate, upright, pose, air time, swing lift and
     head pose are multiplied by g = clip(EMA_0.5s(new ground speed) / (0.3 vy_cmd), 0, 1).
     Marching in place or standing on the beam gets g < 0.2 after 1 s, < 0.03 after 2 s.
     (bridge v5-v10 marched in place, paid by exactly these terms.) 0.3, not
     0.5: slow careful steps on a hard spot keep their pay.
   * failure: -500 x dt = -10 per episode that ends by a failure
     (bridge v11b had -5 x dt = -0.1).
   * costs: offset from the beam axis -3000 y^2 and heading error -10 err^2,
     both saturating (cap 20 mm / 0.25 rad, `mdp_beam.saturating_sq`): 5 s of
     the worst of both (9.1) costs less than one fall (10), so after a drift
     staying up is never worse than falling. The recipe's penalties (action
     rate fixed at -0.4, the value sideways v4 had at iter 750-1000).
   * no env waits and the command is never zero (START_WAIT_S = 0): the
     standing()/waiting() branches and standing_still are kept but inactive.
   * every inherited curriculum is removed (action rate, head range, head
     bias, CoM ranges, standing share): the warm-start checkpoint carries
     step counter 24024, which would have switched their late stages on at
     once. CoM DR fixed at +-5 mm (trunk and head).
5. Warm start from sideways v4 iter 1000 via `rl/human_bridge/rl_policy/warmstart_beam.py`: counter
   removed, critic first layer widened for the 18 new privileged inputs
   (zero weights; their normaliser mean / var measured on warm-start
   rollouts; the critic normaliser count lowered from 98M to 2M so it keeps
   learning), Adam state cleared. Episodes 30 s, gamma 0.995. Symmetry
   loss off (she always walks +x).
6. The foot height scan uses 5 rays across the sole (+-24 mm along her
   forward), not the recipe's ring of radius 40 mm: on a 28 mm beam the ring
   rays all miss the beam. max_distance 45 mm: a foot with no beam under it
   reads 45 mm, not ~300 mm (the height above z = 0), so a gap does not
   make its landing cost 400x a normal one.
7. Motors: the story's motor model (`STORY_MOTORS`). The story (world.py)
   runs BAM with no action delay, 7.4 V and sag gain 0.1, and reads the obs
   with no delay. The walking recipe trains with a 3-6 physics-step action
   delay (15-30 ms), 0-1 step obs delays and a random battery (6.5-8.2 V,
   sag 0-0.2). A/B on a flat 28 mm beam, warm-start actor, FF5 law, 32 envs,
   30 s (s4_fix/flat_ab2.py): recipe motors 8/32 crossed 0.39 m; story
   motors 32/32, at the story bench's speed (0.031 vs 0.032 m/s); action
   delay 0 alone 29/32. Story contact settings (solref 0.008, no sole
   priority) 6/32 and a 2.5 ms x 8 step 4/32 changed nothing. The rest of
   the DR (mass, CoM, friction, armature, encoder bias, IMU misalignment,
   obs noise) stays. With it, the warm start crosses 12/32 (19/32 without
   the obs-side part): training stays harder than the story on purpose.
   This policy is for the story sim: without the delay DR it is not meant
   for the real robot.
8. Measured in this sim (s3_train/diag_law.py): the foot site is AT the sole
   bottom (-0.3 mm median above the beam top in contact), not 10.3 mm above
   it as the sideways v4 notes say. So v4's swing target (0.030 at the site)
   asks for 30 mm of real lift, and its swing peaks (median 26.5 mm on these
   beams) are real lift. Both swing terms keep v4's values.
"""

from __future__ import annotations

import dataclasses
import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from microduck_lab.rl.human_bridge.rl_policy import beam_terrain, mdp_beam
from microduck_lab.rl.walking.microduck_velocity_sideways_env_cfg import (
    MicroduckSidewaysRlCfg,
    make_microduck_velocity_sideways_env_cfg,
)

# -- episode / terrain ---------------------------------------------------------
EPISODE_S = 30.0
MAX_INIT_LEVEL = 2
PROMOTE_M, DEMOTE_M = 0.8, 0.2
NCONMAX = 100          # contact pool = nconmax x envs. Measured peak use 4303 contacts over 2048 envs
                       # (s3_train/bench_step.py); 200 ran out of GPU memory at 4096 envs on an 11 GB GPU.
SOLVER_ITERATIONS, SOLVER_LS_ITERATIONS = 30, 50

# -- spawn ---------------------------------------------------------------------
P_START = 0.25
START_X = (0.12, 0.22)                 # trunk on the start platform (beam starts at 0.30)
SPAWN_Y = (-0.004, 0.004)
SPAWN_YAW = 0.05
STANCE_CLEARANCE = (0.121, 0.127)      # trunk above the surface under her stance

# -- command -------------------------------------------------------------------
SIDE_SPEED = (0.08, 0.14)
# No zero-command wait. In the story her standing policy runs while she waits
# (PolicyInference switches below |cmd| 0.05), so this walker only ever gets
# vy 0.11. A U(0, 12) s wait made the warm start walk off the start platform in
# 42% of those episodes (s4_verify), training a skill the story never uses.
START_WAIT_S = (0.0, 0.0)
RESAMPLE_S = (10.0, 20.0)

# -- rewards (per second, before dt scaling) -------------------------------------
PROGRESS_PER_M = 50.0
PROGRESS_SPEED_CAP = 1.25              # progress pays at most 1.25 x commanded vy
GATED = {"track_lateral_velocity": 2.0, "track_yaw_rate_tight": 1.0, "upright": 1.0,
         "pose": 0.5, "air_time": 2.0, "swing_lift": 1.0, "head_pose_tracking": 0.5}
STANDING_STILL = 2.0
OFFSET_COST, OFFSET_CAP = -3000.0, mdp_beam.OFFSET_CAP      # saturating: <= 3000 x 0.02^2 = 1.2 / s
HEADING_COST, HEADING_CAP = -10.0, mdp_beam.HEADING_CAP     # saturating: <= 10 x 0.25^2 = 0.625 / s
FAIL_COST = -500.0                     # x dt 0.02 = -10 per failure
ACTION_RATE = -0.4
# Foot height scan: a ray that hits nothing (a foot over a gap or beside the
# beam) reads this. At 0.10 each such landing cost 0.33 in swing height, 400x a
# normal landing (s5_review). 45 mm stays above the swing target (30 mm) and
# the lift range (10-25 mm), so normal steps read true.
SCAN_MAX_DISTANCE = 0.045
COM_RANGE = 0.005

# -- motors (see the docstring, item 7) -------------------------------------------
STORY_MOTORS = True
STORY_VIN = 7.4                        # world.BAM_VIN
STORY_SAG = 0.1                        # world.BAM_VIN_DROP_GAIN

# -- terminations -------------------------------------------------------------------
MIN_UP = 0.9                           # story.Judge.MIN_UP
FOOT_OFF_MARGIN = 0.015
STAND_Z, DROP_MARGIN = 0.120, 0.06
END_PAST = 0.10

INHERITED_CURRICULA = ("action_rate_weight", "standing_envs", "head_pose_range", "body_pose_range",
                       "com_range", "head_com_range", "head_pose_bias_weight")

BEAM_SOLID_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(bodyhit|footshell)_.*",),
    contype=0,
    conaffinity=mdp_beam.SOLID_CONAFFINITY,
    condim=3,
    disable_other_geoms=False,
)


def make_microduck_velocity_sideways_beam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_sideways_env_cfg(play=play, rough=False)
    cfg.episode_length_s = EPISODE_S

    # -- robot: her story-solid parts collide with the terrain (only) ----------
    robot = deepcopy(cfg.scene.entities["robot"])
    robot.spec_fn = mdp_beam.beam_walk_spec
    robot.collisions = (*robot.collisions, BEAM_SOLID_COLLISION)
    if STORY_MOTORS:
        robot.articulation.actuators = tuple(
            dataclasses.replace(a, delay_min_lag=0, delay_max_lag=0, vin_range=(STORY_VIN, STORY_VIN),
                                vin_drop_gain_range=(STORY_SAG, STORY_SAG))
            for a in robot.articulation.actuators)
        for term in cfg.observations["actor"].terms.values():
            if getattr(term, "delay_max_lag", 0):
                term.delay_min_lag = term.delay_max_lag = 0
    cfg.scene.entities = {"robot": robot}

    # -- terrain ----------------------------------------------------------------
    assert cfg.scene.spec_fn is None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = beam_terrain.make_beam_terrain_cfg()
    cfg.scene.terrain.max_init_terrain_level = MAX_INIT_LEVEL
    cfg.scene.spec_fn = mdp_beam.terrain_bits
    cfg.sim.nconmax = NCONMAX
    cfg.sim.mujoco.iterations = SOLVER_ITERATIONS
    cfg.sim.mujoco.ls_iterations = SOLVER_LS_ITERATIONS

    # -- sensors ------------------------------------------------------------------
    sensors = {s.name: s for s in cfg.scene.sensors}
    scan = sensors["foot_height_scan"]
    sensors["foot_height_scan"] = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=f.name, entity="robot") for f in scan.frame),
        pattern=GridPatternCfg(size=(0.048, 0.0), resolution=0.012),
        ray_alignment="yaw", max_distance=SCAN_MAX_DISTANCE, exclude_parent_body=True,
        include_geom_groups=(0,), debug_vis=False)
    sensors["body_terrain_contact"] = ContactSensorCfg(
        name="body_terrain_contact",
        primary=ContactMatch(mode="geom", pattern=r"^bodyhit_.*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1)
    cfg.scene.sensors = tuple(sensors.values())

    # -- command --------------------------------------------------------------------
    old = cfg.commands["twist"]
    keep = {k: v for k, v in vars(old).items() if k in mdp_beam.BeamSteeringCommandCfg.__dataclass_fields__}
    twist = mdp_beam.BeamSteeringCommandCfg(**keep)
    twist.vy_range = SIDE_SPEED
    twist.rel_standing_envs = 0.0             # zero commands only while waiting on the start platform
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.heading_command = False
    twist.resampling_time_range = RESAMPLE_S
    twist.ranges = deepcopy(old.ranges)
    twist.ranges.lin_vel_x = (-twist.vx_max, twist.vx_max)
    twist.ranges.lin_vel_y = (0.0, SIDE_SPEED[1])
    twist.ranges.ang_vel_z = (-twist.wz_max, twist.wz_max)
    twist.ranges.heading = None
    cfg.commands["twist"] = twist

    # -- events -----------------------------------------------------------------------
    cfg.events["reset_base"] = EventTermCfg(
        func=mdp_beam.reset_on_beam, mode="reset",
        params={"p_start": P_START, "start_x": START_X, "y": SPAWN_Y,
                "yaw": (mdp_beam.HEADING - SPAWN_YAW, mdp_beam.HEADING + SPAWN_YAW),
                "clearance": STANCE_CLEARANCE, "wait_s": START_WAIT_S})
    cfg.events.pop("push_robot", None)       # a push on a beam is a push off it
    for name in ("randomize_com", "randomize_head_com"):
        if name in cfg.events:
            cfg.events[name].params["ranges"] = (-COM_RANGE, COM_RANGE)

    # -- curricula: none inherited; one terrain curriculum ---------------------------
    for name in INHERITED_CURRICULA:
        cfg.curriculum.pop(name, None)
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=mdp_beam.terrain_levels_beam, params={"promote_m": PROMOTE_M, "demote_m": DEMOTE_M})

    # -- observations: critic only gets the beam ---------------------------------------
    cfg.observations["critic"].terms["beam"] = ObservationTermCfg(func=mdp_beam.beam_privileged)

    # -- rewards ---------------------------------------------------------------------------
    r = cfg.rewards
    for name in ("track_linear_velocity", "track_angular_velocity"):
        r.pop(name)
    for name, weight in GATED.items():
        term = r[name]
        r[name] = RewardTermCfg(func=mdp_beam.gated, weight=weight,
                                params={"inner": term.func, "inner_params": dict(term.params)})
    r["progress"] = RewardTermCfg(func=mdp_beam.progress_ratchet, weight=PROGRESS_PER_M,
                                  params={"speed_cap": PROGRESS_SPEED_CAP})
    r["standing_still"] = RewardTermCfg(func=mdp_beam.standing_still, weight=STANDING_STILL)
    r["beam_offset"] = RewardTermCfg(func=mdp_beam.beam_offset_sq, weight=OFFSET_COST, params={"cap": OFFSET_CAP})
    r["heading"] = RewardTermCfg(func=mdp_beam.heading_err_sq, weight=HEADING_COST, params={"cap": HEADING_CAP})
    r["failed"] = RewardTermCfg(func=mdp_beam.failed, weight=FAIL_COST)
    r["action_rate_l2"].weight = ACTION_RATE
    r["head_pose_bias"].weight = 0.0
    r["body_pose_tracking"].weight = 0.0

    # -- terminations ------------------------------------------------------------------------
    t = cfg.terminations
    t.pop("fell_over", None)
    # mjlab's bound check keeps 0.3 m off the grid edge; the first/last rows'
    # platforms and the outer columns' beams sit on that edge (smoke test 1:
    # mean episode 2.85 steps, nearly all ended by it). The beam terminations
    # below cover leaving the beam.
    t.pop("out_of_terrain_bounds", None)
    t["tilted"] = TerminationTermCfg(func=mjlab_mdp.bad_orientation,
                                     params={"limit_angle": math.acos(MIN_UP)})
    t["body_contact"] = TerminationTermCfg(func=mdp_beam.body_contact,
                                           params={"sensor_name": "body_terrain_contact"})
    t["foot_off"] = TerminationTermCfg(
        func=mdp_beam.foot_off,
        params={"margin": FOOT_OFF_MARGIN,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))})
    t["dropped"] = TerminationTermCfg(func=mdp_beam.dropped,
                                      params={"stand_z": STAND_Z, "margin": DROP_MARGIN})
    t["reached_end"] = TerminationTermCfg(func=mdp_beam.reached_end, params={"past": END_PAST},
                                          time_out=True)
    return cfg


MicroduckSidewaysBeamRlCfg = deepcopy(MicroduckSidewaysRlCfg)
MicroduckSidewaysBeamRlCfg.experiment_name = "velocity_sideways_beam"
MicroduckSidewaysBeamRlCfg.run_name = "sideways_beam"
MicroduckSidewaysBeamRlCfg.max_iterations = 3000
MicroduckSidewaysBeamRlCfg.save_interval = 250
MicroduckSidewaysBeamRlCfg.algorithm.gamma = 0.995
MicroduckSidewaysBeamRlCfg.algorithm.learning_rate = 3.0e-4
# Always +x along the beam: the mirror image (walking -x) never happens.
MicroduckSidewaysBeamRlCfg.algorithm.symmetry_cfg = None
