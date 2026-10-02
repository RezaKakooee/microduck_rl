"""Trampoline flip env invariants (CPU, no simulation of the env itself).

The ones that matter: the physics step the bed needs (1 ms) with the BAM lag
scaled to it, the actor observation kept identical to the roulade's (the 61-D
contract), the sensors the MDP terms read by name, the penalty signs, and the
bed/feet contact wiring.
"""

import math

import mujoco
import numpy as np

from microduck_lab.rl import mdp_trampoline as tmdp
from microduck_lab.rl import trampoline_scene as T
from microduck_lab.rl.microduck_trampoline_flip_env_cfg import make_microduck_trampoline_flip_env_cfg
from mjlab_microduck.tasks.microduck_roulade_env_cfg import make_microduck_roulade_env_cfg


def test_step_and_bam_lag():
    cfg = make_microduck_trampoline_flip_env_cfg()
    assert cfg.sim.mujoco.timestep == 0.001
    assert cfg.decimation == 20                      # 50 Hz policy
    act = T.TRAMPOLINE_ROBOT_CFG.articulation.actuators[0]
    assert (act.delay_min_lag, act.delay_max_lag) == (15, 30)   # 15-30 ms, as the 5 ms envs' 3-6 steps


def test_actor_observation_matches_the_roulade():
    ours = make_microduck_trampoline_flip_env_cfg().observations["actor"].terms
    ref = make_microduck_roulade_env_cfg().observations["actor"].terms
    assert list(ours) == list(ref)
    for name in ours:
        assert ours[name].func is ref[name].func, name


def test_sensors_the_terms_read_exist():
    names = {s.name for s in make_microduck_trampoline_flip_env_cfg().scene.sensors}
    assert {tmdp.FEET_SENSOR, tmdp.BODY_SENSOR, tmdp.FLOOR_SENSOR} <= names


def test_penalty_signs():
    rewards = make_microduck_trampoline_flip_env_cfg().rewards
    for name, term in rewards.items():
        if term.func.__name__.endswith("_cost"):
            assert term.weight <= 0, name
    for name in ("flip_progress", "height_progress", "upright_after_flip"):
        assert rewards[name].weight > 0, name
    assert "upright" not in rewards      # an always-on upright term would oppose the flip


def test_bed_sags_80_mm_under_the_robot():
    m = T.get_bed_spec().compile()
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bed_slide")
    sag = T.ROBOT_MASS * 9.81 / m.jnt_stiffness[j]
    assert abs(sag - 0.080) < 1e-6
    assert m.body_gravcomp[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bed")] == 1.0
    # The explicit spring stays well inside one period of the empty bed.
    assert math.sqrt(m.jnt_stiffness[j] / T.BED_MASS) * T.TIMESTEP < 0.5


def test_feet_touch_the_bed_only_through_the_boxes():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    frame = spec.worldbody.add_frame()
    frame.attach_body(T.get_trampoline_robot_spec().body("trunk_base"), "robot/", "")
    frame.attach_body(T.get_bed_spec().body("bed"), "bed/", "")
    T.add_bed_contacts(spec)
    m = spec.compile()
    pairs = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(m.pair_geom1[i])) for i in range(m.npair)}
    assert pairs == {"robot/left_foot_box", "robot/right_foot_box"}
    assert np.allclose(m.pair_solref[:, 0], 0.002)   # stiff: a soft contact adds energy to the bed
    assert m.nexclude == 3
    for name in T.FOOT_BOX_NAMES:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}")
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def test_pump_flip_variant_waits_for_bounces_and_starts_standing():
    from microduck_lab.rl.microduck_trampoline_flip_env_cfg import MIN_BOUNCES
    cfg = make_microduck_trampoline_flip_env_cfg(pump_first=True)
    reset = cfg.events["reset_trampoline_state"].params
    assert reset["min_bounces"] == MIN_BOUNCES >= 2
    assert reset["stand_prob"] >= 0.5             # mostly standing starts
    assert reset["midflip_prob"] > 0               # the landing stays practised
    assert cfg.rewards["bounce_progress"].weight > 0
    assert cfg.rewards["fall"].weight < 0          # a crash must cost more than standing still
    assert cfg.rewards["fall"].params["only_before_flip"]   # but not tax the flip attempts themselves
    assert reset["per_flight"]                     # small tilts on many bounces must not add up to a "flip"
    assert cfg.episode_length_s >= 8.0
    base = make_microduck_trampoline_flip_env_cfg()
    assert "bounce_progress" not in base.rewards   # the original flip task is unchanged
    assert base.events["reset_trampoline_state"].params.get("min_bounces", 0) == 0


def test_pump_curriculum_starts_at_the_working_flip_and_ends_standing():
    from microduck_lab.rl.microduck_trampoline_flip_env_cfg import MIN_BOUNCES, PUMP_CURRICULUM
    cfg = make_microduck_trampoline_flip_env_cfg(pump_first=True, curriculum=True)
    reset = cfg.events["reset_trampoline_state"].params
    assert reset["stand_prob"] == 0.0 and reset["min_bounces"] == 1      # stage 0: drops, as run 2
    assert [st["step"] for st in cfg.curriculum["spawn_mix"].params["param_stages"]] == [st["step"] for st in PUMP_CURRICULUM]
    slow = make_microduck_trampoline_flip_env_cfg(pump_first=True, curriculum=True, stretch=2.0)
    assert [st["step"] for st in slow.curriculum["spawn_mix"].params["param_stages"]] == [2 * st["step"] for st in PUMP_CURRICULUM]
    last = PUMP_CURRICULUM[-1]["params"]
    assert last["min_bounces"] == MIN_BOUNCES and last["stand_prob"] >= 0.5
    steps = [st["step"] for st in PUMP_CURRICULUM]
    assert steps == sorted(steps)


def test_relative_curriculum_restarts_its_clock_after_a_checkpoint_restore():
    import types
    from microduck_lab.rl.microduck_trampoline_flip_env_cfg import PUMP_CURRICULUM
    params = {}
    term = types.SimpleNamespace(params=params)
    env = types.SimpleNamespace(common_step_counter=0,
                                event_manager=types.SimpleNamespace(get_term_cfg=lambda name: term))
    stage = lambda: int(tmdp.relative_event_param_curriculum(env, None, "x", PUMP_CURRICULUM))
    assert stage() == 0
    env.common_step_counter = 4749 * 24            # the restore of a resumed run
    assert stage() == 0 and params["min_bounces"] == 1
    env.common_step_counter += 700 * 24 // 50      # small steps forward, as in training
    for _ in range(50):
        stage()
        env.common_step_counter += 700 * 24 // 50
    assert stage() == 1 and params["min_bounces"] == 2


def test_bounce_task_pays_every_bounce_and_has_no_flip():
    from microduck_lab.rl.microduck_trampoline_flip_env_cfg import make_microduck_trampoline_bounce_env_cfg
    cfg = make_microduck_trampoline_bounce_env_cfg()
    assert "flip_progress" not in cfg.rewards and "upright_after_flip" not in cfg.rewards
    assert cfg.rewards["bounce_height"].weight > 0 and cfg.rewards["upright"].weight > 0
    assert cfg.rewards["fall"].weight < 0 and not cfg.rewards["fall"].params["only_before_flip"]
    reset = cfg.events["reset_trampoline_state"].params
    assert reset["stand_prob"] >= 0.5 and reset["midflip_prob"] == 0.0
    assert "spawn_mix" not in cfg.curriculum


def test_bounce_pose_variant_pays_for_the_standing_pose_at_the_top():
    from microduck_lab.rl.microduck_trampoline_flip_env_cfg import (
        BOUNCE_POSE_STD, make_microduck_trampoline_bounce_env_cfg)
    assert make_microduck_trampoline_bounce_env_cfg().rewards["bounce_height"].params["pose_std"] is None
    cfg = make_microduck_trampoline_bounce_env_cfg(target=0.30, pose_std=BOUNCE_POSE_STD)
    params = cfg.rewards["bounce_height"].params
    assert params["target"] == 0.30 and params["pose_std"] == BOUNCE_POSE_STD
    # bounce2's top pose (sum of squared joint errors ~5.7 rad^2) must still score visibly
    assert 0.1 < np.exp(-5.7 / BOUNCE_POSE_STD ** 2) < 0.5


def test_flip_anypose_widens_only_the_start_pose_and_tilt():
    from microduck_lab.rl.microduck_trampoline_flip_env_cfg import (
        FLIP_START_POSE_NOISE, make_microduck_trampoline_flip_anypose_env_cfg)
    base = make_microduck_trampoline_flip_env_cfg().events["reset_trampoline_state"].params
    wide = make_microduck_trampoline_flip_anypose_env_cfg().events["reset_trampoline_state"].params
    assert wide["joint_noise_std"] == FLIP_START_POSE_NOISE >= 0.25    # covers the bouncer's ~0.4-0.5 rad offsets
    assert wide["tilt_max"] > math.radians(3.0)
    assert {k: v for k, v in wide.items() if k not in ("joint_noise_std", "tilt_max")} == base
