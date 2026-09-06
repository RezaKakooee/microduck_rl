"""Skate-blade variant invariants.

The load-bearing facts: which pair-friction slot maps to which world axis for
a plane contact (the frame is world-fixed, not foot-fixed), that the spec_fn
really adds the pairs on the real mjlab scene, that the per-step projection
writes what the maths says, and that the shared 61D obs contract survives.
"""

import math

import mujoco
import numpy as np
import pytest
import torch

from mjlab.scene import Scene
from microduck_lab.rl import mdp_blades
from microduck_lab.rl.mdp_blades import (
    PAIR_AXIS_WORLD_X,
    PAIR_AXIS_WORLD_Y,
    blade_axis_radii,
    blade_projection,
    blade_radius,
    get_blade_state,
    project_blade_friction,
    randomize_blade_friction,
    sole_long_axis_in_body,
)
from microduck_lab.rl.microduck_velocity_blades_env_cfg import (
    BLADE_PAIR_NAMES,
    BLADE_RULE,
    BLADE_STAGES,
    FOOT_CONTACT_SENSOR,
    FOOT_GEOM_NAMES,
    _add_blade_pairs,
    make_microduck_velocity_blades_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

WALK_SCENE = "src/mjlab_microduck/robot/microduck/scene.xml"
BLADES_SCENE = "src/microduck_lab/models/scene_blades.xml"


def _repo_path(rel):
    from microduck_lab.paths import REPO

    return str(REPO / rel)


# ── 1. The anisotropy direction ──────────────────────────────────────────────


def _box_slide(s1, s2, axis):
    """Slide a 1 kg box at 2 m/s along a world axis for 3 s; return distance."""
    xml = f"""<mujoco><option timestep="0.005"/><worldbody>
      <geom name="floor" type="plane" size="0 0 .01"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom name="box" type="box" size=".05 .05 .05" mass="1"/>
      </body></worldbody>
      <contact><pair geom1="box" geom2="floor" condim="4"
        friction="{s1} {s2} 0.005 0.0001 0.0001"/></contact></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    d.qvel[axis] = 2.0
    for _ in range(600):
        mujoco.mj_step(m, d)
    return float(d.qpos[axis])


def test_pair_slides_far_along_one_tangent_and_short_across():
    """The verification from the handoff: two different sliding coefficients."""
    far, short = _box_slide(0.02, 1.2, 1), _box_slide(0.02, 1.2, 0)
    assert far > 10 * short, (far, short)


def test_pair_slot_0_is_world_y_and_slot_1_is_world_x():
    """Locks the measured mapping the projection relies on. If MuJoCo ever
    changes how it builds a plane's contact frame, this is what fails."""
    # friction[0] low -> slides along world Y, sticks along X.
    assert _box_slide(0.02, 1.2, 1) > 10 * _box_slide(0.02, 1.2, 0)
    # friction[1] low -> slides along world X, sticks along Y.
    assert _box_slide(1.2, 0.02, 0) > 10 * _box_slide(1.2, 0.02, 1)
    assert PAIR_AXIS_WORLD_Y == 0 and PAIR_AXIS_WORLD_X == 1


def test_plane_contact_frame_is_world_fixed_for_the_duck_foot():
    """t1 = +-Y, t2 = +-X for the sole on the floor, whatever the pose."""
    m = mujoco.MjModel.from_xml_path(_repo_path(WALK_SCENE))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = m.key_ctrl[key]
    for _ in range(20):
        mujoco.mj_step(m, d)
    feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES}
    seen = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 not in feet and c.geom2 not in feet:
            continue
        frame = np.array(c.frame).reshape(3, 3)
        np.testing.assert_allclose(np.abs(frame[0]), [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(np.abs(frame[1]), [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(np.abs(frame[2]), [1, 0, 0], atol=1e-6)
        seen += 1
    assert seen > 0, "no foot-floor contact found"


def test_sole_long_axis_is_fore_aft_at_zero_yaw():
    """The blade runs along the sole's long axis, which is world X when the
    duck faces +X. 54 mm long, 41 mm wide, 13 mm thick (measured)."""
    m = mujoco.MjModel.from_xml_path(_repo_path(WALK_SCENE))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    for name in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        axis_b = sole_long_axis_in_body(m, gid)
        rot = d.xmat[m.geom_bodyid[gid]].reshape(3, 3)
        axis_w = rot @ axis_b
        assert abs(axis_w[0]) > 0.9, (name, axis_w)


# ── 2. The projection maths ──────────────────────────────────────────────────


def test_projection_is_exact_at_0_and_90_deg_and_isotropic_at_45():
    a, c = torch.tensor(0.05), torch.tensor(1.0)
    mx, my = blade_projection(a, c, torch.tensor(1.0))  # yaw 0
    assert math.isclose(mx.item(), 0.05, rel_tol=1e-6) and math.isclose(my.item(), 1.0, rel_tol=1e-6)
    mx, my = blade_projection(a, c, torch.tensor(0.0))  # yaw 90
    assert math.isclose(mx.item(), 1.0, rel_tol=1e-6) and math.isclose(my.item(), 0.05, rel_tol=1e-6)
    mx, my = blade_projection(a, c, torch.tensor(0.5))  # yaw 45: the known weak spot
    iso = math.sqrt((0.05**2 + 1.0**2) / 2)
    assert math.isclose(mx.item(), iso, rel_tol=1e-6) and math.isclose(my.item(), iso, rel_tol=1e-6)


def test_load_rule_radius_matches_the_true_blade_ellipse():
    a, c = torch.tensor(0.08), torch.tensor(1.0)
    # Pushing along the blade: glide. Perpendicular: grip.
    assert math.isclose(blade_radius(a, c, torch.tensor(1.0)).item(), 0.08, rel_tol=1e-6)
    assert math.isclose(blade_radius(a, c, torch.tensor(0.0)).item(), 1.0, rel_tol=1e-6)
    # Foot turned out 30 deg, shoved straight back: (d.u)^2 = cos^2(30) = 0.75.
    r = blade_radius(a, c, torch.tensor(0.75)).item()
    assert math.isclose(r, 1 / math.sqrt(0.75 / 0.08**2 + 0.25 / 1.0), rel_tol=1e-6)
    assert r < 0.1, "a real blade gives a turned-out straight-back push almost nothing"
    # ... whereas the bounding box would hand it ~0.5.
    assert blade_projection(a, c, torch.tensor(0.75))[0].item() > 0.45
    # Unloaded intercepts: exact at 0 / 90 deg.
    rx, ry = blade_axis_radii(a, c, torch.tensor(1.0))
    assert math.isclose(rx.item(), 0.08, rel_tol=1e-6) and math.isclose(ry.item(), 1.0, rel_tol=1e-6)
    rx, ry = blade_axis_radii(a, c, torch.tensor(0.0))
    assert math.isclose(rx.item(), 1.0, rel_tol=1e-6) and math.isclose(ry.item(), 0.08, rel_tol=1e-6)


# ── 3. The spec_fn on the real mjlab scene ───────────────────────────────────


@pytest.fixture(scope="module")
def blades_scene():
    cfg = make_microduck_velocity_blades_env_cfg()
    cfg.scene.num_envs = 2
    scene = Scene(cfg.scene, "cpu")
    return scene, scene.compile()


def test_spec_fn_adds_one_condim4_pair_per_foot_on_the_real_scene(blades_scene):
    scene, m = blades_scene
    assert m.npair == 2
    terrain = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    for name, foot in zip(BLADE_PAIR_NAMES, FOOT_GEOM_NAMES):
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, name)
        assert pid >= 0
        gids = {int(m.pair_geom1[pid]), int(m.pair_geom2[pid])}
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{foot}") in gids
        assert terrain in gids
        assert m.pair_dim[pid] == 4


def test_spec_fn_works_on_a_bare_unprefixed_spec():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    body.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01))
    foot = spec.worldbody.add_body(name="feet", pos=(0, 0, 0.1))
    foot.add_freejoint()
    for name in FOOT_GEOM_NAMES:
        foot.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.02, 0.02, 0.02))
    _add_blade_pairs(spec)
    m = spec.compile()
    assert m.npair == 2 and list(m.pair_dim) == [4, 4]


# ── 4. The events write the model ────────────────────────────────────────────


class _Data:
    def __init__(self, quat):
        self.body_link_quat_w = quat


class _Asset:
    def __init__(self, scene_entity, quat):
        self.body_names = scene_entity.body_names
        self.data = _Data(quat)


class _Sim:
    def __init__(self, mj_model, num_envs):
        self.mj_model = mj_model
        pf = torch.as_tensor(np.array(mj_model.pair_friction), dtype=torch.float32)
        self.model = type("M", (), {})()
        self.model.pair_friction = pf.unsqueeze(0).repeat(num_envs, 1, 1)


class _Sensor:
    def __init__(self, num_envs, nfeet=2):
        self.data = type("D", (), {})()
        self.data.force = torch.zeros(num_envs, nfeet, 3)


class _FakeEnv:
    def __init__(self, scene, mj_model, num_envs, quat):
        self.num_envs = num_envs
        self.device = "cpu"
        self.sim = _Sim(mj_model, num_envs)
        self._asset = _Asset(scene["robot"], quat)
        self.sensor = _Sensor(num_envs)

    @property
    def scene(self):
        return {"robot": self._asset, FOOT_CONTACT_SENSOR: self.sensor}


def _yaw_quat(yaw):
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_projection_writes_along_to_world_x_at_zero_yaw_and_swaps_at_90(blades_scene):
    scene, m = blades_scene
    nbody = len(scene["robot"].body_names)
    # env 0: all bodies at yaw 0; env 1: all bodies at yaw 90 deg.
    quat = torch.zeros(2, nbody, 4)
    quat[0] = _yaw_quat(0.0)
    quat[1] = _yaw_quat(math.pi / 2)
    env = _FakeEnv(scene, m, 2, quat)
    state = get_blade_state(env, BLADE_PAIR_NAMES)
    state.mu_along[:] = 0.05
    state.mu_across[:] = 1.0
    # The body-frame sole axis is not world X at identity body orientation:
    # feed the projection the world yaw directly by rotating the axis instead.
    # (In the real env body_link_quat_w carries the full orientation.)
    axis_yaw = torch.atan2(state.axis_b[:, 1], state.axis_b[:, 0])  # (F,)
    quat[0, state.body_ids] = torch.stack([_yaw_quat(-a.item()) for a in axis_yaw])
    quat[1, state.body_ids] = torch.stack([_yaw_quat(math.pi / 2 - a.item()) for a in axis_yaw])
    # Only meaningful if the sole axis lies in the body XY plane; assert that.
    assert torch.all(state.axis_b[:, 2].abs() < 0.3), state.axis_b
    project_blade_friction(env, None, BLADE_PAIR_NAMES, rule="bbox")
    pf = env.sim.model.pair_friction
    for pid in state.pair_ids.tolist():
        assert abs(pf[0, pid, PAIR_AXIS_WORLD_X].item() - 0.05) < 0.02
        assert abs(pf[0, pid, PAIR_AXIS_WORLD_Y].item() - 1.0) < 0.02
        assert abs(pf[1, pid, PAIR_AXIS_WORLD_X].item() - 1.0) < 0.02
        assert abs(pf[1, pid, PAIR_AXIS_WORLD_Y].item() - 0.05) < 0.02
        # spin / roll untouched
        assert pf[0, pid, 2].item() == pytest.approx(0.005)


def _env_with_feet_at_yaw(scene, m, yaws_deg):
    """Fake env whose feet (both) point at the given world yaw per env."""
    nbody = len(scene["robot"].body_names)
    quat = _yaw_quat(0.0).expand(len(yaws_deg), nbody, 4).clone()
    env = _FakeEnv(scene, m, len(yaws_deg), quat)
    state = get_blade_state(env, BLADE_PAIR_NAMES)
    axis_yaw = torch.atan2(state.axis_b[:, 1], state.axis_b[:, 0])
    for i, yaw in enumerate(yaws_deg):
        quat[i, state.body_ids] = torch.stack(
            [_yaw_quat(math.radians(yaw) - a.item()) for a in axis_yaw]
        )
    state.mu_along[:] = 0.08
    state.mu_across[:] = 1.0
    return env, state


def test_load_rule_grips_across_glides_along_and_refuses_the_turned_out_push(blades_scene):
    scene, m = blades_scene
    # env 0: feet straight, unloaded. env 1: feet at 30 deg, shoved straight
    # back (-X). env 2: feet at 30 deg, pushed perpendicular to the blade.
    # env 3: feet straight, sliding along the blade.
    env, state = _env_with_feet_at_yaw(scene, m, [0.0, 30.0, 30.0, 0.0])
    f = env.sensor.data.force
    f[1, :, 0] = -2.0
    perp = torch.tensor([-math.sin(math.radians(30)), math.cos(math.radians(30))])
    f[2, :, :2] = 3.0 * perp
    f[3, :, 0] = 1.0
    project_blade_friction(env, None, BLADE_PAIR_NAMES, rule="load", sensor_name=FOOT_CONTACT_SENSOR)
    pf = env.sim.model.pair_friction
    for pid in state.pair_ids.tolist():
        # unloaded straight foot: intercepts (along on X, across on Y)
        assert pf[0, pid, PAIR_AXIS_WORLD_X].item() == pytest.approx(0.08, abs=0.01)
        assert pf[0, pid, PAIR_AXIS_WORLD_Y].item() == pytest.approx(1.0, abs=0.02)
        # turned-out straight-back push: nearly nothing, on both axes
        assert pf[1, pid, PAIR_AXIS_WORLD_X].item() < 0.12
        assert pf[1, pid, PAIR_AXIS_WORLD_Y].item() < 0.12
        # perpendicular push: full grip
        assert pf[2, pid, PAIR_AXIS_WORLD_X].item() > 0.95
        # sliding along the blade: glide
        assert pf[3, pid, PAIR_AXIS_WORLD_X].item() == pytest.approx(0.08, abs=0.01)


def test_reset_event_samples_inside_the_ranges_and_only_for_given_envs(blades_scene):
    scene, m = blades_scene
    nbody = len(scene["robot"].body_names)
    quat = _yaw_quat(0.0).expand(4, nbody, 4).clone()
    env = _FakeEnv(scene, m, 4, quat)
    state = get_blade_state(env, BLADE_PAIR_NAMES)
    state.mu_along[:] = -1.0
    randomize_blade_friction(env, torch.tensor([1, 3]), (0.1, 0.2), (0.9, 1.1), BLADE_PAIR_NAMES)
    assert state.mu_along[1] >= 0.1 and state.mu_along[1] <= 0.2
    assert state.mu_along[3] >= 0.1 and state.mu_along[3] <= 0.2
    assert state.mu_along[0] == -1.0 and state.mu_along[2] == -1.0
    assert 0.9 <= state.mu_across[1] <= 1.1


def test_events_declare_the_pair_friction_field():
    """Without this mjlab never expands pair_friction per world and every env
    would share one set of coefficients."""
    assert "pair_friction" in randomize_blade_friction.model_fields
    assert "pair_friction" in project_blade_friction.model_fields


# ── 5. Cfg wiring ────────────────────────────────────────────────────────────


class _EventCfg:
    def __init__(self, params):
        self.params = params


class _EventManager:
    def __init__(self, cfg):
        self._cfg = cfg

    def get_term_cfg(self, name):
        assert name == "blade_friction"
        return self._cfg


class _CurEnv:
    def __init__(self, step, cfg):
        self.common_step_counter = step
        self.event_manager = _EventManager(cfg)


def test_curriculum_ramps_along_down_and_keeps_across_grippy():
    stages = BLADE_STAGES
    assert stages[0]["along"] == stages[0]["across"], "stage 0 must be isotropic walking"
    assert stages[0]["along"][0] >= 0.5
    for a, b in zip(stages, stages[1:]):
        assert b["step"] > a["step"]
        assert b["along"][0] <= a["along"][0] and b["along"][1] <= a["along"][1]
        assert b["across"][0] >= 0.7
    assert stages[-1]["along"][1] < stages[-1]["across"][0]
    cfg = _EventCfg({"along_range": None, "across_range": None})
    mdp_blades.blade_friction_curriculum(_CurEnv(0, cfg), None, "blade_friction", stages)
    assert cfg.params["along_range"] == stages[0]["along"]
    mdp_blades.blade_friction_curriculum(_CurEnv(stages[-1]["step"] + 1, cfg), None, "blade_friction", stages)
    assert cfg.params["along_range"] == stages[-1]["along"]
    assert cfg.params["across_range"] == stages[-1]["across"]


def test_cfg_replaces_geom_friction_dr_with_blade_events():
    cfg = make_microduck_velocity_blades_env_cfg()
    assert "foot_friction" not in cfg.events, "geom DR is a no-op under pairs"
    assert cfg.events["blade_friction"].mode == "reset"
    assert cfg.events["blade_friction"].params["along_range"] == BLADE_STAGES[0]["along"]
    assert cfg.events["blade_projection"].mode == "step"
    for name in ("blade_friction", "blade_projection"):
        assert cfg.events[name].params["rule"] == BLADE_RULE == "load"
        assert cfg.events[name].params["sensor_name"] == FOOT_CONTACT_SENSOR
    assert FOOT_CONTACT_SENSOR in {s.name for s in cfg.scene.sensors}
    assert cfg.curriculum["blade_friction"].params["event_name"] == "blade_friction"
    assert cfg.scene.spec_fn is not None


def test_heading_is_held_and_turn_in_place_is_off():
    cfg = make_microduck_velocity_blades_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.heading_command is True
    assert twist.rel_heading_envs == 1.0
    assert twist.ranges.heading is not None and max(abs(h) for h in twist.ranges.heading) <= 0.5
    assert twist.rel_turn_in_place_envs == 0.0
    yaw = cfg.events["reset_base"].params["pose_range"]["yaw"]
    assert max(abs(y) for y in yaw) <= 0.5
    # Command slots stay alive.
    assert twist.ranges.lin_vel_y[1] > 0 and twist.ranges.ang_vel_z[1] > 0


def test_slip_untaxed_but_air_time_keeps_its_walking_weight():
    blades = make_microduck_velocity_blades_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    assert blades.rewards["foot_slip"].weight == 0.0
    assert walk.rewards["foot_slip"].weight < 0.0
    assert blades.rewards["air_time"].weight == walk.rewards["air_time"].weight


def test_rough_is_refused_loudly():
    with pytest.raises(NotImplementedError):
        make_microduck_velocity_blades_env_cfg(rough=True)


def test_blades_do_not_disturb_the_shared_observation_contract():
    """Every policy in this family must stay hot-swappable: 61D actor obs."""
    blades = make_microduck_velocity_blades_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    assert list(blades.observations["actor"].terms.keys()) == list(
        walk.observations["actor"].terms.keys()
    )
    assert list(blades.observations["critic"].terms.keys()) == list(
        walk.observations["critic"].terms.keys()
    )


# ── 6. CPU playback scene ────────────────────────────────────────────────────


def test_cpu_playback_scene_loads_with_two_blade_pairs():
    m = mujoco.MjModel.from_xml_path(_repo_path(BLADES_SCENE))
    assert m.npair == 2 and list(m.pair_dim) == [4, 4]
    for pid in range(m.npair):
        assert m.pair_friction[pid, PAIR_AXIS_WORLD_Y] > m.pair_friction[pid, PAIR_AXIS_WORLD_X]
