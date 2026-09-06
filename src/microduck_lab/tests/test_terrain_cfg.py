"""Slopes + stairs variant invariants (CPU, no GPU needed).

Locks in: the terrain mix, the slope / step ranges the task was designed
around, the NaN guard inherited from the rough branch, the curriculum
wiring, the shared 61D observation contract, and — the one that would hurt
most silently — that every terrain spawn origin sits ON the terrain surface,
including the inverted (pit) types, so a reset never puts the robot below the
floor.
"""

import math

import mujoco
import numpy as np
import torch

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGenerator

from microduck_lab.rl import mdp_terrain
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)
from microduck_lab.rl.microduck_velocity_terrain_env_cfg import (
    FOOT_SWING_TARGET,
    HfFrustumSlopeTerrainCfg,
    PLATFORM_WIDTH,
    MAX_INIT_LEVEL,
    NUM_LEVELS,
    SLOPE_DEG_RANGE,
    STEP_HEIGHT_RANGE,
    STEP_RUN,
    TERRAIN_PROPORTIONS,
    make_microduck_velocity_terrain_env_cfg,
    make_slopes_terrain_cfg,
)

EXPECTED_TYPES = ("flat", "slope_down", "slope_up", "stairs_down", "stairs_up")


def test_terrain_types_present_in_curriculum_mode():
    cfg = make_microduck_velocity_terrain_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    assert gen.curriculum is True
    assert tuple(gen.sub_terrains.keys()) == EXPECTED_TYPES
    assert gen.num_rows == NUM_LEVELS
    assert set(TERRAIN_PROPORTIONS) == set(EXPECTED_TYPES)
    assert abs(sum(TERRAIN_PROPORTIONS.values()) - 1.0) < 1e-9


def test_slope_angles_within_design_range():
    gen = make_slopes_terrain_cfg()
    lo, hi = SLOPE_DEG_RANGE
    assert 5.0 <= lo < hi <= 15.0
    for name in ("slope_down", "slope_up"):
        st = gen.sub_terrains[name]
        assert isinstance(st, HfFrustumSlopeTerrainCfg)
        deg = tuple(math.degrees(math.atan(g)) for g in st.slope_range)
        assert abs(deg[0] - lo) < 1e-6 and abs(deg[1] - hi) < 1e-6
    assert gen.sub_terrains["slope_down"].inverted is False
    assert gen.sub_terrains["slope_up"].inverted is True


def test_step_heights_within_design_range():
    gen = make_slopes_terrain_cfg()
    lo, hi = STEP_HEIGHT_RANGE
    # 10-20 mm: the measured swing of the walking gait is ~15 mm.
    assert 0.010 <= lo < hi <= 0.020
    for name in ("stairs_down", "stairs_up"):
        st = gen.sub_terrains[name]
        assert isinstance(st, terrain_gen.BoxPyramidStairsTerrainCfg)
        assert st.step_height_range == STEP_HEIGHT_RANGE
        assert st.step_width == STEP_RUN
    assert isinstance(gen.sub_terrains["stairs_up"], terrain_gen.BoxInvertedPyramidStairsTerrainCfg)
    assert not isinstance(gen.sub_terrains["stairs_down"], terrain_gen.BoxInvertedPyramidStairsTerrainCfg)


def test_swing_target_clears_the_tallest_step():
    cfg = make_microduck_velocity_terrain_env_cfg()
    assert cfg.rewards["foot_swing_height"].params["target_height"] == FOOT_SWING_TARGET
    assert FOOT_SWING_TARGET > STEP_HEIGHT_RANGE[1]


def test_rough_physics_guards_are_kept():
    """The contact-softening spec_fn is the NaN guard for box terrain."""
    cfg = make_microduck_velocity_terrain_env_cfg()
    assert cfg.scene.spec_fn is _soften_terrain_contacts
    rough = make_microduck_velocity_env_cfg(rough=True)
    assert cfg.sim.nconmax == rough.sim.nconmax
    assert cfg.sim.mujoco.iterations == rough.sim.mujoco.iterations


def test_curriculum_wiring():
    cfg = make_microduck_velocity_terrain_env_cfg()
    term = cfg.curriculum["terrain_levels"]
    assert term.func is mdp_terrain.terrain_levels_walk
    assert term.params["command_name"] == "twist"
    assert 0 < term.params["demote_fraction"] < term.params["promote_fraction"] < 0.5
    assert 0 <= MAX_INIT_LEVEL < NUM_LEVELS
    assert cfg.scene.terrain.max_init_terrain_level == MAX_INIT_LEVEL
    # The walking recipe's other curricula survive.
    for name in ("action_rate_weight", "standing_envs", "head_pose_bias_weight"):
        assert name in cfg.curriculum


def test_walk_move_masks():
    size_x = 8.0
    distance = torch.tensor([3.0, 0.2, 1.0, 3.0, 0.2])
    cmd = torch.tensor([0.3, 0.3, 0.3, 0.0, 0.0])
    up, down = mdp_terrain.walk_move_masks(distance, cmd, size_x, 0.25, 0.08, 0.05)
    # walked 3 m of an 8 m tile -> up; 0.2 m -> down; 1 m -> stay.
    assert up.tolist() == [True, False, False, False, False]
    assert down.tolist() == [False, True, False, False, False]
    # zero command: never judged, whatever the distance.


def test_play_mode_is_random_mix_on_its_own_copy():
    train = make_microduck_velocity_terrain_env_cfg()
    play = make_microduck_velocity_terrain_env_cfg(play=True)
    assert play.scene.terrain.terrain_generator.curriculum is False
    assert train.scene.terrain.terrain_generator.curriculum is True
    assert play.scene.terrain.terrain_generator is not train.scene.terrain.terrain_generator


def test_terrain_does_not_disturb_the_shared_observation_contract():
    """Every policy in this family must stay hot-swappable: 61D actor obs."""
    terrain = make_microduck_velocity_terrain_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    assert list(terrain.observations["actor"].terms.keys()) == list(
        walk.observations["actor"].terms.keys()
    )
    assert list(terrain.observations["critic"].terms.keys()) == list(
        walk.observations["critic"].terms.keys()
    )


def test_every_spawn_origin_sits_on_the_terrain_surface():
    """Build a 2-level grid on CPU and raycast down at every origin. The
    inverted (pit) terrains put the origin at the pit floor; a wrong origin
    z would spawn the robot inside or under the terrain."""
    gen_cfg = make_slopes_terrain_cfg()
    gen_cfg.num_rows = 2
    gen_cfg.seed = 0
    gen = TerrainGenerator(gen_cfg, device="cpu")
    spec = mujoco.MjSpec()
    gen.compile(spec)
    _soften_terrain_contacts(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = list(gen_cfg.sub_terrains)
    geomid = np.zeros(1, dtype=np.int32)
    for col, name in enumerate(names):
        for row in range(gen_cfg.num_rows):
            o = gen.terrain_origins[row, col]
            start = np.array([o[0], o[1], o[2] + 5.0])
            dist = mujoco.mj_ray(model, data, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
            assert dist >= 0, f"{name} row {row}: no terrain under the origin"
            surface_z = start[2] - dist
            assert abs(surface_z - o[2]) < 0.005, (
                f"{name} row {row}: origin z {o[2]:.3f} vs surface {surface_z:.3f}"
            )
    # Pit types really are below the flat level, top types above it.
    up_col = names.index("slope_up")
    down_col = names.index("slope_down")
    assert gen.terrain_origins[1, up_col][2] < -0.05
    assert gen.terrain_origins[1, down_col][2] > 0.05

    # Measure the built geometry, not the cfg: face angle by raycast at two
    # points on the +x face, and on the diagonal (a planar face gives the same
    # x-gradient there; the stock bilinear pyramid would not).
    def surface_z(x, y):
        start = np.array([x, y, 10.0])
        dist = mujoco.mj_ray(model, data, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        assert dist >= 0
        return start[2] - dist

    half_p = PLATFORM_WIDTH / 2.0
    lo, hi = SLOPE_DEG_RANGE
    for name in ("slope_up", "slope_down"):
        col = names.index(name)
        for row in range(gen_cfg.num_rows):
            o = gen.terrain_origins[row, col]
            za = surface_z(o[0] + half_p + 0.5, o[1])
            zb = surface_z(o[0] + half_p + 2.5, o[1])
            axis_deg = math.degrees(math.atan2(abs(za - zb), 2.0))
            assert lo - 0.3 <= axis_deg <= hi + 0.3, (name, row, axis_deg)
            zc = surface_z(o[0] + half_p + 0.5, o[1] + half_p + 0.5)
            zd = surface_z(o[0] + half_p + 2.5, o[1] + half_p + 2.5)
            diag_deg = math.degrees(math.atan2(abs(zc - zd), 2.0))
            assert abs(diag_deg - axis_deg) < 0.5, (name, row, axis_deg, diag_deg)
    # Riser heights of the built stairs sit inside STEP_HEIGHT_RANGE.
    s_lo, s_hi = STEP_HEIGHT_RANGE
    for name in ("stairs_up", "stairs_down"):
        col = names.index(name)
        for row in range(gen_cfg.num_rows):
            o = gen.terrain_origins[row, col]
            z1 = surface_z(o[0] + half_p + STEP_RUN * 0.5, o[1])
            z2 = surface_z(o[0] + half_p + STEP_RUN * 1.5, o[1])
            riser = abs(z1 - z2)
            assert s_lo - 0.001 <= riser <= s_hi + 0.001, (name, row, riser)
