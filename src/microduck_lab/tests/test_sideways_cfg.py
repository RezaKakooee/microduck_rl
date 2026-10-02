"""Sideways-walking variant invariants (CPU only).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest src/microduck_lab/tests/test_sideways_cfg.py -q
"""

import mujoco
import numpy as np

# Load the task package first. It registers every task, including this one,
# and importing a lab module before it would re-enter it half-built.
import mjlab_microduck.tasks  # noqa: F401
from microduck_lab import paths
from microduck_lab.rl.walking import mdp_sideways
from microduck_lab.rl.walking.microduck_velocity_sideways_env_cfg import (
    HIP_ROLL_STD_WALKING,
    MicroduckSidewaysRlCfg,
    SIDEWAYS_FRACTION,
    make_microduck_velocity_sideways_env_cfg,
)
from mjlab_microduck.tasks import symmetry
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def test_sideways_bucket_is_on():
    cmd = make_microduck_velocity_sideways_env_cfg().commands["twist"]
    assert isinstance(cmd, mdp_sideways.SidewaysVelocityCommandCfg)
    assert cmd.rel_sideways_envs == SIDEWAYS_FRACTION > 0
    assert 0 < cmd.sideways_speed[0] < cmd.sideways_speed[1] <= cmd.ranges.lin_vel_y[1]
    # The recipe's own buckets survive the swap.
    assert cmd.rel_turn_in_place_envs > 0


def test_new_tracking_terms_are_rewards():
    """Both functions return exp(.) >= 0, so the weights must be positive."""
    r = make_microduck_velocity_sideways_env_cfg().rewards
    assert r["track_lateral_velocity"].weight > 0
    assert r["track_yaw_rate_tight"].weight > 0


def test_hip_roll_freed_only_while_walking():
    pose = make_microduck_velocity_sideways_env_cfg().rewards["pose"].params
    for group in ("std_walking", "std_running"):
        rolls = [v for k, v in pose[group].items() if "hip_roll" in k]
        assert rolls == [HIP_ROLL_STD_WALKING]
    assert [v for k, v in pose["std_standing"].items() if "hip_roll" in k] == [0.05]


def test_base_recipe_untouched():
    """Building the variant must not leak into the walking recipe."""
    make_microduck_velocity_sideways_env_cfg()
    base = make_microduck_velocity_env_cfg()
    assert [v for k, v in base.rewards["pose"].params["std_walking"].items() if "hip_roll" in k] == [0.05]
    assert "track_lateral_velocity" not in base.rewards
    assert MicroduckRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckSidewaysRlCfg.algorithm.symmetry_cfg is symmetry.SYMMETRY_CFG


def test_mirror_table_matches_the_model():
    """Mirror a random pose with the symmetry table: the feet and head must
    land at the y-mirrored positions of the original. This is the only check
    the 61D table has ever had against the robot itself."""
    m = mujoco.MjModel.from_xml_path(str(paths.ROBOT / "robot_walk.xml"))
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    site = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    from microduck_lab.sim.upstream import DEFAULT_POSE
    home = np.asarray(DEFAULT_POSE[:14], float)
    perm, sign = np.array(symmetry._JOINT_PERM), np.array(symmetry._JOINT_SIGN)

    def sites(q_rel):
        d.qpos[:] = 0
        d.qpos[3] = 1
        d.qpos[qadr] = home + q_rel
        mujoco.mj_kinematics(m, d)
        return {n: d.site_xpos[site(n)].copy() for n in ("left_foot", "right_foot", "mouth_tip")}

    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-0.3, 0.3, 14)
        a, b = sites(q), sites(q[perm] * sign)
        flip = np.array([1.0, -1.0, 1.0])
        np.testing.assert_allclose(b["left_foot"], a["right_foot"] * flip, atol=2e-3)
        np.testing.assert_allclose(b["right_foot"], a["left_foot"] * flip, atol=2e-3)
        np.testing.assert_allclose(b["mouth_tip"], a["mouth_tip"] * flip, atol=2e-3)


# -- Mjlab-Bridge-Sideways-MicroDuck --------------------------------------------

from microduck_lab.rl.human_bridge import microduck_bridge_sideways_env_cfg as bridge  # noqa: E402


def test_bridge_envs_share_one_origin():
    """The set is baked into the terrain at fixed world coordinates, so every
    env must sit at the origin. The scene copies its OWN env_spacing onto the
    terrain; the first smoke run set only the terrain's and spawned her 1 m
    off the set."""
    cfg = bridge.make_microduck_bridge_sideways_env_cfg()
    assert cfg.scene.env_spacing == 0.0


def test_bridge_spawn_start_and_route():
    """A share of episodes start at the real start (her ledge, whole body
    before its edge); the rest anywhere along the route, above the surface."""
    cfg = bridge.make_microduck_bridge_sideways_env_cfg()
    p = cfg.events["reset_base"].params
    assert p["start_x"][1] < bridge.L.near_edge - 0.10
    assert 0.0 < p["p_start"] < 1.0
    # Never spawn already across (her trunk past the finish line).
    assert p["route_x"][0] >= p["start_x"][0] and p["route_x"][1] < bridge.DONE_X - 0.03
    assert abs(sum(p["yaw"]) / 2 - bridge.HEADING) < 1e-9
    assert "push_robot" not in cfg.events
    # The spawn height table sees his body (hull tops), not just the ledges.
    xs, zs = p["surface_x"], p["surface_z"]
    over_him = [z for x, z in zip(xs, zs) if bridge.L.near_edge < x < bridge.L.gap_end]
    assert min(over_him) > bridge.L.step_z + 0.03


def test_bridge_rewards_pay_progress_not_standing():
    r = bridge.make_microduck_bridge_sideways_env_cfg().rewards
    assert r["progress"].weight > 0
    assert r["centre_line"].weight < 0 and r["heading"].weight < 0   # costs, >= 0 functions
    assert r["track_linear_velocity"].weight < 2.0   # set through ALIVE_WEIGHTS


def test_bridge_command_always_goes_across():
    cmd = bridge.make_microduck_bridge_sideways_env_cfg().commands["twist"]
    assert cmd.rel_sideways_envs == 1.0 and cmd.sideways_sign == 1.0
    assert cmd.rel_standing_envs == 0.0 and cmd.rel_turn_in_place_envs == 0.0
    # Facing -y, her +y (left) is world +x: across the gap.
    import math
    left = (-math.sin(bridge.HEADING), math.cos(bridge.HEADING))
    assert left[0] > 0.99


def test_bridge_geometry_compiles():
    s = mujoco.MjSpec()
    s.worldbody.add_body(name="terrain")
    bridge.add_bridge(s)
    m = s.compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    # 39 solid meshes since 2026-09-26: + 15 servos, 2 thigh plates, 2 ankle brackets.
    assert sum(n.startswith("brother_") for n in names) == 39
    assert bridge.DROP_Z < min(bridge.L.near_z, bridge.L.far_z) + 0.117   # standing trunk stays above it


def test_stepping_is_priced():
    """v1/v2 slid their feet. The swing-height and linear slip costs are the
    fix; both must be costs (negative weight on >= 0 functions)."""
    from microduck_lab.rl.walking import microduck_velocity_sideways_env_cfg as side
    r = make_microduck_velocity_sideways_env_cfg().rewards
    assert r["foot_swing_height"].weight == side.SWING_HEIGHT_WEIGHT < -1.0
    assert r["foot_slip_linear"].weight < 0
    assert r["foot_slip_linear"].params["asset_cfg"].site_names


def test_lift_reward_ignores_hovering():
    """swing_lift must pay nothing below the resting site height (10.3 mm)."""
    from microduck_lab.rl.walking import microduck_velocity_sideways_env_cfg as side
    r = make_microduck_velocity_sideways_env_cfg().rewards["swing_lift"]
    assert r.weight > 0 and r.params["low"] >= 0.010 and r.params["high"] > r.params["low"]
    assert side.SWING_HEIGHT_TARGET > 0.020


def test_bridge_marching_in_place_does_not_pay():
    """v5/v6 marched in place at the edge. The alive-type terms are turned down,
    and the fall term is a cost on a >= 0 function."""
    r = bridge.make_microduck_bridge_sideways_env_cfg().rewards
    for name, w in bridge.ALIVE_WEIGHTS.items():
        assert r[name].weight == w
    assert r["fell"].weight < 0
    assert r["crossed"].weight >= 100
