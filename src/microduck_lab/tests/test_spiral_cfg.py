"""Spiral task invariants.

The one that actually matters is the hip sign. Both legs extend BACKWARD
opposite to their HOME hip_pitch sign, and the two legs mirror each other, so a
sign slip trains a knee-to-chest tuck that scores just as well on every other
term. It is checked here against the real model rather than asserted in prose.
"""

import mujoco
import pytest

from microduck_lab.rl.microduck_spiral_env_cfg import (
    HIP_EXTENSION,
    SKATING_AIR_TIME_WEIGHT,
    make_microduck_spiral_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)

ROLLER_SCENE = "src/mjlab_microduck/robot/microduck/scene_rollers.xml"


def _foot_offset(model, data, leg, hip_pitch):
    """Foot position relative to the trunk, for one hip_pitch value."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip_pitch")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    mujoco.mj_resetData(model, data)
    data.qpos[model.jnt_qposadr[jid]] = hip_pitch
    mujoco.mj_forward(model, data)
    return data.site_xpos[sid] - data.xpos[trunk]


@pytest.mark.parametrize("leg,sign", [("left", +1.0), ("right", -1.0)])
def test_hip_extension_sign_puts_the_foot_behind_and_up(leg, sign):
    """+HIP_EXTENSION for the left leg, -HIP_EXTENSION for the right."""
    model = mujoco.MjModel.from_xml_path(ROLLER_SCENE)
    data = mujoco.MjData(model)

    home = _foot_offset(model, data, leg, -0.4579 if leg == "left" else 0.4579)
    extended = _foot_offset(model, data, leg, sign * HIP_EXTENSION)

    # Behind: x decreases. And clearly so, not marginally.
    assert extended[0] < home[0] - 0.05, f"{leg} foot did not go backward"
    # Up: z increases.
    assert extended[2] > home[2] + 0.05, f"{leg} foot did not lift"


def test_the_opposite_sign_would_tuck_not_extend():
    """Guards the failure mode: a sign slip trains a tuck that scores the same
    on single support, glide and wheel speed."""
    model = mujoco.MjModel.from_xml_path(ROLLER_SCENE)
    data = mujoco.MjData(model)
    wrong = _foot_offset(model, data, "left", -HIP_EXTENSION)
    right = _foot_offset(model, data, "left", +HIP_EXTENSION)
    assert wrong[0] > right[0], "the wrong sign should put the foot FORWARD"


def test_swing_frequency_reward_is_off():
    """`skating_air_time` pays each swing, which is the stroking gait the spiral
    replaces. Left on, it fights `single_support_hold` directly."""
    cfg = make_microduck_spiral_env_cfg()
    rollers = make_microduck_velocity_rollers_env_cfg()
    assert cfg.rewards["skating_air_time"].weight == SKATING_AIR_TIME_WEIGHT == 0.0
    assert rollers.rewards["skating_air_time"].weight > 0.0


def test_pose_terms_are_gated_on_moving_forward():
    """Parking on one leg going nowhere must earn nothing, or the policy will
    stand there instead of gliding — the ice task's exact failure."""
    cfg = make_microduck_spiral_env_cfg()
    for term in ("spiral_free_leg", "single_support_hold"):
        assert "vel_ref" in cfg.rewards[term].params
        assert cfg.rewards[term].params["vel_ref"] > 0.0
        assert cfg.rewards[term].params["sensor_name"] == "feet_ground_contact"


def test_pose_terms_are_gated_on_rolling_and_upright():
    """v1 farmed the pose reward by TOPPLING forward: the forward gate reads
    trunk velocity, which a fall supplies. spiral_free_leg went 0.06 -> 2.14
    while wheel_speed went to 0 and falls went up 11x. Wheels only turn if the
    robot is really rolling, and the tilt gate closes once it is going over."""
    from microduck_lab.rl import mdp_pose as mdp
    import inspect

    for fn in (mdp.spiral_free_leg_reward, mdp.single_support_hold_reward):
        params = inspect.signature(fn).parameters
        assert "wheel_ref" in params, f"{fn.__name__} has no wheel gate"
        assert "tilt_gate_deg" in params, f"{fn.__name__} has no upright gate"
        assert params["wheel_ref"].default > 0.0
        assert 0.0 < params["tilt_gate_deg"].default < 90.0


def test_hold_latch_makes_a_walking_stride_worth_zero():
    """The v1 failure: a stroking walk satisfied both new terms. It held one
    foot for 0.11 s, was in single support 92% of the time, and carried a
    free-leg hip of +0.99 rad against a +1.10 target. The latch has to exclude
    that stride at EVERY curriculum stage."""
    from microduck_lab.rl.microduck_spiral_env_cfg import MIN_HOLD_STAGES

    # v1's stroking walk: mean hold 0.11 s, LONGEST 0.12 s. The bound that
    # matters is the longest — a latch under it would admit the best stride.
    v1_longest_walk_hold_s = 0.12
    for stage in MIN_HOLD_STAGES:
        assert stage["value"] > v1_longest_walk_hold_s * 1.5

    # ...but it must also stay under what a spawned pose survives unaided
    # (measured: the free foot drops at 0.38 s with no balancing), or the first
    # stage is unreachable and the term is dead weight — the v3 failure.
    assert MIN_HOLD_STAGES[0]["value"] < 0.38

    # And it must tighten, not loosen.
    for a, b in zip(MIN_HOLD_STAGES, MIN_HOLD_STAGES[1:]):
        assert b["step"] > a["step"]
        assert b["value"] > a["value"]

    # Both terms latch, or the pose term keeps paying for swings.
    cfg = make_microduck_spiral_env_cfg()
    for term in ("spiral_free_leg", "single_support_hold"):
        assert cfg.rewards[term].params["min_hold_s"] == MIN_HOLD_STAGES[0]["value"]


def test_gait_symmetry_is_off():
    """It penalises lopsided left/right foot usage at -1.0 — that IS a spiral."""
    cfg = make_microduck_spiral_env_cfg()
    rollers = make_microduck_velocity_rollers_env_cfg()
    assert cfg.rewards["gait_symmetry"].weight == 0.0
    assert rollers.rewards["gait_symmetry"].weight < 0.0


def test_standing_still_scores_far_below_the_pose():
    """v5's failure, as a number. The two-layer pose match must not pay much for
    doing nothing, or standing (zero fall risk) beats attempting the spiral.
    At v5's std_wide=1.8 standing scored 0.237 against a perfect 1.000 — a 4.2x
    ratio — and the policy never lifted a foot in 10000 iterations."""
    import numpy as np
    from microduck_lab.rl.microduck_spiral_env_cfg import (
        HIP_EXTENSION, POSE_STD_TIGHT, POSE_STD_WIDE,
    )

    home = np.array([0, -0.0873, -0.4579, -0.0049, 0.453,
                     0.3491, 0.3491, 0, 0,
                     0, 0.0873, 0.4579, 0.0049, -0.453])
    ref = home.copy()
    ref[2] = HIP_EXTENSION
    ref[3] = 0.0

    def match(q):
        d2 = float(((q - ref) ** 2).sum())
        return (0.5 * np.exp(-d2 / POSE_STD_WIDE ** 2)
                + 0.5 * np.exp(-d2 / POSE_STD_TIGHT ** 2))

    standing, in_pose = match(home), match(ref)
    assert in_pose > 20 * standing, (
        f"standing scores {standing:.3f} vs {in_pose:.3f} in pose — too close, "
        "the policy will just stand there"
    )
    # ...but not zero, or there is no gradient from where every run starts.
    assert standing > 0.005

    # And the climb must be monotonic, so partial progress always pays.
    vals = [match(home + f * (ref - home)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_still_has_a_reason_to_move():
    cfg = make_microduck_spiral_env_cfg()
    assert cfg.rewards["wheel_speed"].weight > 0.0


def test_spiral_keeps_the_shared_observation_contract():
    cfg = make_microduck_spiral_env_cfg()
    rollers = make_microduck_velocity_rollers_env_cfg()
    assert list(cfg.observations["actor"].terms.keys()) == list(
        rollers.observations["actor"].terms.keys()
    )
