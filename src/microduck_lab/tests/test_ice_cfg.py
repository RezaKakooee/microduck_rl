"""Ice variant invariants.

The one that actually matters is the friction pinning: MuJoCo combines contact
friction as the elementwise MAXIMUM of the two geoms, so a low foot friction
against the default mu=1.0 ground plane is a silent no-op and the "ice" task
would quietly train on ordinary ground.
"""

import mujoco

from microduck_lab.rl.microduck_velocity_ice_env_cfg import (
    ICE_FLOOR_MU,
    ICE_FOOT_MU_RANGE,
    ICE_FRICTION_STAGES,
    _make_terrain_icy,
    make_microduck_velocity_ice_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_floor_is_pinned_below_every_foot_sample():
    """max(floor, foot) must always resolve to the foot sample."""
    assert ICE_FLOOR_MU < ICE_FOOT_MU_RANGE[0]


def test_mujoco_combines_friction_by_maximum():
    """The assumption the whole variant rests on. If MuJoCo ever changes this
    rule, pinning the floor stops being necessary and this test says so."""

    def slid(mu_plane, mu_box):
        xml = f"""<mujoco><option timestep="0.005"/><worldbody>
          <geom type="plane" size="0 0 .01" friction="{mu_plane} 0.005 0.0001"/>
          <body pos="0 0 0.05"><freejoint/>
            <geom type="box" size=".05 .05 .05" mass="1" friction="{mu_box} 0.005 0.0001"/>
          </body></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        d.qvel[0] = 2.0
        for _ in range(600):
            mujoco.mj_step(m, d)
        return float(d.qpos[0])

    grippy = slid(1.0, 1.0)
    # One side slippery changes nothing: the max picks the grippy one.
    assert abs(slid(0.02, 1.0) - grippy) < 0.05
    assert abs(slid(1.0, 0.02) - grippy) < 0.05
    # Both slippery, and it glides an order of magnitude further.
    assert slid(0.02, 0.02) > 10 * grippy


def test_foot_friction_starts_grippy_and_resamples():
    """The curriculum only reaches the physics if the event re-samples: mjlab's
    base term is startup-mode, which would freeze friction at stage 0 forever."""
    cfg = make_microduck_velocity_ice_env_cfg()
    assert cfg.events["foot_friction"].mode == "reset"
    assert cfg.events["foot_friction"].params["ranges"] == ICE_FRICTION_STAGES[0]["ranges"]
    # "abs" means these are absolute coefficients, not scale factors on 1.0.
    assert cfg.events["foot_friction"].params["operation"] == "abs"


def test_spec_fn_makes_terrain_geoms_icy():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    body.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01))
    _make_terrain_icy(spec)
    assert spec.body("terrain").geoms[0].friction[0] == ICE_FLOOR_MU


def test_friction_curriculum_ramps_grip_down_to_ice():
    cfg = make_microduck_velocity_ice_env_cfg()
    assert cfg.curriculum["ice_friction"].params["event_name"] == "foot_friction"
    stages = ICE_FRICTION_STAGES
    # Monotonically slipperier, and it ends on the ice range.
    for a, b in zip(stages, stages[1:]):
        assert b["step"] > a["step"]
        assert b["ranges"][0] < a["ranges"][0]
        assert b["ranges"][1] < a["ranges"][1]
    assert stages[-1]["ranges"] == ICE_FOOT_MU_RANGE
    # Stage 0 must be genuine grip, or there is no walking phase to build on.
    assert stages[0]["ranges"][0] >= 0.5


def test_slip_untaxed_but_air_time_keeps_its_walking_weight():
    """Slip is the medium on ice. Air time is what pays for moving a foot at
    all — v1 softened it and the policy stopped stepping entirely."""
    ice = make_microduck_velocity_ice_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    assert ice.rewards["foot_slip"].weight == 0.0
    assert walk.rewards["foot_slip"].weight < 0.0
    assert ice.rewards["air_time"].weight == walk.rewards["air_time"].weight
    assert (
        ice.rewards["air_time"].params["threshold_max"]
        > walk.rewards["air_time"].params["threshold_max"]
    )


def test_ice_does_not_disturb_the_shared_observation_contract():
    """Every policy in this family must stay hot-swappable: 61D actor obs."""
    ice = make_microduck_velocity_ice_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    assert list(ice.observations["actor"].terms.keys()) == list(
        walk.observations["actor"].terms.keys()
    )
    assert list(ice.observations["critic"].terms.keys()) == list(
        walk.observations["critic"].terms.keys()
    )
