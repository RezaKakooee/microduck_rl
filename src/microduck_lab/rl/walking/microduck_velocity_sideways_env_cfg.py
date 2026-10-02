"""Microduck velocity environment — sideways walking.

Same task as the walking recipe (track a commanded twist, head / body pose
slots zero-padded into the shared 61D obs), trained so that the vy slot of
the twist actually moves the duck sideways. Built on
``make_microduck_velocity_env_cfg`` so DR / obs / noise / delays stay in sync.

No policy we have strafes (see ``mdp_sideways.py`` for the measurements and
the three reasons). What changes here:

1. **A sideways bucket.** ``SIDEWAYS_FRACTION`` of command resamples are pure
   sideways: vx within +-3 cm/s, |vy| in ``SIDEWAYS_SPEED``, wz within
   +-0.1 rad/s. The rest is the recipe's own sampling (including its
   turn-in-place bucket and standing curriculum), so forward walking and
   turning keep being trained.
2. **Two tight tracking terms** on top of the recipe's loose ones:
   ``track_lateral_velocity`` (std 0.08 m/s) and ``track_yaw_rate_tight``
   (std 0.2 rad/s, weight 3), so that ignoring vy or drifting in yaw while
   stepping sideways stops paying.
3. **Hip roll is freed while walking.** The walking pose std on hip roll goes
   from 0.05 to ``HIP_ROLL_STD_WALKING`` rad. The standing std stays at 0.05,
   so the flat-sole standing stance is untouched.
4. **Steps, not slides.** v1-v3 shuffled: swing peaks 1.7-5 mm of real lift.
   The swing-height cost is 12x heavier with a target of ~20 mm real lift,
   ``swing_lift`` pays for a foot clearly up, and sliding a touching foot
   costs linearly in its speed (``feet_slip_linear``).
5. **Symmetry mirror loss is ON.** Stepping left and stepping right are mirror
   images, and ``slopes_v1`` learned only one of them. The 61D mirror table in
   ``mjlab_microduck/tasks/symmetry.py`` had never been used by an env; its
   joint-sign rules are checked against the model in
   ``tests/test_sideways_cfg.py``.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from microduck_lab.rl.walking import mdp_sideways
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

SIDEWAYS_FRACTION = 0.40
SIDEWAYS_SPEED = (0.10, 0.25)       # m/s; the recipe's own vy range is +-0.3
LATERAL_STD = 0.08                  # m/s
LATERAL_WEIGHT = 2.0
# v1 (std 0.3, weight 1.0) strafed 170 mm/s at iter 250 but turned 40-110
# degrees per 6 s sideways segment: a steady 0.2 rad/s drift cost only 0.36 of
# a weight-1 term against up to 2.0 for sideways tracking, so turning was the
# cheap way to score. At std 0.2 / weight 3 the same drift costs ~1.9.
# v2 (std 0.2 / weight 3) held heading better but SHUFFLED: never turning is
# cheapest with both feet on the ground. v3 sits between v1 and v2.
YAW_STD = 0.25                      # rad/s
YAW_WEIGHT = 2.0
HIP_ROLL_STD_WALKING = 0.20         # rad; recipe 0.05
# v1 and v2 slid their feet instead of stepping (95th-percentile foot lift
# 0.5-4.5 mm, slip 13-71 mm/s while touching). The recipe prices neither:
# its slip cost is squared (5 cm/s -> 0.0025 per foot) and its swing-height
# cost has weight -0.25 (a 0.5 mm shuffle: ~0.01 per step).
SWING_HEIGHT_WEIGHT = -3.0          # recipe -0.25
# v3 (target 20 mm at the site, slip -5) still shuffled: swing peaks 1.7-3.4 mm
# of REAL lift. The site sits 10.3 mm above a flat sole, so the recipe's
# 20 mm target only ever asked for ~10 mm. v4:
SWING_HEIGHT_TARGET = 0.030         # at the site: ~20 mm of real lift
SLIP_LINEAR_WEIGHT = -10.0          # * sum |foot speed| on contact (m/s)
LIFT_WEIGHT = 1.0                   # swing_lift: 0 at site 10 mm, 1 at 25 mm
LIFT_RANGE = (0.010, 0.025)


def make_microduck_velocity_sideways_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Velocity tracking with real sideways steps."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    twist = cfg.commands["twist"]
    cfg.commands["twist"] = mdp_sideways.SidewaysVelocityCommandCfg(
        **vars(twist),
        rel_sideways_envs=SIDEWAYS_FRACTION,
        sideways_speed=SIDEWAYS_SPEED,
    )

    cfg.rewards["track_lateral_velocity"] = RewardTermCfg(
        func=mdp_sideways.track_lateral_velocity,
        weight=LATERAL_WEIGHT,
        params={"command_name": "twist", "std": LATERAL_STD},
    )
    cfg.rewards["track_yaw_rate_tight"] = RewardTermCfg(
        func=mdp_sideways.track_yaw_rate_tight,
        weight=YAW_WEIGHT,
        params={"command_name": "twist", "std": YAW_STD},
    )

    cfg.rewards["foot_swing_height"].weight = SWING_HEIGHT_WEIGHT
    cfg.rewards["foot_swing_height"].params["target_height"] = SWING_HEIGHT_TARGET
    swing = cfg.rewards["foot_swing_height"].params
    cfg.rewards["swing_lift"] = RewardTermCfg(
        func=mdp_sideways.swing_lift, weight=LIFT_WEIGHT,
        params={"sensor_name": swing["sensor_name"], "height_sensor_name": swing["height_sensor_name"],
                "low": LIFT_RANGE[0], "high": LIFT_RANGE[1], "command_name": "twist"})
    slip = cfg.rewards["foot_slip"].params
    cfg.rewards["foot_slip_linear"] = RewardTermCfg(
        func=mdp_sideways.feet_slip_linear, weight=SLIP_LINEAR_WEIGHT,
        params={"sensor_name": slip["sensor_name"], "command_name": "twist",
                "command_threshold": slip.get("command_threshold", 0.01),
                "asset_cfg": deepcopy(slip["asset_cfg"])})

    pose = cfg.rewards["pose"].params
    pose["std_walking"] = dict(pose["std_walking"])
    pose["std_running"] = dict(pose["std_running"])
    for key in list(pose["std_walking"]):
        if "hip_roll" in key:
            pose["std_walking"][key] = HIP_ROLL_STD_WALKING
            pose["std_running"][key] = HIP_ROLL_STD_WALKING

    return cfg


MicroduckSidewaysRlCfg = deepcopy(MicroduckRlCfg)
MicroduckSidewaysRlCfg.experiment_name = "velocity_sideways"
MicroduckSidewaysRlCfg.run_name = "velocity_sideways"
MicroduckSidewaysRlCfg.max_iterations = 5000
MicroduckSidewaysRlCfg.algorithm.symmetry_cfg = SYMMETRY_CFG
