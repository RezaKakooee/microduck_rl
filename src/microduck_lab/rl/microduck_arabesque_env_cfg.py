"""Microduck arabesque — stand on ONE leg, other leg extended straight behind.

The figure-skating spiral pose, on the normal feet, standing still.

This replaces the roller-based spiral task, which failed six times. Two reasons,
both measured rather than guessed:

1. **Rollers cannot do it.** A roller is a line contact fore-aft and a knife edge
   sideways: zero lateral support width, so no restoring force. A scripted PD
   controller holds the pose on rollers for 0.30 s no matter the gains, lean or
   steering. A normal foot is 41 mm wide, half-width 20.5 mm.

2. **The reference pose was never balanced.** Extending the free leg alone
   leaves the centre of mass 42.8 mm to the SIDE of the support foot -- outside
   the support polygon at t=0, falling before the policy acts. A person shifts
   their weight across before lifting a foot. Adding that shift (both hip_rolls
   to the support side, bounded by hip_roll's +/-0.384 rad limit) brings the CoM
   to +2.0 mm against a foot spanning -34.1..+6.3 mm: inside, with margin.

With the balanced reference, a hand-written PD controller holds the pose for
~0.96 s (`src/microduck_lab/tasks/skating/expert_spiral.py --feet normal`). So the pose is feasible and
holding it is a genuine control problem. That is the bar RL has to beat.

The reward is two-layer pose tracking (wide Gaussian for gradient, tight for
precision), the technique from jonathanhawkins/microduck-lab's `imitate`. The
stds are chosen by computing the score of STANDING against the score of the
pose -- picking a std by eye is what wasted run v5, where standing scored 0.237
of a maximum 1.000 and the policy simply never lifted a foot.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from microduck_lab.rl import mdp_pose
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Free-leg hip_pitch. Measured: puts the foot 113 mm behind and ~108 mm up.
HIP_EXTENSION = 1.1

# Sideways weight shift, at hip_roll's mechanical limit (+/-0.384 rad). This is
# the binding constraint on how far the duck can get its CoM over one foot.
HIP_ROLL_SHIFT = 0.384

# Squared joint error between standing and this pose is 2.74, so at these stds
# standing scores ~0.017 against 1.000 in pose (~59x), and the climb is smooth.
POSE_STD_WIDE = 0.9
POSE_STD_TIGHT = 0.30

POSE_MATCH_WEIGHT = 12.0
# Standing on one leg is not walking: the gait terms actively fight the pose.
AIR_TIME_WEIGHT = 0.0
FOOT_CLEARANCE_WEIGHT = 0.0
FOOT_SWING_HEIGHT_WEIGHT = 0.0


def make_microduck_arabesque_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """One-legged arabesque hold on normal feet."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    cfg.rewards["arabesque_pose"] = RewardTermCfg(
        func=mdp_pose.arabesque_pose_match_reward,
        weight=POSE_MATCH_WEIGHT,
        params={
            "hip_extension": HIP_EXTENSION,
            "hip_roll_shift": HIP_ROLL_SHIFT,
            "std_wide": POSE_STD_WIDE,
            "std_tight": POSE_STD_TIGHT,
        },
    )

    # The walking recipe pays for lifting feet through a stride window and for
    # swing height. Both reward a GAIT, which is the opposite of holding still
    # on one leg — left on, they push the duck to step out of the pose.
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["foot_clearance"].weight = FOOT_CLEARANCE_WEIGHT
    cfg.rewards["foot_swing_height"].weight = FOOT_SWING_HEIGHT_WEIGHT

    return cfg


MicroduckArabesqueRlCfg = deepcopy(MicroduckRlCfg)
MicroduckArabesqueRlCfg.experiment_name = "arabesque"
