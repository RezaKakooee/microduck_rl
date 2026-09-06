"""Microduck spiral — glide on ONE roller with the other leg extended behind.

The figure-skating spiral (arabesque): the duck rolls forward balanced on a
single skate while the free leg is held straight out behind it.

Built on the roller velocity recipe, which already supplies the wheels, the BAM
actuators, the whole DR stack, and — importantly — `glide_reward`, which pays
for coasting on ONE blade with quiet legs. Three things are added or removed to
turn a skating gait into a held pose:

1. **`skating_air_time` goes to zero.** In the roller recipe it pays for each
   swing, which drives swing FREQUENCY: exactly the stroke-stroke-stroke gait we
   are trying to replace with a single sustained glide. Its own cfg comments
   record the fight between it and `glide`; here the fight is over.

2. **`spiral_free_leg`** rewards what the lifted leg actually does. `glide` is
   indifferent between a tucked foot and an extended one, so on its own it gives
   a one-legged coast with the free foot dangling. This term is the arabesque.

3. **`single_support_hold`** pays for HOLDING the pose, behind a LATCH.

v1 produced a stroking walk on wheels, not a spiral, and the metrics said it had
succeeded. Measured on that policy: single support 92% of the time, free-leg hip
+0.99 rad against a +1.10 target -- and a mean hold of 0.11 s, swapping support
feet 108 times in 13 s. A walk IS single support most of the time, and a walking
swing passes through a backward-extended hip, so both new terms were fully
satisfiable by ordinary walking. Hence the latch: below `min_hold_s` on one
wheel, both terms pay exactly zero. Held is the only thing that scores.

4. **`gait_symmetry` goes to zero.** It penalises "lopsided left/right foot
   usage" at weight -1.0, which is the literal definition of a one-legged
   spiral: it was fighting the task head-on.

The two new terms are gated on single support AND forward progress, so parking
on one leg going nowhere earns nothing — the duck has to glide.

Sign convention is measured, not assumed; see `spiral_free_leg_reward`. Both
legs extend backward OPPOSITE to their HOME hip_pitch sign, and getting that
backwards trains a knee-to-chest tuck instead of an arabesque.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from microduck_lab.rl import mdp_pose
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)

# ── The pose ─────────────────────────────────────────────────────────────────
# Free-leg hip_pitch magnitude. Measured on the roller model: at 1.2 the foot
# sits 113 mm behind the trunk and ~108 mm higher than at home, which reads as a
# proper arabesque. 1.1 leaves a little headroom below the joint limit.
HIP_EXTENSION = 1.1

# Deliberately wide: the current policy must score visibly on these or there is
# no gradient to climb. Tighten later if the pose comes out sloppy.
HIP_STD = 0.5
KNEE_STD = 0.6

# Time constant for the hold ramp, measured from the latch.
HOLD_TAU_S = 1.5

# The latch, ramped. A 1.5 s requirement is invisible to a policy that cannot yet
# balance on one wheel at all -- nothing would ever reach it and the term would
# be dead weight. So it starts just above a walking stride (v1 held 0.11 s, so
# 0.30 s already forbids stroking) and tightens as the skill consolidates.
# Measured: held open-loop in the pose with PD control and rolling, the duck's
# free foot comes down at 0.38 s. A latch at 0.30 s therefore sat right at the
# edge of what a spawned episode survives WITHOUT any balancing — almost no
# gradient. It starts at 0.20 s so improving on the unaided topple pays from the
# first iteration, and still excludes v1's 0.11 s walking stride.
MIN_HOLD_STAGES = [
    {"step":    0 * 24, "value": 0.20},
    {"step": 3000 * 24, "value": 0.45},
    {"step": 5000 * 24, "value": 0.80},
    {"step": 7000 * 24, "value": 1.20},
]

# Reverse curriculum. v3 latched the rewards and both read exactly 0.0 for all
# 10000 iterations — the policy never held one wheel for even 0.30 s, so it
# never saw the reward at all. Loosening the latch is not the answer: v1 showed
# a latch loose enough for a walking stride to reach is also satisfied BY a
# walking stride. Instead hand it the pose, then withdraw the help.
SPIRAL_SPAWN_STAGES = [
    {"step":    0 * 24, "fraction": 0.70},
    {"step": 3000 * 24, "fraction": 0.40},
    {"step": 5000 * 24, "fraction": 0.20},
    {"step": 7000 * 24, "fraction": 0.05},
    {"step": 9000 * 24, "fraction": 0.00},
]

# Forward speed at which the gates are fully open. Matches the roller recipe's
# glide/air-time gating so all three agree on what "moving" means.
VEL_REF = 0.2

# ── Pose tracking (v5) ───────────────────────────────────────────────────────
# v1-v4 all specified the spiral as a conjunction of hard gates and all four
# failed: v1's was satisfiable by a walking stride (0.11 s holds), v3's latch was
# never cleared once in 10000 iterations, v4's could not exploit even a 70%
# in-pose spawn. The common fault is that a conjunction of gates is binary — it
# is either farmable or unreachable, with no gradient between.
#
# v5 replaces it with two-layer pose tracking, the technique from
# jonathanhawkins/microduck-lab's `imitate` behaviour: a wide Gaussian that
# keeps a gradient far from the target plus a tight one that pays for precision.
# The latch and the hold reward are retired; the pose itself is the objective.
POSE_MATCH_WEIGHT = 10.0

# v5 used 1.8 / 0.6 and the policy never lifted a foot at all. Computed after
# the fact: the squared joint error between STANDING and the spiral pose is
# 2.43, so at std_wide 1.8 simply standing scored 0.237 of the maximum 1.0 —
# only 4.2x worse than a perfect spiral, for zero risk of falling. Doing nothing
# won, the same trap as the ice task.
#
# At 0.9 / 0.30 the ratio is 40x (standing 0.025, in pose 1.000) and the climb
# is still smooth, so there is a gradient the whole way:
#   0% of the way 0.025 | 25% 0.093 | 50% 0.237 | 75% 0.507 | 90% 0.867 | 100% 1.000
# Check these numbers with the snippet in src/microduck_lab/tests/test_spiral_cfg.py before
# changing them — picking a std by eye is what cost run v5.
POSE_STD_WIDE = 0.9
POSE_STD_TIGHT = 0.30

# ── Weights ──────────────────────────────────────────────────────────────────
SPIRAL_FREE_LEG_WEIGHT = 6.0
# Raised 4 -> 8: behind the latch this is now the only term that pays for
# commitment, and it has to outweigh the roller stack's pull toward stroking.
# Retired in v5: the latch mechanism is what failed. Kept at 0 rather than
# deleted so a run can switch it back on for an A/B without a code change.
SINGLE_SUPPORT_HOLD_WEIGHT = 0.0
SPIRAL_FREE_LEG_WEIGHT_V5 = 0.0
SKATING_AIR_TIME_WEIGHT = 0.0
# Penalises lopsided left/right foot usage — the definition of a spiral.
GAIT_SYMMETRY_WEIGHT = 0.0
GLIDE_WEIGHT = 4.0
# The roller recipe's sole positive task reward. Kept, but halved: at 10.0 it
# dominates the stack and the fastest way to spin wheels is to keep stroking,
# which is the gait we are replacing. It stays non-zero because a spiral that
# is not moving is not a spiral.
WHEEL_SPEED_WEIGHT = 5.0


def make_microduck_spiral_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """One-legged gliding spiral on rollers."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # Stop paying for swing frequency — see the module docstring.
    cfg.rewards["skating_air_time"].weight = SKATING_AIR_TIME_WEIGHT
    cfg.rewards["gait_symmetry"].weight = GAIT_SYMMETRY_WEIGHT
    cfg.rewards["glide"].weight = GLIDE_WEIGHT
    cfg.rewards["wheel_speed"].weight = WHEEL_SPEED_WEIGHT

    cfg.rewards["spiral_pose_match"] = RewardTermCfg(
        func=mdp_pose.spiral_pose_match_reward,
        weight=POSE_MATCH_WEIGHT,
        params={
            "command_name": "twist",
            "hip_extension": HIP_EXTENSION,
            "std_wide": POSE_STD_WIDE,
            "std_tight": POSE_STD_TIGHT,
        },
    )

    cfg.rewards["spiral_free_leg"] = RewardTermCfg(
        func=mdp_pose.spiral_free_leg_reward,
        weight=SPIRAL_FREE_LEG_WEIGHT_V5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_ref": VEL_REF,
            "min_hold_s": MIN_HOLD_STAGES[0]["value"],
            "hip_extension": HIP_EXTENSION,
            "hip_std": HIP_STD,
            "knee_std": KNEE_STD,
        },
    )

    cfg.rewards["single_support_hold"] = RewardTermCfg(
        func=mdp_pose.single_support_hold_reward,
        weight=SINGLE_SUPPORT_HOLD_WEIGHT,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_ref": VEL_REF,
            "min_hold_s": MIN_HOLD_STAGES[0]["value"],
            "tau_s": HOLD_TAU_S,
        },
    )

    # Hand the policy the pose it cannot otherwise discover. Runs on reset,
    # after the base reset that places the robot.
    cfg.events["spiral_spawn"] = EventTermCfg(
        func=mdp_pose.reset_spiral_pose,
        mode="reset",
        params={
            "hip_extension": HIP_EXTENSION,
            "fraction_stages": SPIRAL_SPAWN_STAGES,
        },
    )

    # Tighten the latch on both terms together, or the pose term keeps paying
    # for swings the hold term has already stopped paying for.
    for term in ("spiral_free_leg", "single_support_hold"):
        cfg.curriculum[f"min_hold_{term}"] = CurriculumTermCfg(
            func=mdp_pose.reward_param_curriculum,
            params={
                "reward_name": term,
                "param": "min_hold_s",
                "stages": MIN_HOLD_STAGES,
            },
        )

    return cfg


MicroduckSpiralRlCfg = deepcopy(MicroduckRollersRlCfg)
MicroduckSpiralRlCfg.experiment_name = "spiral"
