"""Microduck velocity environment — ice variant.

Same task as the walking recipe (track a commanded twist, head/body pose slots
zero-padded into the shared 61D obs) on a surface with almost no grip. The duck
has no blades: its flat soles ARE the skates, so the skill it has to find is
pushing sideways against the ice and riding the glide, not stepping.

Built on `make_microduck_velocity_env_cfg` so the DR / obs / noise / delay stack
stays in sync with the walking recipe for free. Three things change:

1. **Both surfaces go icy.** MuJoCo combines contact friction as the elementwise
   MAXIMUM of the two geoms, verified here: a box at mu=1.0 on a plane at
   mu=0.02 slides exactly as far as mu=1.0 on mu=1.0 (0.25 m), and only when
   BOTH are 0.02 does it glide (5.12 m). Dropping the foot friction alone would
   have been a silent no-op against the default mu=1.0 ground plane.

2. **`foot_slip` goes to zero.** The walking recipe prices slip at -0.1. On ice
   slip is not an escapable error, it is the medium — AGENTS.md's rule is to
   price only the escapable part, and a slip tax here is a tax on skating.

3. **The ice arrives on a curriculum.** v1 dropped the duck straight onto full
   ice and it learned to stand perfectly still: 6000 iterations, feet never
   leaving the ground (peak swing height 0.2 mm), velocity error 0.46 m/s
   against commands of at most 0.3. Standing banked upright + pose +
   head_pose_tracking, about 5.5 of ~7.2 total positive reward, for free, while
   any attempt to push risked a fall and paid the action-rate tax. Exactly the
   failure AGENTS.md describes — a tax on a skill that does not exist yet makes
   "do nothing" the argmax. So friction now starts at walking grip and ramps
   down to ice over ~4200 iterations, and `air_time` keeps its walking weight:
   softening it in v1 removed the one term that pays for moving a foot at all.

Per-env ice quality comes from the foot friction DR range, which is also what
the curriculum drives. The floor is pinned BELOW every foot sample so max()
always picks the foot value — that is what makes a curriculum on the FEET alone
control the effective friction across the whole ramp.
"""

from copy import deepcopy

import mujoco as _mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# ── Ice ──────────────────────────────────────────────────────────────────────
# Pinned below the bottom of ICE_FOOT_MU_RANGE so the contact max() always
# resolves to the per-env foot sample.
ICE_FLOOR_MU = 0.02  # below every foot sample, so max() always picks the foot

# Effective per-env friction. Real ice-on-steel is ~0.005-0.02; this sits
# deliberately higher because the duck skates on rubber soles, not blades, and
# because the top of the range gives early training something to push against.
# The spread is the domain randomisation: the policy cannot observe friction, so
# it has to find a gait robust across the range rather than tuned to one value.
# v2 ended at (0.04, 0.18) and the policy UNLEARNED stepping over the 3800
# iterations it spent there: air_time_mean decayed 0.095 -> 0.037 and the
# air_time reward collapsed 0.80 -> 0.07, ending exactly where v1 did.
#
# The reason is physical, not a tuning miss. Real skating works because a blade
# is ANISOTROPIC — slippery along its length, grippy across it — so a skater has
# something to push against sideways while gliding forward. MuJoCo does support
# that (a <pair> with 5 friction terms slides 0.17 m one way and 3.61 m the
# other), but the duck has flat isotropic rubber soles. At mu 0.04 in every
# direction there is nothing to push against at all, so "stand still" is not a
# local optimum the policy got stuck in, it is close to the only option.
#
# So the floor of the ramp moves to where this robot can still generate force.
# At mu 0.1 v2 was demonstrably still stepping (air_time_mean 0.095, stable for
# 1000 iterations), so that is a skatable surface for flat soles; below it is
# not. Anisotropic skate contacts are the real fix and are noted in HANDOFF.md.
ICE_FOOT_MU_RANGE = (0.10, 0.25)

# Torsional/rolling friction stay at MuJoCo defaults — only sliding is icy.
ICE_TORSIONAL_MU = 0.005
ICE_ROLLING_MU = 0.0001

# ── Friction curriculum ──────────────────────────────────────────────────────
# Stage steps are env steps: iteration * NUM_STEPS_PER_ENV (24).
# Starts at the walking recipe's own foot friction so the first stage IS the
# walking task, then slides to ICE_FOOT_MU_RANGE.
ICE_FRICTION_STAGES = [
    {"step":    0 * 24, "ranges": (0.70, 1.30)},   # ordinary ground — learn to walk
    {"step": 1000 * 24, "ranges": (0.40, 0.90)},   # greasy
    {"step": 2000 * 24, "ranges": (0.25, 0.60)},   # wet tile
    {"step": 3000 * 24, "ranges": (0.15, 0.40)},   # slick
    {"step": 4000 * 24, "ranges": ICE_FOOT_MU_RANGE},  # ice
]

# ── Reward adjustments ───────────────────────────────────────────────────────
FOOT_SLIP_WEIGHT = 0.0
# Keep the walking weight. v1 softened this to 1.0 and the policy simply stopped
# picking its feet up (Episode_Reward/air_time settled at 0.0001).
AIR_TIME_WEIGHT = 3.0
# Window widened at the top end only: a glide holds a foot down longer than a
# walking stride, so allow it without removing the reason to lift at all.
AIR_TIME_WINDOW = (0.10, 0.40)


def _make_terrain_icy(spec: _mujoco.MjSpec) -> None:
    """Drop the sliding friction of every terrain geom to ICE_FLOOR_MU.

    Mirrors `_soften_terrain_contacts`: everything the TerrainImporter creates
    lives under the body named "terrain", for the plane and the generator alike.
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.friction = [ICE_FLOOR_MU, ICE_TORSIONAL_MU, ICE_ROLLING_MU]
        count += 1
    print(f"[ice] spec_fn: set {count} terrain geom(s) to mu={ICE_FLOOR_MU}")


def make_microduck_velocity_ice_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Velocity tracking on ice."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # The rough variant already installs a spec_fn to soften box-edge contacts.
    # Chain rather than clobber, or rough-ice loses the NaN guard that fix is.
    previous_spec_fn = cfg.scene.spec_fn

    if previous_spec_fn is None:
        cfg.scene.spec_fn = _make_terrain_icy
    else:

        def _soften_then_ice(spec: _mujoco.MjSpec) -> None:
            previous_spec_fn(spec)
            _make_terrain_icy(spec)

        cfg.scene.spec_fn = _soften_then_ice

    # Per-env ice quality. operation="abs" is inherited from the mjlab base term,
    # so these are absolute coefficients, not scales.
    #
    # mjlab samples foot friction ONCE at startup. The curriculum below rewrites
    # this range as training goes, which only reaches the physics if the event
    # re-samples — so it moves to reset mode.
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["foot_friction"].params["ranges"] = ICE_FRICTION_STAGES[0]["ranges"]

    # Ramp grip -> ice. `wheel_friction_curriculum` is not roller-specific: it
    # just writes `ranges` on a named event term at stage boundaries.
    cfg.curriculum["ice_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "foot_friction",
            "ranges_stages": ICE_FRICTION_STAGES,
        },
    )

    # See the module docstring for why these two move.
    cfg.rewards["foot_slip"].weight = FOOT_SLIP_WEIGHT
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_WINDOW[0]
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_WINDOW[1]

    return cfg


MicroduckIceRlCfg = deepcopy(MicroduckRlCfg)
MicroduckIceRlCfg.experiment_name = "velocity_ice"
