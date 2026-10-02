"""Registers our RL tasks with mjlab.

These registrations used to sit in upstream `mjlab_microduck/tasks/__init__.py`
as 71 lines of ours in the middle of theirs. Every upstream merge had to be
resolved by hand there. Now upstream carries a single line -- an import of this
module -- and everything else is here.

mjlab discovers tasks through the `mjlab.tasks` entry point, which loads
`mjlab_microduck.tasks`. The last line of that module calls `register_all` here
with its runner class. The runner is passed in rather than imported back,
because importing it would close a cycle: upstream imports us while it is still
half-built, so its class does not exist yet. This registers:

    Mjlab-Arabesque-Flat-MicroDuck      stand on one leg, other leg behind
    Mjlab-Spiral-Flat-MicroDuck         the same, gliding on one roller
    Mjlab-Velocity-Ice-MicroDuck        velocity tracking with almost no grip
    Mjlab-Velocity-Slopes-MicroDuck     5-15 deg slopes and 10-20 mm stairs
    Mjlab-Velocity-Blades-MicroDuck     anisotropic skate-blade contact pairs
    Mjlab-Velocity-Sideways-MicroDuck   walking recipe that really steps sideways
    Mjlab-Bridge-Sideways-MicroDuck     step sideways across a brother lying over a gap
    Mjlab-Velocity-SidewaysBeam-MicroDuck  step sideways along narrow beams with height changes
    Mjlab-Trampoline-Flip-MicroDuck     front flip on a trampoline, land on the feet, keep bouncing
    Mjlab-Trampoline-PumpFlip-MicroDuck start standing, bounce a few times, then the flip
    Mjlab-Trampoline-PumpFlipCurriculum-MicroDuck  the same, reached in steps from drop starts
    Mjlab-Trampoline-PumpFlipCurriculumSlow-MicroDuck  the same with stages twice as long
    Mjlab-Trampoline-Bounce-MicroDuck   start standing and keep bouncing (no flip)
    Mjlab-Trampoline-BounceHigh-MicroDuck  the same, paid up to 0.30 m bounces (hand-over height for the flip)
    Mjlab-Trampoline-BounceHighPose-MicroDuck  the same, in the standing pose at the top of each flight
    Mjlab-Trampoline-BounceHighPoseTight-MicroDuck  the same, pose_std 1.0 (bounce3 at 2.0 kept the neck ~40 deg off)
    Mjlab-Trampoline-FlipAnyPose-MicroDuck  the flip from drops, from wider start poses and tilts (for the hand-over)
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from microduck_lab.rl.skating.microduck_arabesque_env_cfg import (
    make_microduck_arabesque_env_cfg,
    MicroduckArabesqueRlCfg,
)
from microduck_lab.rl.skating.microduck_spiral_env_cfg import (
    make_microduck_spiral_env_cfg,
    MicroduckSpiralRlCfg,
)
from microduck_lab.rl.skating.microduck_velocity_ice_env_cfg import (
    make_microduck_velocity_ice_env_cfg,
    MicroduckIceRlCfg,
)
from microduck_lab.rl.walking.microduck_velocity_terrain_env_cfg import (
    make_microduck_velocity_terrain_env_cfg,
    MicroduckSlopesRlCfg,
)
from microduck_lab.rl.skating.microduck_velocity_blades_env_cfg import (
    make_microduck_velocity_blades_env_cfg,
    MicroduckBladesRlCfg,
)
from microduck_lab.rl.human_bridge.microduck_bridge_sideways_env_cfg import (
    make_microduck_bridge_sideways_env_cfg,
    MicroduckBridgeSidewaysRlCfg,
)
from microduck_lab.rl.walking.microduck_velocity_sideways_env_cfg import (
    make_microduck_velocity_sideways_env_cfg,
    MicroduckSidewaysRlCfg,
)
from microduck_lab.rl.human_bridge.rl_policy.microduck_velocity_sideways_beam_env_cfg import (
    make_microduck_velocity_sideways_beam_env_cfg,
    MicroduckSidewaysBeamRlCfg,
)
from microduck_lab.rl.microduck_trampoline_flip_env_cfg import (
    make_microduck_trampoline_flip_env_cfg,
    MicroduckTrampolineFlipRlCfg,
    MicroduckTrampolinePumpFlipRlCfg,
    BOUNCE_POSE_STD,
    make_microduck_trampoline_bounce_env_cfg,
    make_microduck_trampoline_flip_anypose_env_cfg,
    MicroduckTrampolineBounceRlCfg,
)

def register_all(runner_cls) -> None:
    """Register our tasks. Called once, from `mjlab_microduck.tasks`."""
    # Arabesque — stand on one leg, other leg extended behind (normal feet).
    register_mjlab_task(
        task_id="Mjlab-Arabesque-Flat-MicroDuck",
        env_cfg=make_microduck_arabesque_env_cfg(),
        play_env_cfg=make_microduck_arabesque_env_cfg(play=True),
        rl_cfg=MicroduckArabesqueRlCfg,
        runner_cls=runner_cls,
    )

    # Spiral — glide on one roller, free leg extended behind (figure-skating spiral).
    register_mjlab_task(
        task_id="Mjlab-Spiral-Flat-MicroDuck",
        env_cfg=make_microduck_spiral_env_cfg(),
        play_env_cfg=make_microduck_spiral_env_cfg(play=True),
        rl_cfg=MicroduckSpiralRlCfg,
        runner_cls=runner_cls,
    )

    # Ice — velocity tracking with almost no grip. Flat soles as skates.
    register_mjlab_task(
        task_id="Mjlab-Velocity-Ice-MicroDuck",
        env_cfg=make_microduck_velocity_ice_env_cfg(),
        play_env_cfg=make_microduck_velocity_ice_env_cfg(play=True),
        rl_cfg=MicroduckIceRlCfg,
        runner_cls=runner_cls,
    )

    # Slopes and stairs — velocity tracking on gentle slopes (5-15 deg, up and
    # down) and low stairs (10-20 mm risers), normal feet. Built on the rough
    # velocity recipe with its own terrain generator + distance curriculum.
    register_mjlab_task(
        task_id="Mjlab-Velocity-Slopes-MicroDuck",
        env_cfg=make_microduck_velocity_terrain_env_cfg(),
        play_env_cfg=make_microduck_velocity_terrain_env_cfg(play=True),
        rl_cfg=MicroduckSlopesRlCfg,
        runner_cls=runner_cls,
    )

    # Blades — velocity tracking on anisotropic skate-blade contact pairs:
    # slippery along the sole, grippy across it, projected onto the world-fixed
    # pair frame every step (see microduck_velocity_blades_env_cfg.py).
    register_mjlab_task(
        task_id="Mjlab-Velocity-Blades-MicroDuck",
        env_cfg=make_microduck_velocity_blades_env_cfg(),
        play_env_cfg=make_microduck_velocity_blades_env_cfg(play=True),
        rl_cfg=MicroduckBladesRlCfg,
        runner_cls=runner_cls,
    )

    # Sideways — the walking recipe with a pure-sideways command bucket, tight
    # vy / yaw-rate tracking, hip roll freed while walking, and the symmetry
    # mirror loss on (see microduck_velocity_sideways_env_cfg.py).
    register_mjlab_task(
        task_id="Mjlab-Velocity-Sideways-MicroDuck",
        env_cfg=make_microduck_velocity_sideways_env_cfg(),
        play_env_cfg=make_microduck_velocity_sideways_env_cfg(play=True),
        rl_cfg=MicroduckSidewaysRlCfg,
        runner_cls=runner_cls,
    )

    # Bridge — the human-bridge scene as an RL task: she steps sideways across
    # her brother's baked lying body (microduck_bridge_sideways_env_cfg.py).
    register_mjlab_task(
        task_id="Mjlab-Bridge-Sideways-MicroDuck",
        env_cfg=make_microduck_bridge_sideways_env_cfg(),
        play_env_cfg=make_microduck_bridge_sideways_env_cfg(play=True),
        rl_cfg=MicroduckBridgeSidewaysRlCfg,
        runner_cls=runner_cls,
    )

    # Sideways beam — the sideways walker on narrow beams (150 -> 28 mm) with
    # steps, ramps, gaps and bumps, steered by the deployment steering law
    # (microduck_velocity_sideways_beam_env_cfg.py).
    register_mjlab_task(
        task_id="Mjlab-Velocity-SidewaysBeam-MicroDuck",
        env_cfg=make_microduck_velocity_sideways_beam_env_cfg(),
        play_env_cfg=make_microduck_velocity_sideways_beam_env_cfg(play=True),
        rl_cfg=MicroduckSidewaysBeamRlCfg,
        runner_cls=runner_cls,
    )

    # Trampoline flip — bounce, front flip in the air, land on the feet, keep
    # bouncing, on a spring bed entity at a 1 ms step
    # (microduck_trampoline_flip_env_cfg.py, trampoline_scene.py).
    register_mjlab_task(
        task_id="Mjlab-Trampoline-Flip-MicroDuck",
        env_cfg=make_microduck_trampoline_flip_env_cfg(),
        play_env_cfg=make_microduck_trampoline_flip_env_cfg(play=True),
        rl_cfg=MicroduckTrampolineFlipRlCfg,
        runner_cls=runner_cls,
    )

    # Pump-flip — the same, but it starts standing on the bed and the flip only
    # counts after a few bounces (make_..._env_cfg(pump_first=True)).
    register_mjlab_task(
        task_id="Mjlab-Trampoline-PumpFlip-MicroDuck",
        env_cfg=make_microduck_trampoline_flip_env_cfg(pump_first=True),
        play_env_cfg=make_microduck_trampoline_flip_env_cfg(play=True, pump_first=True),
        rl_cfg=MicroduckTrampolinePumpFlipRlCfg,
        runner_cls=runner_cls,
    )


    # Pump-flip curriculum — from drop starts (the flip that works) to standing
    # starts with 3 bounces first, in steps (PUMP_CURRICULUM).
    register_mjlab_task(
        task_id="Mjlab-Trampoline-PumpFlipCurriculum-MicroDuck",
        env_cfg=make_microduck_trampoline_flip_env_cfg(pump_first=True, curriculum=True),
        play_env_cfg=make_microduck_trampoline_flip_env_cfg(play=True, pump_first=True, curriculum=True),
        rl_cfg=MicroduckTrampolinePumpFlipRlCfg,
        runner_cls=runner_cls,
    )

    register_mjlab_task(
        task_id="Mjlab-Trampoline-PumpFlipCurriculumSlow-MicroDuck",
        env_cfg=make_microduck_trampoline_flip_env_cfg(pump_first=True, curriculum=True, stretch=2.0),
        play_env_cfg=make_microduck_trampoline_flip_env_cfg(play=True, pump_first=True, curriculum=True, stretch=2.0),
        rl_cfg=MicroduckTrampolinePumpFlipRlCfg,
        runner_cls=runner_cls,
    )

    # Bounce — start standing, pump to a steady bounce height, stay upright (no
    # flip): the first half of a bounce-then-flip policy pair.
    register_mjlab_task(
        task_id="Mjlab-Trampoline-Bounce-MicroDuck",
        env_cfg=make_microduck_trampoline_bounce_env_cfg(),
        play_env_cfg=make_microduck_trampoline_bounce_env_cfg(play=True),
        rl_cfg=MicroduckTrampolineBounceRlCfg,
        runner_cls=runner_cls,
    )

    register_mjlab_task(
        task_id="Mjlab-Trampoline-BounceHigh-MicroDuck",
        env_cfg=make_microduck_trampoline_bounce_env_cfg(target=0.30),
        play_env_cfg=make_microduck_trampoline_bounce_env_cfg(play=True, target=0.30),
        rl_cfg=MicroduckTrampolineBounceRlCfg,
        runner_cls=runner_cls,
    )

    register_mjlab_task(
        task_id="Mjlab-Trampoline-BounceHighPose-MicroDuck",
        env_cfg=make_microduck_trampoline_bounce_env_cfg(target=0.30, pose_std=BOUNCE_POSE_STD),
        play_env_cfg=make_microduck_trampoline_bounce_env_cfg(play=True, target=0.30, pose_std=BOUNCE_POSE_STD),
        rl_cfg=MicroduckTrampolineBounceRlCfg,
        runner_cls=runner_cls,
    )

    register_mjlab_task(
        task_id="Mjlab-Trampoline-BounceHighPoseTight-MicroDuck",
        env_cfg=make_microduck_trampoline_bounce_env_cfg(target=0.30, pose_std=1.0),
        play_env_cfg=make_microduck_trampoline_bounce_env_cfg(play=True, target=0.30, pose_std=1.0),
        rl_cfg=MicroduckTrampolineBounceRlCfg,
        runner_cls=runner_cls,
    )

    register_mjlab_task(
        task_id="Mjlab-Trampoline-FlipAnyPose-MicroDuck",
        env_cfg=make_microduck_trampoline_flip_anypose_env_cfg(),
        play_env_cfg=make_microduck_trampoline_flip_anypose_env_cfg(play=True),
        rl_cfg=MicroduckTrampolineFlipRlCfg,
        runner_cls=runner_cls,
    )
