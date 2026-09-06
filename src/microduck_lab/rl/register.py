"""Registers our five RL tasks with mjlab.

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
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from microduck_lab.rl.microduck_arabesque_env_cfg import (
    make_microduck_arabesque_env_cfg,
    MicroduckArabesqueRlCfg,
)
from microduck_lab.rl.microduck_spiral_env_cfg import (
    make_microduck_spiral_env_cfg,
    MicroduckSpiralRlCfg,
)
from microduck_lab.rl.microduck_velocity_ice_env_cfg import (
    make_microduck_velocity_ice_env_cfg,
    MicroduckIceRlCfg,
)
from microduck_lab.rl.microduck_velocity_terrain_env_cfg import (
    make_microduck_velocity_terrain_env_cfg,
    MicroduckSlopesRlCfg,
)
from microduck_lab.rl.microduck_velocity_blades_env_cfg import (
    make_microduck_velocity_blades_env_cfg,
    MicroduckBladesRlCfg,
)

def register_all(runner_cls) -> None:
    """Register our five tasks. Called once, from `mjlab_microduck.tasks`."""
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
