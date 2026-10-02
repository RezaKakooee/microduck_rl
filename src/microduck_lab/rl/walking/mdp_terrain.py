"""MDP functions for the terrain (slopes + stairs) velocity task.

Kept out of mdp.py on purpose: mdp.py is the shared module every task imports,
and this file only serves `microduck_velocity_terrain_env_cfg.py`.

The one function here is a terrain-level curriculum. mjlab ships
`terrain_levels_vel`, but its demotion rule is
``distance < |cmd| * episode_length * 0.5`` — a robot that walks 0.12 m/s
against a 0.3 m/s command (what the pretrained duck actually does) covers
2.4 m in 20 s, which is below that 3 m bar, so it would be demoted on every
episode and never leave level 0. The rule below is tile-relative instead, and
only judges envs that were commanded to move: standing / turn-in-place envs
neither rise nor fall.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def walk_move_masks(
    distance: torch.Tensor,
    cmd_lin_norm: torch.Tensor,
    size_x: float,
    promote_fraction: float = 0.25,
    demote_fraction: float = 0.08,
    command_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Promotion / demotion masks for the walking terrain curriculum.

    Args:
        distance: planar distance walked from the spawn origin, per env.
        cmd_lin_norm: norm of the commanded planar velocity, per env.
        size_x: sub-terrain tile length (m).
        promote_fraction: walked more than this fraction of a tile -> harder.
        demote_fraction: walked less than this fraction of a tile -> easier
            (early fall, or stuck on the obstacle).
        command_threshold: envs commanded below this are not judged at all.

    Pure tensor math so it can be unit-tested on CPU.
    """
    moving = cmd_lin_norm > command_threshold
    move_up = moving & (distance > promote_fraction * size_x)
    move_down = moving & (distance < demote_fraction * size_x) & ~move_up
    return move_up, move_down


def terrain_levels_walk(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    command_name: str = "twist",
    promote_fraction: float = 0.25,
    demote_fraction: float = 0.08,
    command_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Terrain-level curriculum keyed on distance walked per tile.

    Called at reset for the envs that just ended. Returns the same logging
    dict shape as mjlab's ``terrain_levels_vel`` (mean / max level, plus a
    per-sub-terrain mean in curriculum mode) so the tensorboard panels match.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    cmd_lin_norm = torch.norm(command[env_ids, :2], dim=1)
    move_up, move_down = walk_move_masks(
        distance,
        cmd_lin_norm,
        terrain_generator.size[0],
        promote_fraction=promote_fraction,
        demote_fraction=demote_fraction,
        command_threshold=command_threshold,
    )
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }
    # One column per sub-terrain type in curriculum mode -> per-type mean level.
    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    if terrain_origins is not None and terrain_origins.shape[1] == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])
    return result
