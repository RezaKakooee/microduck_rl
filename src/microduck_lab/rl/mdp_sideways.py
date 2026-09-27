"""MDP functions for the sideways-walking task.

Kept out of ``mdp.py`` on purpose, like the other lab tasks.

Why the walking recipe never learned to strafe, measured on the policies we
have (8 s at a 0.2 m/s sideways command, BAM, CPU MuJoCo):

* ``alpha_walking``: 2 mm left, 0 mm right.
* ``slopes_v1`` (trained here on the recipe): 155 mm left, but turned 61
  degrees doing it; 0 mm right.

Three things in the recipe make "ignore vy" the cheap answer:

1. ``track_linear_velocity`` is exp(-err^2 / 0.1). Standing still against a
   0.2 m/s sideways command still pays exp(-0.4) = 67 %.
2. The walking pose reward holds hip roll to a std of 0.05 rad. A sideways
   step needs roughly 0.15-0.2 rad of hip roll, so every step is taxed.
3. ``track_angular_velocity`` has a std of 0.7 rad/s, so drifting in yaw
   while stepping sideways costs nearly nothing.

This module adds the command bucket and the two tight tracking terms; the cfg
loosens the hip-roll std.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


class SidewaysVelocityCommand(VelocityCommandCommandOnly):
    """The recipe's command, plus a bucket of pure sideways commands.

    Independent uniform sampling almost never gives "vx ~ 0, |vy| large,
    wz ~ 0", which is exactly the skill. Same mechanism as the recipe's
    turn-in-place bucket, which it runs first.
    """

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        p = self.cfg.rel_sideways_envs
        if p <= 0.0 or len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        ids = env_ids[r.uniform_(0.0, 1.0) < p]
        n = len(ids)
        if n == 0:
            return
        lo, hi = self.cfg.sideways_speed
        if self.cfg.sideways_sign:
            sign = torch.full((n,), float(self.cfg.sideways_sign), device=self.device)
        else:
            sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[ids, 0] = torch.empty(n, device=self.device).uniform_(
            -self.cfg.sideways_vx_noise, self.cfg.sideways_vx_noise)
        self.vel_command_b[ids, 1] = sign * torch.empty(n, device=self.device).uniform_(lo, hi)
        self.vel_command_b[ids, 2] = torch.empty(n, device=self.device).uniform_(
            -self.cfg.sideways_wz_noise, self.cfg.sideways_wz_noise)
        self.is_standing_env[ids] = False
        self.vel_command_w[ids] = self.vel_command_b[ids]


@dataclass(kw_only=True)
class SidewaysVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    rel_sideways_envs: float = 0.0          # fraction of resamples that are pure sideways
    sideways_speed: tuple = (0.10, 0.25)    # |vy| range for the bucket, m/s
    sideways_vx_noise: float = 0.03         # small, so the input stays alive
    sideways_wz_noise: float = 0.10
    sideways_sign: float = 0.0              # 0: random left/right; +1 / -1: always that way

    def build(self, env: "ManagerBasedRlEnv") -> SidewaysVelocityCommand:
        return SidewaysVelocityCommand(self, env)


def track_lateral_velocity(env: "ManagerBasedRlEnv", std: float, command_name: str,
                           asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-(vy_cmd - vy)^2 / std^2), body frame. >= 0: positive weight."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = cmd[:, 1] - asset.data.root_link_lin_vel_b[:, 1]
    return torch.exp(-torch.square(err) / std**2)


def track_yaw_rate_tight(env: "ManagerBasedRlEnv", std: float, command_name: str,
                         asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-(wz_cmd - wz)^2 / std^2). The recipe's own term uses std 0.7 rad/s,
    under which yaw drift while strafing is almost free. >= 0: positive weight."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(err) / std**2)


def feet_slip_linear(env: "ManagerBasedRlEnv", sensor_name: str, command_name: str,
                     command_threshold: float = 0.01,
                     asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Sum over feet of |horizontal foot speed| while the foot touches.

    The recipe's feet_slip squares the speed, so a 5 cm/s shuffle costs
    0.0025 per foot: sliding was free, and sideways v1/v2 learned to shuffle
    (95th-percentile foot lift 0.5-4.5 mm, against 6-7 mm for the forward
    walker). Linear, the same slide costs 0.05 per foot. A cost (>= 0):
    negative weight."""
    asset = env.scene[asset_cfg.name]
    contact = env.scene[sensor_name]
    cmd = env.command_manager.get_command(command_name)
    active = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) > command_threshold).float()
    in_contact = (contact.data.found > 0).float()
    v = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    return torch.sum(v * in_contact, dim=1) * active


def swing_lift(env: "ManagerBasedRlEnv", sensor_name: str, height_sensor_name: str,
               low: float, high: float, command_name: str,
               command_threshold: float = 0.01) -> torch.Tensor:
    """Pay each airborne foot for being clearly up, every step it is up.

    The foot site sits 10.3 mm above the floor when the sole is flat, and the
    height sensor measures from the site. So `low` = 0.010 means "sole just
    off the ground" and pays nothing; `high` = 0.025 (~15 mm of real lift)
    pays 1. A foot hovering 0.5 mm -- the v1-v3 shuffle -- earns nothing,
    though the contact sensor already calls it airborne. >= 0: positive weight."""
    contact = env.scene[sensor_name]
    heights = env.scene[height_sensor_name].data.heights            # [B, feet], site above terrain
    cmd = env.command_manager.get_command(command_name)
    active = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) > command_threshold).float()
    in_air = (contact.data.found == 0).float()
    lift = torch.clamp((heights - low) / (high - low), 0.0, 1.0)
    return torch.sum(lift * in_air, dim=1) * active
