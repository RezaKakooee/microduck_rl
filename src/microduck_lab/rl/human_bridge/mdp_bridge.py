"""Reward terms for crossing the brother (`microduck_bridge_sideways_env_cfg`).

World-frame terms: the bridge sits at a fixed place in every env's world
(env_spacing = 0), so the world y of her trunk IS her offset from his centre
line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def world_y_gaussian(env: "ManagerBasedRlEnv", std: float,
                     asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-(y / std)^2): stay over his centre line. >= 0: positive weight."""
    y = env.scene[asset_cfg.name].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.exp(-torch.square(y / std))


def heading_gaussian(env: "ManagerBasedRlEnv", target: float, std: float,
                     asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-(yaw error / std)^2), yaw from the root quaternion. >= 0."""
    q = env.scene[asset_cfg.name].data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    err = torch.atan2(torch.sin(yaw - target), torch.cos(yaw - target))
    return torch.exp(-torch.square(err / std))


def world_x_progress(env: "ManagerBasedRlEnv", max_speed: float,
                     asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Her world +x speed (across the gap), clipped to [0, max_speed].

    v1 of this task paid only for tracking and for staying centred; standing
    still on her ledge kept most of that and never risked a fall, so she learned
    to stand still. This term pays only for going across."""
    vx = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 0]
    return torch.clamp(vx, 0.0, max_speed)


def world_y_sq(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """y^2 off his centre line. A cost (>= 0): negative weight."""
    y = env.scene[asset_cfg.name].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.square(y)


def heading_err_sq(env: "ManagerBasedRlEnv", target: float,
                   asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """(yaw error)^2. A cost (>= 0): negative weight."""
    q = env.scene[asset_cfg.name].data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    err = torch.atan2(torch.sin(yaw - target), torch.cos(yaw - target))
    return torch.square(err)


def reset_on_route(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None,
                   start_x: tuple, route_x: tuple, p_start: float, y: tuple, yaw: tuple,
                   surface_x: tuple, surface_z: tuple, clearance: tuple,
                   edge_x: tuple = (0.0, 0.0), p_edge: float = 0.0,
                   asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Spawn her standing somewhere along the route, at the right height.

    With probability `p_start` she starts at the real start (her ledge);
    otherwise anywhere in `route_x` -- on his legs, trunk or head, or at the far
    edge -- so the hard middle gets practised from the first iteration (the
    reverse-curriculum rule in AGENTS.md). With probability `p_edge` she
    starts right in front of the step from her ledge onto him (`edge_x`).
    Height = the highest surface under
    her stance at that x (`surface_x/z`, measured on the baked set) plus a
    standing `clearance`."""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    dev = env.device
    asset = env.scene[asset_cfg.name]
    u = lambda lo_hi: torch.empty(n, device=dev).uniform_(*lo_hi)
    r = torch.rand(n, device=dev)
    x = torch.where(r < p_start, u(start_x),
                    torch.where(r < p_start + p_edge, u(edge_x), u(route_x)))
    sx = torch.tensor(surface_x, device=dev, dtype=torch.float)
    sz = torch.tensor(surface_z, device=dev, dtype=torch.float)
    i = torch.clamp(torch.searchsorted(sx, x) - 1, 0, len(sx) - 2)
    w = (x - sx[i]) / (sx[i + 1] - sx[i])
    z = sz[i] + w * (sz[i + 1] - sz[i]) + u(clearance)
    root = asset.data.default_root_state[env_ids].clone()
    pos = torch.stack([x, u(y), z], dim=-1) + env.scene.env_origins[env_ids]
    zero = torch.zeros(n, device=dev)
    rot = quat_mul(root[:, 3:7], quat_from_euler_xyz(zero, zero, u(yaw)))
    asset.write_root_link_pose_to_sim(torch.cat([pos, rot], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def crossed(env: "ManagerBasedRlEnv", x: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Her trunk is past world x (all of her is on the far ledge)."""
    return env.scene[asset_cfg.name].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0] > x


def crossed_bonus(env: "ManagerBasedRlEnv", x: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 on the step she gets across (the episode then ends). >= 0."""
    return crossed(env, x, asset_cfg).float()


def bridge_state(env: "ManagerBasedRlEnv", mid_x: float, surface_x: tuple, surface_z: tuple,
                 stand_height: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Where she is on the bridge, fed through the 6 body-pose command slots.

    The policy is otherwise blind: it cannot know when his ankles, battery box
    or head come under her feet, and training stalled at ~1.4 s episodes with
    most of them ending off his side. These 6 numbers are what a child
    crossing would see. The real robot cannot measure them, so a policy using
    them is a simulation expert, like the balance-board LQR expert.

      0  position along the route, (x - mid_x) / 0.3
      1  offset from his centre line, y / 0.05
      2  heading error, rad / 0.3
      3  surface rise 50 mm ahead (her left, +x), / 0.03
      4  surface rise 100 mm ahead, / 0.03
      5  how far her trunk sits below its standing height over the surface, / 0.03
    """
    a = env.scene[asset_cfg.name].data
    pos = a.root_link_pos_w - env.scene.env_origins
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    q = a.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    err = torch.atan2(torch.sin(yaw + torch.pi / 2), torch.cos(yaw + torch.pi / 2))
    sx = torch.tensor(surface_x, device=env.device, dtype=torch.float)
    sz = torch.tensor(surface_z, device=env.device, dtype=torch.float)

    def h(xq):
        i = torch.clamp(torch.searchsorted(sx, xq) - 1, 0, len(sx) - 2)
        w = torch.clamp((xq - sx[i]) / (sx[i + 1] - sx[i]), 0.0, 1.0)
        return sz[i] + w * (sz[i + 1] - sz[i])

    here = h(x)
    return torch.stack([(x - mid_x) / 0.3, y / 0.05, err / 0.3,
                        (h(x + 0.05) - here) / 0.03, (h(x + 0.10) - here) / 0.03,
                        (here + stand_height - z) / 0.03], dim=-1)


def fell_or_dropped(env: "ManagerBasedRlEnv", minimum_height: float, limit_angle: float,
                    asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 on the step she falls over or drops off him. Paired with a negative
    weight: once the per-step 'alive' rewards are small, a fall must still cost."""
    a = env.scene[asset_cfg.name].data
    tilted = a.projected_gravity_b[:, 2] > -torch.cos(torch.tensor(limit_angle, device=env.device))
    low = a.root_link_pos_w[:, 2] < minimum_height
    return (tilted | low).float()
