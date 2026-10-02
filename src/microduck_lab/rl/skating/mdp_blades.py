"""MDP functions for the skate-blade task (anisotropic foot-floor contact).

Kept out of ``mdp.py`` on purpose: this is one task's physics plumbing, not a
shared reward vocabulary.

Physics facts these functions rest on (all measured; ``src/microduck_lab/tests/test_blades_cfg.py``
locks them in):

* A ``<contact><pair>`` overrides geom friction and carries TWO sliding
  coefficients, one per contact-frame tangent: ``pair_friction[0]`` acts along
  tangent1, ``pair_friction[1]`` along tangent2.
* For any contact with a +Z plane the contact frame is WORLD-FIXED:
  ``normal=+Z, tangent1=+Y, tangent2=-X`` — for a box and for the duck's sole
  alike, whatever the foot's yaw. MuJoCo (C and Warp) builds the frame from
  the normal alone (``mju_makeFrame`` / ``make_frame``). So ``pair_friction[0]``
  is the world-Y coefficient and ``pair_friction[1]`` the world-X coefficient,
  and there is no way to pin the frame to the foot.

A static pair is therefore only a "blade" while the foot points along world X.
The real blade is a friction ELLIPSE in the foot frame — radius ``mu_along``
on the sole's long axis ``u``, ``mu_across`` on the perpendicular ``v`` — and
MuJoCo can only hold an axis-aligned one. :func:`project_blade_friction` runs
every control step (mjlab ``mode="step"`` event) and writes the best
axis-aligned stand-in for each foot's current yaw. Two rules:

``"load"`` (default) — the radius of the true ellipse along the direction the
foot is currently loaded (its tangential contact force ``d``), written to both
slots::

    r(d) = 1 / sqrt((d.u)^2 / mu_along^2 + (d.v)^2 / mu_across^2)

so a foot pushing perpendicular to its blade grips (``mu_across``), a foot
sliding along it glides (``mu_along``), and a foot turned out 30 degrees but
shoved straight back gets ~0.09 — as a real blade would — instead of the
~0.5 the bounding box would hand it. Unloaded / airborne feet get the
ellipse's intercepts with the world axes (the same formula with ``d`` = X
and ``d`` = Y). The force is one control step stale, so a push that changes
direction gets one 20 ms step at the old radius. That is the honest limit.

``"bbox"`` — the bounding box of the rotated ellipse::

    mu_x = sqrt(mu_along^2 cos^2 + mu_across^2 sin^2),  mu_y likewise swapped

Exact at 0 and 90 degrees, generous in between: at 30 degrees a straight-back
push already gets ~0.5, so a walking gait with turned-out feet never has to
skate. Kept for comparison; measured on the pretrained walker it recovers
mu_x 0.5-0.6 from the gait's natural 30-37 degree foot yaw.

Per-env domain randomisation lives in :func:`randomize_blade_friction`
(``mode="reset"``): it samples ``mu_along`` / ``mu_across`` per env into
buffers on the env; the projection reads them. Writes are absolute, so nothing
accumulates across resets.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

# Which pair_friction slot governs which world axis for a +Z plane contact.
# Measured: friction=[0.02, 1.2] slides 5.1 m along world Y and 0.19 m along
# world X; the contact frame reads normal=(0,0,1), t1=(0,1,0), t2=(-1,0,0).
PAIR_AXIS_WORLD_Y = 0  # tangent1
PAIR_AXIS_WORLD_X = 1  # tangent2 (sign of the axis is irrelevant for friction)

RULES = ("load", "bbox")
# Tangential force below this counts as "unloaded" (the duck weighs ~8 N).
LOAD_EPS_N = 0.2

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_STATE_ATTR = "_blade_state"


@dataclass
class BladeState:
    """Per-env blade bookkeeping, cached on the env after the first call."""

    pair_ids: torch.Tensor  # (F,) global pair ids, one per foot
    body_ids: torch.Tensor  # (F,) entity-local body index of each foot
    axis_b: torch.Tensor  # (F, 3) sole long axis in each foot body frame
    mu_along: torch.Tensor  # (N,) along-blade coefficient per env
    mu_across: torch.Tensor  # (N,) across-blade coefficient per env


# ── Pure maths (unit-tested) ─────────────────────────────────────────────────


def blade_projection(
    mu_along: torch.Tensor, mu_across: torch.Tensor, cos2_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """"bbox" rule: world-axis (mu_x, mu_y) as the bounding box of the yawed ellipse."""
    sin2 = 1.0 - cos2_yaw
    a2 = mu_along * mu_along
    c2 = mu_across * mu_across
    mu_x = torch.sqrt(a2 * cos2_yaw + c2 * sin2)
    mu_y = torch.sqrt(a2 * sin2 + c2 * cos2_yaw)
    return mu_x, mu_y


def blade_radius(
    mu_along: torch.Tensor, mu_across: torch.Tensor, du2: torch.Tensor
) -> torch.Tensor:
    """Radius of the blade ellipse along a unit in-plane direction d.

    ``du2`` is (d . u)^2 with u the blade axis; (d . v)^2 = 1 - du2.
    """
    dv2 = 1.0 - du2
    return 1.0 / torch.sqrt(du2 / (mu_along * mu_along) + dv2 / (mu_across * mu_across))


def blade_axis_radii(
    mu_along: torch.Tensor, mu_across: torch.Tensor, cos2_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ellipse radii along world X and Y for a blade yawed by ``yaw`` (cos^2 given).

    Used for unloaded feet under the "load" rule. (X . u)^2 = cos^2,
    (Y . u)^2 = sin^2.
    """
    return blade_radius(mu_along, mu_across, cos2_yaw), blade_radius(
        mu_along, mu_across, 1.0 - cos2_yaw
    )


def sole_long_axis_in_body(mj_model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Unit vector of a mesh geom's longest extent, in its parent body frame.

    Mesh vertices live in the geom frame; ``geom_quat`` maps geom -> body. The
    principal axis (SVD) of the vertex cloud is the sole's fore-aft direction.
    """
    mid = int(mj_model.geom_dataid[geom_id])
    if mid < 0:
        raise ValueError(f"geom {geom_id} is not a mesh; cannot find its long axis")
    adr, num = int(mj_model.mesh_vertadr[mid]), int(mj_model.mesh_vertnum[mid])
    verts = np.asarray(mj_model.mesh_vert[adr : adr + num], dtype=np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(mj_model.geom_quat[geom_id], dtype=np.float64))
    verts_b = verts @ rot.reshape(3, 3).T
    verts_b -= verts_b.mean(axis=0)
    _, _, vt = np.linalg.svd(verts_b, full_matrices=False)
    axis = vt[0]
    return axis / np.linalg.norm(axis)


# ── State ────────────────────────────────────────────────────────────────────


def _env_ids_tensor(env: ManagerBasedRlEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[
            slice(None) if env_ids is None else env_ids
        ]
    return env_ids.to(env.device, dtype=torch.long)


def get_blade_state(
    env: ManagerBasedRlEnv,
    pair_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> BladeState:
    """Resolve pair ids, foot bodies and sole axes once; cache on the env."""
    state = getattr(env, _STATE_ATTR, None)
    if state is not None and state.mu_along.shape[0] == env.num_envs:
        return state

    mjm: mujoco.MjModel = env.sim.mj_model
    asset = env.scene[asset_cfg.name]
    body_names = tuple(asset.body_names)

    pair_ids, body_ids, axes = [], [], []
    for name in pair_names:
        pid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_PAIR, name)
        if pid < 0:
            raise ValueError(
                f"[blades] pair '{name}' not in the compiled model — did the "
                "spec_fn that adds the blade pairs run?"
            )
        foot_geom = None
        for gid in (int(mjm.pair_geom1[pid]), int(mjm.pair_geom2[pid])):
            bname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[gid]))
            local = (bname or "").split("/")[-1]
            if local in body_names:
                foot_geom = gid
                body_ids.append(body_names.index(local))
                break
        if foot_geom is None:
            raise ValueError(f"[blades] pair '{name}' touches no body of '{asset_cfg.name}'")
        pair_ids.append(pid)
        axes.append(sole_long_axis_in_body(mjm, foot_geom))

    dev = env.device
    state = BladeState(
        pair_ids=torch.tensor(pair_ids, dtype=torch.long, device=dev),
        body_ids=torch.tensor(body_ids, dtype=torch.long, device=dev),
        axis_b=torch.tensor(np.stack(axes), dtype=torch.float32, device=dev),
        mu_along=torch.ones(env.num_envs, device=dev),
        mu_across=torch.ones(env.num_envs, device=dev),
    )
    setattr(env, _STATE_ATTR, state)
    return state


def blade_axis_world_xy(state: BladeState, asset, env_ids: torch.Tensor) -> torch.Tensor:
    """Unit in-plane blade direction u of each foot. Shape (n, F, 2).

    A vertical sole has no in-plane direction (and is off the ground); treat
    it as pointing along world X.
    """
    quat = asset.data.body_link_quat_w[env_ids][:, state.body_ids]  # (n, F, 4)
    n, nfeet = quat.shape[:2]
    axis = state.axis_b.unsqueeze(0).expand(n, nfeet, 3)
    d = quat_apply(quat.reshape(-1, 4), axis.reshape(-1, 3)).reshape(n, nfeet, 3)
    dxy = d[..., :2]
    norm = dxy.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(dxy)
    fallback[..., 0] = 1.0
    return torch.where(norm < 1e-3, fallback, dxy / norm.clamp_min(1e-9))


def _write_projection(
    env: ManagerBasedRlEnv,
    state: BladeState,
    asset,
    env_ids: torch.Tensor,
    rule: str,
    sensor_name: str | None,
    load_eps: float,
) -> None:
    if rule not in RULES:
        raise ValueError(f"[blades] unknown rule '{rule}', expected one of {RULES}")
    u = blade_axis_world_xy(state, asset, env_ids)  # (n, F, 2)
    cos2 = u[..., 0] ** 2
    mu_a = state.mu_along[env_ids].unsqueeze(1)
    mu_c = state.mu_across[env_ids].unsqueeze(1)

    if rule == "bbox":
        mu_x, mu_y = blade_projection(mu_a, mu_c, cos2)
    else:
        mu_x, mu_y = blade_axis_radii(mu_a, mu_c, cos2)
        if sensor_name is not None:
            # Global-frame net contact force per foot (reduce="netforce"); the
            # plane is horizontal so its XY part is the tangential load.
            nfeet = u.shape[1]
            force = env.scene[sensor_name].data.force[env_ids][:, :nfeet, :2]
            mag = force.norm(dim=-1)
            loaded = mag > load_eps
            d = force / mag.clamp_min(1e-9).unsqueeze(-1)
            du2 = (d * u).sum(-1) ** 2
            r = blade_radius(mu_a, mu_c, du2.clamp(0.0, 1.0))
            mu_x = torch.where(loaded, r, mu_x)
            mu_y = torch.where(loaded, r, mu_y)

    pf = env.sim.model.pair_friction  # (N, npair, 5) after per-world expansion
    env_grid, pair_grid = torch.meshgrid(env_ids, state.pair_ids, indexing="ij")
    pf[env_grid, pair_grid, PAIR_AXIS_WORLD_X] = mu_x.to(pf.dtype)
    pf[env_grid, pair_grid, PAIR_AXIS_WORLD_Y] = mu_y.to(pf.dtype)


# ── Events ───────────────────────────────────────────────────────────────────


@requires_model_fields("pair_friction")
def randomize_blade_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    along_range: tuple[float, float],
    across_range: tuple[float, float],
    pair_names: tuple[str, ...],
    rule: str = "load",
    sensor_name: str | None = None,
    load_eps: float = LOAD_EPS_N,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset-mode DR: sample per-env blade coefficients and apply them.

    ``along_range`` / ``across_range`` are absolute sliding coefficients (not
    scales), rewritten by :func:`blade_friction_curriculum`. Both feet of one
    env share a sample: ice quality is a property of the rink, not the foot.
    """
    state = get_blade_state(env, pair_names, asset_cfg)
    ids = _env_ids_tensor(env, env_ids)
    n = len(ids)
    state.mu_along[ids] = torch.empty(n, device=env.device).uniform_(*along_range)
    state.mu_across[ids] = torch.empty(n, device=env.device).uniform_(*across_range)
    # Body quats / forces are one step stale inside a reset; the step-mode
    # projection rewrites this before the next physics step. Writing here
    # keeps the model sane even if only the reset event is wired.
    _write_projection(env, state, env.scene[asset_cfg.name], ids, rule, sensor_name, load_eps)


@requires_model_fields("pair_friction")
def project_blade_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pair_names: tuple[str, ...],
    rule: str = "load",
    sensor_name: str | None = None,
    load_eps: float = LOAD_EPS_N,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Step-mode event: rewrite each foot's pair friction for its current yaw / load."""
    state = get_blade_state(env, pair_names, asset_cfg)
    ids = _env_ids_tensor(env, env_ids)
    _write_projection(env, state, env.scene[asset_cfg.name], ids, rule, sensor_name, load_eps)


def blade_friction_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
) -> torch.Tensor:
    """Step-staged ranges for the blade DR event.

    ``stages`` is a list of ``{"step": int, "along": (lo, hi), "across": (lo, hi)}``;
    the latest stage whose step has elapsed is written onto the live event term
    (via the manager — ``env.cfg`` is a dead copy). Returns the along-range
    midpoint so ``Curriculum/<name>`` shows where the ramp is.
    """
    del env_ids
    current = stages[0]
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            current = stage
    term_cfg = env.event_manager.get_term_cfg(event_name)
    term_cfg.params["along_range"] = current["along"]
    term_cfg.params["across_range"] = current["across"]
    along = current["along"]
    return torch.tensor([0.5 * (along[0] + along[1])])
