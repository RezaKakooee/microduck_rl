"""MDP functions for the sideways beam task (`microduck_velocity_sideways_beam_env_cfg`).

Kept out of `mdp.py`, like the other lab tasks.

Everything that needs "where is she on her beam" reads one `BeamTracker`,
created on first use and stored on the env. It holds:

* tables of every tile (from `beam_terrain.beam_profile`, the same function
  that built the geometry): surface height, width, gap mask, spawn points;
* per env: the spawn x, the best x reached so far (the progress ratchet), an
  EMA of the ratchet speed (the progress gate), and how long she must still
  wait on the start platform before she is told to walk.

Frames. The terrain origin of a tile is the start of its beam, on the beam
axis, at the start platform top. So for env i, with o = env_origins[i]:
x_rel = x - o.x is the distance along the beam, y - o.y the offset from its
axis, and the surface height is o.z + h(x_rel).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from microduck_lab.rl.human_bridge.rl_policy import beam_terrain as T

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
DX = 0.005                  # table resolution along the beam (m)
SPAWN_K = 48                # spawn candidates per tile
SITE_Z = 0.0                # foot site above the beam top, sole flat on it: measured -0.3 mm
                            # median (p10 -1.3, p90 +1.6) over 148k contact samples (s3_train/diag_law.py).
                            # The site is at the sole bottom, not 10.3 mm above it.
HEADING = -math.pi / 2      # she faces -y; her left (+vy) is world +x, along the beam
PROFILE_AHEAD = tuple(0.03 * k for k in range(-2, 10))   # critic height samples (m)
GAP_VALUE = -0.09           # critic: height reported over a gap (m)
# Costs saturate: cost = cap^2 (1 - exp(-(v / cap)^2)). Near 0 this is v^2; it never
# passes cap^2. With the cfg weights (-3000, -10) the worst case of BOTH is
# 3000 x 0.020^2 + 10 x 0.25^2 = 1.825 per second: 5 s of it (9.1) cost less than
# one fall (500 x dt 0.02 = 10). Uncapped, a 50 mm drift cost 7.5/s and a turned-
# round duck 99/s, so after a drift a fall was the cheaper way out (s3_review).
OFFSET_CAP = 0.020          # m
HEADING_CAP = 0.25          # rad


def _yaw(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----------------------------------------------------------------- tracker ---

class BeamTracker:
    """Tables of all tiles + per-env progress state. One per env object."""

    def __init__(self, env: "ManagerBasedRlEnv", spawn_x_max: float, gate_tau: float,
                 gate_frac: float, gate_init: float):
        self.env = env
        dev = env.device
        terrain = env.scene.terrain
        gen = terrain.cfg.terrain_generator
        subs = list(gen.sub_terrains.values())
        assert all(isinstance(s, T.BeamLaneTerrainCfg) for s in subs), "not a beam terrain"
        self.kinds = [s.kind for s in subs]
        R, C = gen.num_rows, len(subs)
        top, width, gap, stance, spawn = [], [], [], [], []
        k_st = int(round(0.065 / DX))
        for r in range(R):
            row = [[], [], [], [], []]
            for c, sub in enumerate(subs):
                segs = sub.profile(r)
                xs, t, w, g = T.sample_profile(segs, DX, gen.size[0])
                st = np.array([t[max(0, i - k_st):i + k_st + 1].max() for i in range(len(t))])
                sp = T.spawn_points(segs, SPAWN_K, spawn_x_max, seed=1000 * r + c)
                for lst, v in zip(row, (t, w, g, st, sp)):
                    lst.append(v)
            for lst, v in zip((top, width, gap, stance, spawn), row):
                lst.append(v)
        f = lambda a, dt=torch.float: torch.tensor(np.array(a), device=dev, dtype=dt)
        self.top, self.width, self.stance, self.spawn = f(top), f(width), f(stance), f(spawn)
        self.gap = f(gap, torch.bool)
        self.nx = self.top.shape[-1]
        self.difficulty = torch.tensor([T.difficulty(r, R) for r in range(R)], device=dev)
        n = env.num_envs
        self.spawn_x = torch.zeros(n, device=dev)
        self.best_x = torch.zeros(n, device=dev)
        self.delta = torch.zeros(n, device=dev)
        self.ema_v = torch.full((n,), gate_init, device=dev)
        self.steps = torch.zeros(n, device=dev)          # env steps since the spawn
        self.wait = torch.zeros(n, device=dev)           # s to stand on the start platform first
        self.gate_tau, self.gate_frac, self.gate_init = gate_tau, gate_frac, gate_init
        self._step = -1

    # -- lookups -----------------------------------------------------------
    def tile(self):
        t = self.env.scene.terrain
        return t.terrain_levels, t.terrain_types

    def lookup(self, table: torch.Tensor, x_rel: torch.Tensor) -> torch.Tensor:
        """table[level, type, x] for every env; x_rel is [B] or [B, k]."""
        lvl, typ = self.tile()
        idx = torch.clamp(torch.round(x_rel / DX).long(), 0, self.nx - 1)
        if x_rel.dim() == 2:
            return table[lvl[:, None], typ[:, None], idx]
        return table[lvl, typ, idx]

    def rel(self):
        """Her trunk in the tile frame: x along the beam, y off its axis, z."""
        a = self.env.scene["robot"].data
        p = a.root_link_pos_w - self.env.scene.env_origins
        return p[:, 0], p[:, 1], p[:, 2]

    def heading_err(self) -> torch.Tensor:
        return _wrap(_yaw(self.env.scene["robot"].data.root_link_quat_w) - HEADING)

    # -- per-step state ----------------------------------------------------
    def update(self) -> None:
        env = self.env
        if self._step == env.common_step_counter:
            return
        self._step = env.common_step_counter
        x, _, _ = self.rel()
        self.delta = torch.clamp(x - self.best_x, min=0.0)
        self.best_x = self.best_x + self.delta
        self.steps = self.steps + 1
        a = env.step_dt / self.gate_tau
        ema = self.ema_v + a * (self.delta / env.step_dt - self.ema_v)
        # While she is told to stand the gate is 1 anyway; hold the EMA at its
        # start value so the gate is not shut the moment she is told to walk.
        self.ema_v = torch.where(self.standing(), torch.full_like(ema, self.gate_init), ema)

    def start(self, env_ids: torch.Tensor, x_rel: torch.Tensor, wait_s: torch.Tensor | None = None) -> None:
        self.spawn_x[env_ids] = x_rel
        self.best_x[env_ids] = x_rel
        self.delta[env_ids] = 0.0
        self.ema_v[env_ids] = self.gate_init
        self.steps[env_ids] = 0.0
        self.wait[env_ids] = 0.0 if wait_s is None else wait_s

    def waiting(self) -> torch.Tensor:
        """True while she still stands on the start platform (the story's wait)."""
        return self.steps * self.env.step_dt < self.wait

    def standing(self) -> torch.Tensor:
        cmd = self.env.command_manager.get_command("twist")
        return torch.norm(cmd, dim=1) < 1e-6

    def gate(self) -> torch.Tensor:
        """1 while she keeps setting new best positions at >= gate_frac of the
        commanded speed (EMA over gate_tau), falling to 0 when she stops or
        marches in place. Always 1 for zero-command (standing) envs."""
        self.update()
        vy = self.env.command_manager.get_command("twist")[:, 1].abs()
        g = torch.clamp(self.ema_v / torch.clamp(self.gate_frac * vy, min=1e-3), 0.0, 1.0)
        return torch.where(self.standing(), torch.ones_like(g), g)


# gate_frac 0.3 (was 0.5): careful slow steps on a hard spot keep most of their pay.
# At vy 0.11 the gate is fully open from 33 mm/s of new ground.
TRACKER_PARAMS = dict(spawn_x_max=T.X_END - 1.0, gate_tau=0.5, gate_frac=0.3, gate_init=0.06)


def tracker(env: "ManagerBasedRlEnv") -> BeamTracker:
    tr = getattr(env, "_beam_tracker", None)
    if tr is None:
        tr = BeamTracker(env, **TRACKER_PARAMS)
        env._beam_tracker = tr
    return tr


# ----------------------------------------------------------------- command ---

class BeamSteeringCommand(UniformVelocityCommand):
    """The deployment steering law, evaluated every step (m_beam/beam.py Steer):

        vx = clip(vx0 + k_y * y_off,    +-vx_max)   her forward is world -y:
                                                    y_off > 0 -> step forward
        vy = U(vy_range), held until the next resample
        wz = clip(wz0 - k_yaw * heading_err, +-wz_max)

    vx0 / wz0 get a small per-resample jitter and a small piecewise-constant
    noise is added, so the inputs never sit on one value.

    The command is exactly zero only while she waits on the start platform
    (`BeamTracker.waiting`, set by `reset_on_beam`: the story's wait before
    she walks). On the beam the law is always on: a zero command there
    switched the steering off and she walked off (s3_review: ~21 % of
    episodes). `rel_standing_envs` still works but the task sets it to 0.

    vx_max 0.3: vx is her only offset sensor. At 0.1 the input clipped at
    y < -9 mm and y > +31 mm (vx0 -0.055, k_y 5); at 0.3 it clips at -49 / +71 mm.
    """

    cfg: "BeamSteeringCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        z = lambda: torch.zeros(n, device=self.device)
        self.vy_base, self.vx0, self.wz0, self.vx_noise, self.wz_noise = z(), z(), z(), z(), z()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n, c, d = len(env_ids), self.cfg, self.device
        u = lambda lo, hi: torch.empty(n, device=d).uniform_(lo, hi)
        self.vy_base[env_ids] = u(*c.vy_range)
        self.vx0[env_ids] = c.vx0 + u(-c.vx0_jitter, c.vx0_jitter)
        self.wz0[env_ids] = c.wz0 + u(-c.wz0_jitter, c.wz0_jitter)
        self.vx_noise[env_ids] = u(-c.vx_noise, c.vx_noise)
        self.wz_noise[env_ids] = u(-c.wz_noise, c.wz_noise)
        self.is_standing_env[env_ids] = torch.rand(n, device=d) < c.rel_standing_envs
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False

    def steer(self, y_off: torch.Tensor, heading_err: torch.Tensor):
        c = self.cfg
        vx = torch.clamp(self.vx0 + c.k_y * y_off + self.vx_noise, -c.vx_max, c.vx_max)
        wz = torch.clamp(self.wz0 - c.k_yaw * heading_err + self.wz_noise, -c.wz_max, c.wz_max)
        return vx, wz

    def _update_command(self) -> None:
        origin_y = self._env.scene.env_origins[:, 1]
        y_off = self.robot.data.root_link_pos_w[:, 1] - origin_y
        err = _wrap(_yaw(self.robot.data.root_link_quat_w) - self.cfg.heading)
        vx, wz = self.steer(y_off, err)
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = self.vy_base
        self.vel_command_b[:, 2] = wz
        stand = self.is_standing_env | tracker(self._env).waiting()
        self.vel_command_b[stand] = 0.0
        self.vel_command_w[:] = self.vel_command_b


@dataclass(kw_only=True)
class BeamSteeringCommandCfg(UniformVelocityCommandCfg):
    vy_range: tuple = (0.08, 0.14)
    k_y: float = 5.0
    k_yaw: float = 2.0
    vx0: float = -0.055
    wz0: float = 0.06
    vx_max: float = 0.30
    wz_max: float = 0.40
    heading: float = HEADING
    vx0_jitter: float = 0.01
    wz0_jitter: float = 0.02
    vx_noise: float = 0.005
    wz_noise: float = 0.02

    def build(self, env: "ManagerBasedRlEnv") -> BeamSteeringCommand:
        return BeamSteeringCommand(self, env)


# ------------------------------------------------------------------- robot ---

# Her parts that collide in the story (story.Judge: she.solid, 39 geoms), by
# (body, mesh) on the walking model. The soles already collide. The foot
# shells may touch (Judge counts them as feet); every other part ends the
# episode when it touches the terrain.
BODY_HIT = (
    ("ankle_left", "ankle_left"), ("ankle_right", "ankle_right"),
    ("bearing_roll", "xl330"), ("hip_l", "hip_l"), ("hip_l_2", "hip_l"),
    ("jaw_soft", "bottom_head_shell"), ("jaw_soft", "top_head_shell"), ("jaw_soft", "xl330"),
    ("jaw_soft", "jaw"), ("leg", "leg"), ("leg", "xl330"), ("leg_2", "leg"), ("leg_2", "xl330"),
    ("neck", "neck"), ("neck", "xl330"),
    ("trunk_base", "left_shell"), ("trunk_base", "right_shell"), ("trunk_base", "trunk_base"),
    ("trunk_base", "np_f970"), ("trunk_base", "power_support"), ("trunk_base", "xl330"),
    ("upper_leg_left", "upper_leg_left"), ("upper_leg_left", "upper_leg_rigidity_plate"),
    ("upper_leg_left", "xl330"), ("upper_leg_right", "upper_leg_right"),
    ("upper_leg_right", "upper_leg_rigidity_plate"), ("upper_leg_right", "xl330"),
    ("yaw2roll", "xl330"), ("yaw_roll_motion", "xl330"),
)
FOOT_SHELL = (("ankle_left", "foot_left"), ("ankle_right", "foot_right"))
# Collision bits: the terrain gets contype 1|8, these parts contype 0 /
# conaffinity 8. So they touch the terrain and nothing else (not her soles, not
# each other, not the self-collision geoms on bit 2).
TERRAIN_CONTYPE = 1 | 8
SOLID_CONAFFINITY = 8


def name_solid_geoms(spec) -> dict:
    """Name her story-solid visual geoms `bodyhit_*` / `footshell_*` (in place).
    Returns {prefix: count}. A CollisionCfg matching those names turns them on."""
    import mujoco
    hit, shell = set(BODY_HIT), set(FOOT_SHELL)
    counts = {"bodyhit": 0, "footshell": 0}
    for body in spec.bodies:
        for k, geom in enumerate(body.geoms):
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH or geom.group != 2:
                continue
            key = (body.name, geom.meshname)
            prefix = "bodyhit" if key in hit else "footshell" if key in shell else None
            if prefix is None:
                continue
            geom.name = f"{prefix}_{body.name}_{geom.meshname}_{k}"
            counts[prefix] += 1
    return counts


def beam_walk_spec():
    """The walking model with her story-solid parts named (see BODY_HIT)."""
    from mjlab_microduck.robot.microduck_constants import get_walk_spec
    spec = get_walk_spec()
    name_solid_geoms(spec)
    return spec


def terrain_bits(spec) -> None:
    """Scene spec_fn: every terrain geom also collides with bit 8."""
    for geom in spec.body("terrain").geoms:
        geom.contype = TERRAIN_CONTYPE
        geom.conaffinity = 1


# ----------------------------------------------------------------- rewards ---

def progress_ratchet(env: "ManagerBasedRlEnv", speed_cap: float = 1.25) -> torch.Tensor:
    """New ground along the beam this step / dt: pays only when her trunk x
    passes its best so far. Rocking or stepping back and forth pays nothing.
    The pay rate is capped at speed_cap x the commanded vy: ground gained
    faster (rushing, lunging or falling toward +x) is not paid. Uncapped, 5
    PPO iterations already made her 54% faster with 1.5-2.6x more falls on
    rows 3-9 (s5_review). >= 0: positive weight (per metre)."""
    tr = tracker(env)
    tr.update()
    vy = env.command_manager.get_command("twist")[:, 1].abs()
    rate = torch.minimum(tr.delta / env.step_dt, speed_cap * vy)
    return rate * (~tr.standing()).float()


class gated:
    """Any reward term, times the progress gate (see BeamTracker.gate).

    Marching in place, standing on the beam or rocking earns nothing from the
    wrapped term once the EMA of new ground drops well below gate_frac (0.3) x
    the commanded speed. params: inner (function or class term), inner_params (its params)."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        inner, ip = cfg.params["inner"], cfg.params["inner_params"]
        for v in ip.values():
            if isinstance(v, SceneEntityCfg):
                v.resolve(env.scene)
        if inspect.isclass(inner):
            inner = inner(cfg=RewardTermCfg(func=inner, weight=1.0, params=ip), env=env)
        self._inner = inner

    def __call__(self, env, inner, inner_params):
        del inner
        return self._inner(env, **inner_params) * tracker(env).gate()

    def reset(self, env_ids=None):
        r = getattr(self._inner, "reset", None)
        if r is not None:
            r(env_ids)


def standing_still(env: "ManagerBasedRlEnv", std_lin: float = 0.03, std_yaw: float = 0.2,
                   asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Zero-command envs only: exp(-|v_xy|^2 / std_lin^2) * exp(-wz^2 / std_yaw^2). >= 0."""
    a = env.scene[asset_cfg.name].data
    v = torch.sum(torch.square(a.root_link_lin_vel_b[:, :2]), dim=1)
    w = torch.square(a.root_link_ang_vel_b[:, 2])
    return torch.exp(-v / std_lin**2) * torch.exp(-w / std_yaw**2) * tracker(env).standing().float()


def saturating_sq(v: torch.Tensor, cap: float) -> torch.Tensor:
    """cap^2 (1 - exp(-(v / cap)^2)): v^2 near 0, never above cap^2, and still a
    small gradient back towards 0 past the cap."""
    return cap**2 * (1.0 - torch.exp(-torch.square(v / cap)))


def beam_offset_sq(env: "ManagerBasedRlEnv", cap: float = OFFSET_CAP) -> torch.Tensor:
    """(trunk y - beam axis y)^2, saturating at cap^2 (see OFFSET_CAP). A cost (>= 0): negative weight."""
    _, y, _ = tracker(env).rel()
    return saturating_sq(y, cap)


def heading_err_sq(env: "ManagerBasedRlEnv", cap: float = HEADING_CAP) -> torch.Tensor:
    """(yaw - (-pi/2))^2, saturating at cap^2. A cost (>= 0): negative weight."""
    return saturating_sq(tracker(env).heading_err(), cap)


def failed(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """1 on the step an episode ends by a failure (not a time-out). Paired with a
    negative weight. mjlab scales every term by dt (0.02 s), so weight -500 is
    -10 per fall."""
    return env.termination_manager.terminated.float()


# ------------------------------------------------------------ terminations ---

def body_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Any of her non-foot parts touches the terrain (story.Judge: only feet)."""
    found = env.scene[sensor_name].data.found
    return torch.any(found > 0, dim=1)


def foot_off(env: "ManagerBasedRlEnv", margin: float = 0.015,
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
             ) -> torch.Tensor:
    """A foot sank `margin` below the path surface under it: it is beside the
    beam or down a gap (story.Judge: feet only on his back and the ledges).
    Over a gap the surface is the lower of the two edges (a sole bridging the
    gap rests there)."""
    tr = tracker(env)
    a = env.scene[asset_cfg.name].data
    o = env.scene.env_origins
    p = a.site_pos_w[:, asset_cfg.site_ids]                    # [B, 2, 3]
    top = tr.lookup(tr.top, p[..., 0] - o[:, None, 0])         # [B, 2]
    return torch.any(p[..., 2] < o[:, None, 2] + top + SITE_Z - margin, dim=1)


def dropped(env: "ManagerBasedRlEnv", stand_z: float = 0.120, margin: float = 0.06) -> torch.Tensor:
    """Her trunk is `margin` below standing height over the highest surface
    under her stance (+-65 mm)."""
    tr = tracker(env)
    x, _, z = tr.rel()
    return z < tr.lookup(tr.stance, x) + stand_z - margin


def reached_end(env: "ManagerBasedRlEnv", past: float = 0.10) -> torch.Tensor:
    """Her trunk is `past` onto the end platform. Used as a time-out."""
    x, _, _ = tracker(env).rel()
    return x > T.X_END + past


# ------------------------------------------------------------- observation ---

def beam_privileged(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Critic only (18): offset / 0.05, heading error / 0.3, width / 0.1,
    gate EMA / 0.1, gate, difficulty, and the surface at 12 points from
    -60 mm to +270 mm along the beam relative to under her trunk, / 0.03
    (a gap reads -90 mm)."""
    tr = tracker(env)
    x, y, _ = tr.rel()
    lvl, _ = tr.tile()
    ahead = torch.tensor(PROFILE_AHEAD, device=env.device)
    xs = x[:, None] + ahead[None, :]
    h = tr.lookup(tr.top, xs) - tr.lookup(tr.top, x)[:, None]
    h = torch.where(tr.lookup(tr.gap, xs), torch.full_like(h, GAP_VALUE), h)
    return torch.cat([torch.stack([y / 0.05, tr.heading_err() / 0.3, tr.lookup(tr.width, x) / 0.1,
                                   tr.ema_v / 0.1, tr.gate(), tr.difficulty[lvl]], dim=-1),
                      h / 0.03], dim=-1)


# ------------------------------------------------------------------ events ---

def reset_on_beam(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None, p_start: float,
                  start_x: tuple, y: tuple, yaw: tuple, clearance: tuple, wait_s: tuple = (0.0, 0.0),
                  asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Spawn her standing on her beam, facing -y.

    With probability p_start on the start platform (start_x); otherwise at a
    random spawn point along the beam (reverse curriculum: every flat run,
    including the ones right before a feature). Height = the highest surface
    under her stance + a standing clearance. Starts the progress ratchet.
    Start-platform spawns first stand for U(wait_s) seconds with a zero
    command (the story's wait); beam spawns walk at once."""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env_ids = env_ids.long()
    tr = tracker(env)
    n, dev = len(env_ids), env.device
    u = lambda lo_hi: torch.empty(n, device=dev).uniform_(*lo_hi)
    lvl, typ = tr.tile()
    k = torch.randint(0, tr.spawn.shape[-1], (n,), device=dev)
    on_start = torch.rand(n, device=dev) < p_start
    x = torch.where(on_start, u(start_x), tr.spawn[lvl[env_ids], typ[env_ids], k])
    idx = torch.clamp(torch.round(x / DX).long(), 0, tr.nx - 1)
    z = tr.stance[lvl[env_ids], typ[env_ids], idx] + u(clearance)
    asset = env.scene[asset_cfg.name]
    root = asset.data.default_root_state[env_ids].clone()
    pos = torch.stack([x, u(y), z], dim=-1) + env.scene.env_origins[env_ids]
    zero = torch.zeros(n, device=dev)
    rot = quat_mul(root[:, 3:7], quat_from_euler_xyz(zero, zero, u(yaw)))
    asset.write_root_link_pose_to_sim(torch.cat([pos, rot], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
    tr.start(env_ids, x, torch.where(on_start, u(wait_s), torch.zeros(n, device=dev)))


# -------------------------------------------------------------- curriculum ---

def beam_move_masks(progress: torch.Tensor, judged: torch.Tensor, promote_m: float, demote_m: float,
                    long_enough: torch.Tensor | None = None, failed: torch.Tensor | None = None):
    """Promote: >= promote_m of new ground AND the episode did not fail (a time-out
    or the end platform). Demote: < demote_m (fell early or stuck).
    Only judged envs (walking command, episode actually ran) move. A demotion also
    needs `long_enough` (the episode failed, or ran long enough to have walked
    demote_m): the runner's random first episode lengths must not demote. Pure tensor math.
    (s3_review: 22 % of promotions came after a failed episode.)"""
    up = judged & (progress >= promote_m)
    if failed is not None:
        up = up & ~failed
    down = judged & (progress < demote_m) & ~up
    if long_enough is not None:
        down = down & long_enough
    return up, down


def terrain_levels_beam(env: "ManagerBasedRlEnv", env_ids: torch.Tensor,
                        promote_m: float = 0.8, demote_m: float = 0.2, demote_min_s: float = 10.0) -> dict:
    """Terrain-level curriculum on the ratchet: new ground covered this episode.
    Called at reset, before the spawn event overwrites the tracker."""
    tr = tracker(env)
    terrain = env.scene.terrain
    progress = (tr.best_x - tr.spawn_x)[env_ids]
    judged = (~tr.standing()[env_ids]) & (env.episode_length_buf[env_ids] > 0)
    failed = env.termination_manager.terminated[env_ids]
    long_enough = failed | (tr.steps[env_ids] * env.step_dt >= demote_min_s)
    up, down = beam_move_masks(progress, judged, promote_m, demote_m, long_enough, failed)
    terrain.update_env_origins(env_ids, up, down)
    levels = terrain.terrain_levels.float()
    out = {"mean": levels.mean(), "max": levels.max()}
    for i, name in enumerate(tr.kinds):
        m = terrain.terrain_types == i
        if m.any():
            out[name] = levels[m].mean()
    if judged.any():
        out["progress_m"] = progress[judged].mean()
        out["promoted"] = up.float().sum() / judged.float().sum()
        out["demoted"] = down.float().sum() / judged.float().sum()
    return out
