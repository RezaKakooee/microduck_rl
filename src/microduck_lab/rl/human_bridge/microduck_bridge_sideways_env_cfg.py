"""Microduck: step sideways across a brother lying over a gap.

The human-bridge scene (`tasks/human_bridge`) as an RL task for the sister.
Built on the sideways-walking task, so obs (61D), actions, DR, noise and
delays are the walking recipe's.

Why sideways, and why train on him at all:

* Walking forward, her feet sit side by side 84 mm apart; his trunk is 64 mm
  wide, and the walking policy fell off it in 6 of 6 runs.
* Sideways, her feet go one behind the other along his length, and each
  54 mm sole lies across his 64 mm back.
* The sideways policy trained on flat ground (v1) still fell every time on
  his back, 1.5-4 s after stepping on: his back is narrow, bumpy, and 16 mm
  lower at his hips than at his trunk. Nothing in flat-ground training looks
  like that. So she is fine-tuned on his body.

The set:

* The ledges come from `human_bridge.scene.LAYOUT`.
* He is FIXED geometry: the 20 solid meshes of his body, at the world poses
  they settle into after the real fall and 3 s of holding himself straight
  under BAM (`human_bridge/bake.py` -> `bridge_pose.json`).
* Everything is added to the "terrain" body, so the recipe's foot contact
  sensor (secondary body "terrain") and foot height scan see his back.
* All envs share one origin (`env_spacing=0`); every env is its own world.

The task: she faces -y, so her left points across. A constant slow sideways
command (vy 0.08-0.14 m/s, vx and wz near zero) runs the whole episode. A
quarter of episodes start at the real start on her ledge; the rest start
anywhere along the route, at the height of his back there. Rewards: the
sideways recipe, plus pay for world +x speed, and costs for leaving his
centre line or turning. The episode ends if she falls, or her trunk drops
below `DROP_Z` (off his side into the gap).
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

from microduck_lab import paths
from microduck_lab.rl.human_bridge import mdp_bridge
from microduck_lab.rl.walking.microduck_velocity_sideways_env_cfg import (
    MicroduckSidewaysRlCfg,
    make_microduck_velocity_sideways_env_cfg,
)
from microduck_lab.tasks.human_bridge.rl_policy.scene import LAYOUT as L

POSE_FILE = paths.LAB / "tasks" / "human_bridge" / "rl_policy" / "bridge_pose.json"

START_X = -0.25               # her trunk, 170 mm before her ledge edge
START_JITTER = (0.02, 0.01, 0.05)   # x, y (m), yaw (rad)
HEADING = -math.pi / 2        # facing -y: her left is +x, across the gap
SIDE_SPEED = (0.08, 0.14)     # m/s, "carefully"
EPISODE_S = 12.0              # 0.1 m/s crosses his 0.3 m back in ~6 s
DROP_Z = min(L.near_z, L.far_z) + 0.05   # trunk this low = off him / in the gap
# v1 of this task (Gaussian centre-line / heading rewards, no progress term)
# learned to STAND STILL on her ledge: every episode ran to the time-out, 0
# falls, speed error = the command. Standing kept the centre-line and heading
# pay, the recipe's loose velocity term still paid 0.88 x 2 per step against a
# 0.11 m/s command, and stepping onto him risked a fall. v2:
# v3 (max 0.15) rushed: after 1000 iterations episodes lasted 1.4 s and
# most ended off his side. Speed beyond a careful walk now pays nothing.
PROGRESS_MAX = 0.10           # m/s; pay world +x speed up to this
# v2 paid 10 (1.1 per step at 0.11 m/s) and STILL unlearned walking within
# 250 iterations of the warm start: the recipe's alive-type terms (upright,
# pose, head tracking, ...) pay just for standing, and a fall ends them all.
PROGRESS_WEIGHT = 40.0        # 4.0 per step at 0.10 m/s
DONE_X = L.far_edge + 0.08    # trunk here = all of her on the far ledge
DONE_BONUS = 100.0            # once, then the episode ends
# v5/v6 learned to MARCH IN PLACE at her ledge edge: in the training sim, all
# 64 of 64 envs started at the real start timed out without stepping onto
# him. Stepping in place still collects upright, pose, head tracking, heading,
# air time and foot lift every step; stepping onto him risks a fall that ends
# them all. So those per-step "alive" terms are turned down here, and a fall
# costs something.
# v7-v9 still marched in place at the edge (64 of 64 time-outs from her
# ledge, also on the corrected brother): marching paid ~3 per step, a step onto
# him paid ~4 but risked a fall. v10 removes every per-step pay marching can
# earn; only moving across pays. Trying must be cheap while the step is new.
ALIVE_WEIGHTS = {"upright": 0.1, "pose": 0.1, "head_pose_tracking": 0.0,
                 "track_yaw_rate_tight": 0.2, "track_linear_velocity": 0.2,
                 "air_time": 0.2, "swing_lift": 0.2}
FALL_COST = -5.0
CENTRE_COST = -1500.0         # * y^2: -1.35 at 3 cm off (v3: -300 was ignored)
HEADING_COST = -3.0           # * yaw_err^2: -0.12 at 0.2 rad
# Where she spawns: a quarter at the real start, the rest anywhere along the
# route (his legs, trunk, head, the far edge) -- reverse curriculum.
P_START = 0.20
# v5-v7 always stopped ~50 mm before her ledge edge: the step from her ledge
# onto his feet was the one move never practised (route spawns land on him
# already; start spawns walk up to the edge and stop). Half the episodes now
# start right in front of that step.
P_EDGE = 0.50
EDGE_X = (-0.15, -0.05)
ROUTE_X = (-0.25, L.far_edge + 0.02)
STANCE_CLEARANCE = (0.122, 0.130)   # trunk above the highest point under her feet, + 4 mm


def add_bridge(spec: mujoco.MjSpec) -> None:
    """Ledges and his baked body, as static geoms on the terrain body."""
    terrain = spec.body("terrain")
    for box in L.design().boxes:
        c = [(a + b) / 2 for a, b in zip(box.lo, box.hi)]
        h = [(b - a) / 2 for a, b in zip(box.lo, box.hi)]
        terrain.add_geom(name=f"bridge_{box.name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=c, size=h, rgba=box.rgba)
    pose = json.loads(Path(POSE_FILE).read_text())
    added = set()
    for i, g in enumerate(pose["geoms"]):
        mesh = f"brother_{g['mesh']}"
        if mesh not in added:
            spec.add_mesh(name=mesh, file=str(paths.REPO / g["file"]), scale=g["scale"])
            added.add(mesh)
        terrain.add_geom(name=f"brother_{i}_{g['mesh']}", type=mujoco.mjtGeom.mjGEOM_MESH,
                         meshname=mesh, pos=g["pos"], quat=g["quat"],
                         friction=g["friction"], rgba=(0.30, 0.53, 0.93, 1))


def surface_table(step=0.005):
    """Highest point under her stance at each trunk x, on the baked set.

    Collision uses each mesh's convex hull, whose top is the highest mesh
    vertex, so this reads vertices, not rays: rays pass straight through his
    hollow shell meshes and put her spawn inside his trunk. Per x column:
    the highest vertex or box top across her sideways sole (|y| < 25 mm), then
    the max over her stance (feet +-42 mm, soles +-21 mm)."""
    import numpy as np
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    add_bridge(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    P = np.vstack(pts)
    P = P[np.abs(P[:, 1]) < 0.025]
    boxes = [b for b in L.design().boxes if b.lo[1] <= 0.0 <= b.hi[1]]
    xs = np.arange(START_X - 0.10, L.far_edge + 0.30, step)
    top = np.zeros(len(xs))
    for i, x in enumerate(xs):
        col = P[np.abs(P[:, 0] - x) <= step / 2, 2]
        h = [b.hi[2] for b in boxes if b.lo[0] <= x <= b.hi[0]]
        top[i] = max([*h, *(col.tolist() or [0.0])])
    k = int(round(0.063 / step))
    stance = np.array([top[max(0, i - k):i + k + 1].max() for i in range(len(top))])
    return tuple(round(float(v), 4) for v in xs), tuple(round(float(v), 4) for v in stance)


def make_microduck_bridge_sideways_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_sideways_env_cfg(play=play)

    previous = cfg.scene.spec_fn
    if previous is None:
        cfg.scene.spec_fn = add_bridge
    else:
        def chained(spec):
            previous(spec)
            add_bridge(spec)
        cfg.scene.spec_fn = chained
    # The scene copies ITS env_spacing onto the terrain at build time, so the
    # terrain's own field is overwritten; set both.
    cfg.scene.env_spacing = 0.0
    cfg.scene.terrain.env_spacing = 0.0
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 80)
    cfg.episode_length_s = EPISODE_S

    # Spawn along the route, facing -y, at the right height for that spot.
    jx, jy, jyaw = START_JITTER
    sx, sz = surface_table()
    cfg.events["reset_base"] = EventTermCfg(
        func=mdp_bridge.reset_on_route, mode="reset",
        params={"start_x": (START_X - jx, START_X + jx), "route_x": ROUTE_X, "p_start": P_START,
                "y": (-jy, jy), "yaw": (HEADING - jyaw, HEADING + jyaw),
                "surface_x": sx, "surface_z": sz, "clearance": STANCE_CLEARANCE,
                "edge_x": EDGE_X, "p_edge": P_EDGE})
    # A push on his back is a push into the gap; this task is not about pushes.
    cfg.events.pop("push_robot", None)

    # One command for the whole episode: slow, steady, sideways.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.02, 0.02)
    twist.ranges.lin_vel_y = SIDE_SPEED
    twist.ranges.ang_vel_z = (-0.05, 0.05)
    twist.rel_sideways_envs = 1.0
    twist.sideways_speed = SIDE_SPEED
    twist.sideways_vx_noise = 0.02
    twist.sideways_wz_noise = 0.05
    twist.rel_turn_in_place_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (EPISODE_S, EPISODE_S)
    cfg.curriculum.pop("standing_envs", None)

    # The sideways bucket samples |vy| with a random sign; she must always go +y
    # in her frame (+x in the world). Fix the sign in the sampler.
    twist.sideways_sign = 1.0

    # Where she is on him, through the 6 body-pose command slots (unused here,
    # always ~0 before). See mdp_bridge.bridge_state and rl/human_bridge/warmstart.py.
    state = ObservationTermCfg(func=mdp_bridge.bridge_state, params={
        "mid_x": 0.05, "surface_x": sx, "surface_z": sz, "stand_height": 0.122})
    cfg.observations["actor"].terms["body_command"] = state
    cfg.observations["critic"].terms["body_command"] = deepcopy(state)

    cfg.rewards["progress"] = RewardTermCfg(
        func=mdp_bridge.world_x_progress, weight=PROGRESS_WEIGHT, params={"max_speed": PROGRESS_MAX})
    cfg.rewards["centre_line"] = RewardTermCfg(func=mdp_bridge.world_y_sq, weight=CENTRE_COST)
    cfg.rewards["heading"] = RewardTermCfg(
        func=mdp_bridge.heading_err_sq, weight=HEADING_COST, params={"target": HEADING})

    cfg.rewards["crossed"] = RewardTermCfg(
        func=mdp_bridge.crossed_bonus, weight=DONE_BONUS, params={"x": DONE_X})
    cfg.terminations["crossed"] = TerminationTermCfg(func=mdp_bridge.crossed, params={"x": DONE_X})

    for name, weight in ALIVE_WEIGHTS.items():
        cfg.rewards[name].weight = weight
    cfg.rewards["fell"] = RewardTermCfg(
        func=mdp_bridge.fell_or_dropped, weight=FALL_COST,
        params={"minimum_height": DROP_Z, "limit_angle": math.radians(70.0)})

    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["dropped"] = TerminationTermCfg(
        func=mjlab_mdp.root_height_below_minimum, params={"minimum_height": DROP_Z})
    return cfg


MicroduckBridgeSidewaysRlCfg = deepcopy(MicroduckSidewaysRlCfg)
MicroduckBridgeSidewaysRlCfg.experiment_name = "bridge_sideways"
MicroduckBridgeSidewaysRlCfg.run_name = "bridge_sideways"
MicroduckBridgeSidewaysRlCfg.max_iterations = 2000
# The bridge is not left-right symmetric for her: she always goes the same way
# across it. The mirror loss would ask her to also cross in the other
# direction, which never happens here.
MicroduckBridgeSidewaysRlCfg.algorithm.symmetry_cfg = None
