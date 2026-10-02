"""Evaluate a trampoline-flip checkpoint: success rates, and a clip of one episode.

    .venv/bin/python -m microduck_lab.rl.trampoline_eval --checkpoint logs/rsl_rl/microduck_trampoline_flip/<run>/model_2000.pt
    MUJOCO_GL=egl .venv/bin/python -m microduck_lab.rl.trampoline_eval --checkpoint ... --video flip_rl_run1_2000

By default every episode starts with a drop onto the bed (no mid-flip
spawns), so the numbers measure the whole skill. Per episode:

- flip: the forward rotation inside ONE flight reached 300 deg (measured here,
  not read from the reward's counter: run 4 farmed that counter with ~60 deg
  tilts on many bounces and never flipped);
- landed: the flip was done and the robot was still up 1 s later;
- stayed up: the flip was done and the episode ran to its time limit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# First: load the task registry the way `train` does. Importing our mdp module
# before it closes an import cycle (mjlab loads the task packages on import,
# they register this env, and the env imports the half-loaded mdp module).
import mjlab_microduck.tasks  # noqa: F401,E402

TASK = "Mjlab-Trampoline-Flip-MicroDuck"   # --task changes it
START = "drop"                             # --start stand: standing still on the settled bed
DONE_DEG = 300.0   # in ONE flight; the last ~40 deg of a good flip finish on the bed


def set_bed(sag_mm=None, rebound=None, mat_kg=None):
    """Change the bed before the env is built (the bed spec reads these at build time)."""
    from microduck_lab.rl import trampoline_scene as T
    from microduck_lab.tasks.trampoline.world import fit_damping
    sag = T.SAG_MM if sag_mm is None else sag_mm
    reb = T.REBOUND if rebound is None else rebound
    T.BED_MASS = T.BED_MASS if mat_kg is None else mat_kg
    T.BED_K = T.ROBOT_MASS * 9.81 / (sag / 1000)
    T.BED_C = fit_damping(T.BED_K, T.ROBOT_MASS, reb, T.BED_MASS, T.REBOUND_DROP_MM)
    return dict(sag_mm=sag, rebound=reb, mat_kg=T.BED_MASS)


def build(num_envs: int, drop_only: bool, device: str, seed: int | None = None, drop_height=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed          # None: a different random episode every run
    # The bed under test is the one set by set_bed(), not a random one.
    for name in ("randomize_bed_stiffness", "randomize_bed_damping", "randomize_bed_mass"):
        cfg.events.pop(name, None)
    if drop_only:
        cfg.curriculum.pop("spawn_mix", None)
        cfg.events["reset_trampoline_state"].params.update(drop_prob=1.0, midflip_prob=0.0, stand_prob=0.0)
    if START == "stand":
        cfg.curriculum.pop("spawn_mix", None)
        cfg.events["reset_trampoline_state"].params.update(drop_prob=0.0, midflip_prob=0.0, stand_prob=1.0)
    if drop_height is not None:
        cfg.events["reset_trampoline_state"].params["drop_height_range"] = tuple(drop_height)
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), agent_cfg


def load_policy(wrapped, agent_cfg, checkpoint: str, device: str):
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_runner_cls
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


class Film:
    """Views of env 0.

    camera="both": wide and fixed (the bounce height) next to a close view
    that follows the robot. camera="side": one side view (the flip turns in
    its image plane) that follows the robot along the bed at a fixed height,
    so the bounce height still shows.
    """

    def __init__(self, env, width=640, height=480, camera="both"):
        import mujoco
        from microduck_lab.rl.trampoline_scene import BED_TOP
        self.env, self.mujoco, self.top, self.camera = env, mujoco, BED_TOP, camera
        self.m = env.sim.mj_model
        if camera == "side":
            width, height = 1280, 720
            self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
            self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height=height, width=width)
        self.side = mujoco.MjvCamera()
        self.side.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.side.distance, self.side.azimuth, self.side.elevation = 1.6, 90.0, -6.0
        self.wide = mujoco.MjvCamera()
        self.wide.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.wide.distance, self.wide.azimuth, self.wide.elevation = 3.0, 90.0, -4.0
        self.wide.lookat[:] = [0.0, 0.0, BED_TOP + 0.4]
        self.near = mujoco.MjvCamera()
        self.near.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.near.distance, self.near.azimuth, self.near.elevation = 0.8, 60.0, -5.0
        self.frames = []

    def __call__(self, text):
        mj = self.mujoco
        self.d.qpos[:] = self.env.sim.data.qpos[0].cpu().numpy()
        mj.mj_forward(self.m, self.d)
        trunk = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "robot/trunk_base")
        self.near.lookat[:] = self.d.xpos[trunk]
        x, y = self.d.xpos[trunk][:2]
        self.side.lookat[:] = [x, y, self.top + 0.30]
        views = []
        for cam in ((self.side,) if self.camera == "side" else (self.wide, self.near)):
            self.r.update_scene(self.d, camera=cam)
            views.append(self.r.render())
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.concatenate(views, axis=1))
        ImageDraw.Draw(im).text((10, 10), text, fill=(255, 255, 255))
        self.frames.append(np.asarray(im))

    def write(self, stem, fps=25):
        import imageio.v3 as iio
        from microduck_lab.tasks.trampoline.video import next_free
        path = next_free(stem)
        iio.imwrite(path, np.stack(self.frames), fps=fps)
        slow = path.with_name(f"{path.stem}_slow4x.mp4")
        iio.imwrite(slow, np.stack(self.frames), fps=fps / 4 * 2)   # every control step, 4x slower
        return path, slow


def evaluate(checkpoint, num_envs=256, seconds=30.0, drop_only=True, device="cuda:0", video=None, bed=None,
             camera="both", seed=None, drop_height=None, handoff=None, switch_after=3.0):
    from microduck_lab.rl import mdp_trampoline as tmdp
    bed_used = set_bed(**(bed or {}))
    wrapped, agent_cfg = build(num_envs, drop_only, device, seed, drop_height)
    env = wrapped.unwrapped
    policy = load_policy(wrapped, agent_cfg, checkpoint, device)
    # Hand-over: `checkpoint` drives until switch_after s into the episode, then
    # `handoff` takes over at the next moment the robot is in the air (each env
    # on its own), as the runtime swaps policies that share the observation.
    policy_b = load_policy(wrapped, agent_cfg, handoff, device) if handoff else None
    obs = wrapped.get_observations()
    n = env.num_envs
    use_b = torch.zeros(n, dtype=torch.bool, device=env.device)
    switched_at = torch.full((n,), -1.0, device=env.device)
    frontier = torch.zeros(n, device=env.device)      # max rotation inside one flight, this episode
    counter = torch.zeros(n, device=env.device)       # the reward's rotation counter, for comparison
    ep_bounces = torch.zeros(n, device=env.device)
    ep_apex = torch.zeros(n, device=env.device)       # highest root point above standing on the bed
    flight_rot = torch.zeros(n, device=env.device)
    done_at = torch.full((n,), -1.0, device=env.device)
    t_ep = torch.zeros(n, device=env.device)
    bounces_at_flip = torch.full((n,), -1.0, device=env.device)
    spawn_h = env.scene["robot"].data.root_link_pos_w[:, 2] - (tmdp.BED_TOP + tmdp.STAND_ROOT_Z)
    episodes = []
    term_names = list(env.termination_manager.active_terms)
    film = Film(env, camera=camera) if video else None
    steps = int(seconds / env.step_dt)
    with torch.inference_mode():
        for k in range(steps):
            actions = policy(obs)
            if policy_b is not None:
                actions = torch.where(use_b.unsqueeze(-1), policy_b(obs), actions)
            obs, _, dones, _ = wrapped.step(actions)
            s = tmdp._state(env)
            t_ep += env.step_dt
            ended = dones.bool()
            # s[] already holds the NEW episode for ended envs; use the value kept from the last step.
            live = ~ended
            air = ~(tmdp._any(env, tmdp.FEET_SENSOR) | tmdp._any(env, tmdp.BODY_SENSOR) | tmdp._any(env, tmdp.FLOOR_SENSOR))
            omega = env.scene["robot"].data.root_link_ang_vel_b[:, 1]
            flight_rot = torch.where(air, flight_rot + omega * env.step_dt, torch.zeros_like(flight_rot))
            frontier[live] = torch.maximum(frontier[live], flight_rot[live])
            if policy_b is not None:
                go = live & ~use_b & air & (t_ep >= switch_after)
                switched_at[go] = t_ep[go]
                use_b |= go
            counter[live] = torch.maximum(counter[live], s["frontier"][live])
            ep_bounces[live] = s["bounces"][live]
            ep_apex[live] = torch.maximum(ep_apex[live], s["apex"][live])
            newly = live & (done_at < 0) & (frontier >= math.radians(DONE_DEG))
            done_at[newly] = t_ep[newly]
            bounces_at_flip[newly] = s["bounces"][newly]
            if film is not None and k % 2 == 0:
                tag = ("   flip policy" if use_b[0] else "   bounce policy") if policy_b is not None else ""
                film(f"t {t_ep[0].item():5.2f} s   rotation {math.degrees(frontier[0].item()):5.0f} deg{tag}")
            if ended.any():
                reasons = {name: env.termination_manager.get_term(name)[ended].cpu().numpy() for name in term_names}
                idx = ended.nonzero().flatten().cpu().numpy()
                for j, e in enumerate(idx):
                    why = [name for name in term_names if reasons[name][j]]
                    episodes.append(dict(length_s=float(t_ep[e]), frontier_deg=math.degrees(float(frontier[e])),
                                         counter_deg=math.degrees(float(counter[e])),
                                         bounces=float(ep_bounces[e]), apex_m=float(ep_apex[e]),
                                         flip_at_s=float(done_at[e]), end=why[0] if why else "time_out",
                                         bounces_before_flip=float(bounces_at_flip[e]),
                                         switched_at_s=float(switched_at[e]),
                                         spawn_h=float(spawn_h[e])))
                spawn_h[ended] = (env.scene["robot"].data.root_link_pos_w[:, 2] - (tmdp.BED_TOP + tmdp.STAND_ROOT_Z))[ended]
                use_b[ended] = False
                switched_at[ended] = -1.0
                frontier[ended] = 0.0
                counter[ended] = 0.0
                ep_bounces[ended] = 0.0
                ep_apex[ended] = 0.0
                flight_rot[ended] = 0.0
                done_at[ended] = -1.0
                bounces_at_flip[ended] = -1.0
                t_ep[ended] = 0.0
                if film is not None and ended[0]:
                    break
    ep_len = env.max_episode_length * env.step_dt
    flips = [e for e in episodes if e["frontier_deg"] >= DONE_DEG]
    landed = [e for e in flips if e["length_s"] - e["flip_at_s"] >= 1.0 or e["end"] == "time_out"]
    bb = [e["bounces_before_flip"] for e in flips]
    stayed = [e for e in flips if e["end"] == "time_out"]
    ends = {}
    for e in episodes:
        ends[e["end"]] = ends.get(e["end"], 0) + 1
    out = dict(checkpoint=checkpoint, episodes=len(episodes), episode_limit_s=ep_len,
               flip_rate=len(flips) / max(len(episodes), 1),
               landed_rate=len(landed) / max(len(episodes), 1),
               stayed_up_rate=len(stayed) / max(len(episodes), 1),
               mean_frontier_deg=float(np.mean([e["frontier_deg"] for e in episodes])) if episodes else 0.0,
               mean_counter_deg=float(np.mean([e["counter_deg"] for e in episodes])) if episodes else 0.0,
               mean_bounces=float(np.mean([e["bounces"] for e in episodes])) if episodes else 0.0,
               mean_apex_m=float(np.mean([e["apex_m"] for e in episodes])) if episodes else 0.0,
               mean_length_s=float(np.mean([e["length_s"] for e in episodes])) if episodes else 0.0,
               mean_up_after_flip_s=float(np.mean([e["length_s"] - e["flip_at_s"] for e in flips])) if flips else 0.0,
               ends=ends, bed=bed_used, task=TASK, handoff=handoff, switch_after=switch_after if handoff else None,
               switched_rate=(sum(e["switched_at_s"] >= 0 for e in episodes) / max(len(episodes), 1)) if handoff else None, start=START if not drop_only or START == "stand" else "drop",
               mean_flip_at_s=float(np.mean([e["flip_at_s"] for e in flips])) if flips else 0.0,
               bounces_before_flip=({int(b): bb.count(b) for b in sorted(set(bb))} if bb else {}))
    if drop_only and episodes:
        bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.5)]
        out["by_drop_height"] = {}
        for lo, hi in bins:
            sel = [e for e in episodes if lo <= e["spawn_h"] < hi]
            ok = [e for e in sel if e["frontier_deg"] >= DONE_DEG and e["end"] == "time_out"]
            out["by_drop_height"][f"{lo:.2f}-{hi:.2f} m"] = f"{len(ok)}/{len(sel)} flip and stay up"
    if film is not None:
        out["videos"] = [str(p) for p in film.write(video)]
    return out


def main():
    global TASK, START
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--with-midflip", action="store_true", help="keep the training spawn mix")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--handoff", default=None, help="second checkpoint: takes over in the air after --switch-after s")
    ap.add_argument("--switch-after", type=float, default=3.0)
    ap.add_argument("--start", default="drop", choices=("drop", "stand"))
    ap.add_argument("--video", default=None, help="film env 0 for one episode, under this name")
    ap.add_argument("--camera", default="both", choices=("both", "side"), help="two views, or one side view")
    ap.add_argument("--seed", type=int, default=None, help="fix the episode (for a replayable clip)")
    ap.add_argument("--drop-height", type=float, nargs=2, default=None,
                    help="start height range above standing on the bed (m); 0 0 = standing still")
    ap.add_argument("--bed-sag", type=float, default=None, help="mm (trained: 80)")
    ap.add_argument("--bed-rebound", type=float, default=None, help="rigid-weight rebound (trained: 0.95)")
    ap.add_argument("--bed-mat", type=float, default=None, help="kg of moving mat (trained: 0.01)")
    a = ap.parse_args()
    TASK, START = a.task, a.start
    if a.video:
        a.num_envs, a.seconds = 1, 12.0
    bed = dict(sag_mm=a.bed_sag, rebound=a.bed_rebound, mat_kg=a.bed_mat)
    out = evaluate(a.checkpoint, a.num_envs, a.seconds, drop_only=not a.with_midflip and a.start == "drop", video=a.video, bed=bed,
                   camera=a.camera, seed=a.seed, drop_height=a.drop_height, handoff=a.handoff,
                   switch_after=a.switch_after)
    out["seed"] = a.seed
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
