#!/usr/bin/env python3
"""Run an ONNX policy on the skate-blade scene in CPU MuJoCo, no viewer.

Same loop and summary as `headless_rollout.py`, plus the one thing that scene
cannot do on its own: the blade pairs' friction frame is WORLD-fixed (slot 0 =
world Y, slot 1 = world X, whatever the foot's yaw), so training rewrites each
foot's pair friction every control step from its yaw and its contact load
(`mdp_blades.project_blade_friction`, rule "load": the true blade ellipse's
radius along the push direction; "bbox": the ellipse's bounding box). This
script does the same, so a policy sees the physics it trained on. `--static`
keeps the XML's pair values instead (a blade only while the duck faces +X).

    uv run python -m microduck_lab.tasks.skating.blades_rollout --walking blades_v1.onnx \
        --mu-along 0.08 --mu-across 1.0 --lin-vel-x 0.3 --seconds 15
"""

import argparse
import os
import sys

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION, FALL_HEIGHT, REPO, Recorder, load_scene, make_policy  # noqa: E402
from microduck_lab.rl.mdp_blades import (  # noqa: E402
    LOAD_EPS_N,
    PAIR_AXIS_WORLD_X,
    PAIR_AXIS_WORLD_Y,
    sole_long_axis_in_body,
)

BLADES_XML = os.path.join(REPO, "src/microduck_lab/models/scene_blades.xml")
PAIRS = (("blade_left", "left_foot_collision"), ("blade_right", "right_foot_collision"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walking", required=True, help="Path to the policy ONNX (61 obs / 14 act)")
    p.add_argument("--xml", default=BLADES_XML)
    p.add_argument("--mu-along", type=float, default=0.08, help="Sliding friction along the sole")
    p.add_argument("--mu-across", type=float, default=1.0, help="Sliding friction across the sole")
    p.add_argument("--rule", choices=("load", "bbox"), default="load",
                   help="How the yawed blade is squeezed into the world-aligned pair (see mdp_blades.py)")
    p.add_argument("--static", action="store_true",
                   help="Do not re-project per foot yaw; keep the XML pair values (world-aligned)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--lin-vel-x", type=float, default=0.0)
    p.add_argument("--lin-vel-y", type=float, default=0.0)
    p.add_argument("--ang-vel-z", type=float, default=0.0)
    p.add_argument("--current-limit", type=float, default=1.75)
    p.add_argument("--video", default=None, help="Write an mp4 here (needs MUJOCO_GL=egl and a GPU)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    model, data = load_scene(args.xml, current_limit=args.current_limit)
    policy, adr = make_policy(model, data, walking_onnx_path=args.walking)
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # Resolve the pairs and each foot's blade axis in its body frame.
    feet = []
    for pair_name, geom_name in PAIRS:
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, pair_name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if pid < 0 or gid < 0:
            sys.exit(f"scene has no pair '{pair_name}' / geom '{geom_name}'")
        feet.append((pid, gid, int(model.geom_bodyid[gid]), sole_long_axis_in_body(model, gid)))
        if args.static:
            continue
        model.pair_friction[pid, PAIR_AXIS_WORLD_X] = args.mu_along
        model.pair_friction[pid, PAIR_AXIS_WORLD_Y] = args.mu_across

    force6 = np.zeros(6)

    def foot_loads():
        """Net world-frame contact force per foot geom (what the mjlab sensor reports)."""
        loads = np.zeros((len(feet), 3))
        for i in range(data.ncon):
            c = data.contact[i]
            for k, (_, gid, _, _) in enumerate(feet):
                if c.geom1 == gid or c.geom2 == gid:
                    mujoco.mj_contactForce(model, data, i, force6)
                    # contact frame rows are (normal, t1, t2) in world coords
                    loads[k] += np.array(c.frame).reshape(3, 3).T @ force6[:3]
        return loads

    def radius(du2):
        """True blade-ellipse radius along a unit direction d, (d.u)^2 = du2."""
        return 1.0 / np.sqrt(du2 / args.mu_along ** 2 + (1.0 - du2) / args.mu_across ** 2)

    def project():
        """Same rules as mdp_blades._write_projection, on the CPU model."""
        a2, c2 = args.mu_along ** 2, args.mu_across ** 2
        loads = foot_loads() if args.rule == "load" else None
        yaws, mux, loaded = [], [], []
        for k, (pid, _, bid, axis_b) in enumerate(feet):
            axis_w = data.xmat[bid].reshape(3, 3) @ axis_b
            h = np.hypot(axis_w[0], axis_w[1])
            u = np.array([1.0, 0.0]) if h < 1e-3 else axis_w[:2] / h
            cos2 = u[0] ** 2
            if args.rule == "bbox":
                mu_x = np.sqrt(a2 * cos2 + c2 * (1 - cos2))
                mu_y = np.sqrt(a2 * (1 - cos2) + c2 * cos2)
                is_loaded = False
            else:
                mu_x, mu_y = radius(cos2), radius(1.0 - cos2)
                f = loads[k][:2]
                mag = np.hypot(f[0], f[1])
                is_loaded = mag > LOAD_EPS_N
                if is_loaded:
                    du2 = float(np.clip((f @ u / mag) ** 2, 0.0, 1.0))
                    mu_x = mu_y = radius(du2)
            model.pair_friction[pid, PAIR_AXIS_WORLD_X] = mu_x
            model.pair_friction[pid, PAIR_AXIS_WORLD_Y] = mu_y
            yaws.append(np.degrees(np.arctan2(abs(u[1]), abs(u[0]))))
            mux.append(mu_x)
            loaded.append(is_loaded)
        return yaws, mux, loaded

    steps = int(round(args.seconds / CONTROL_DT))
    rec = Recorder(model, fps=args.fps) if args.video else None
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    v_body_sum = np.zeros(2)
    yaw_sum = 0.0
    start_xy = data.xpos[trunk][:2].copy()
    min_height = float("inf")
    fell_at = None
    foot_yaw_sum = np.zeros(2)
    mu_x_sum = np.zeros(2)
    loaded_mu_sum = np.zeros(2)
    loaded_steps = np.zeros(2)

    print(f"scene:       {args.xml}")
    print(f"blade:       mu_along={args.mu_along} mu_across={args.mu_across} "
          f"({'static world-aligned pair' if args.static else 'rule=%s, rewritten each step' % args.rule})")
    print(f"command:     vx={args.lin_vel_x} vy={args.lin_vel_y} wz={args.ang_vel_z}")

    for i in range(steps):
        if not args.static:
            yaws, mux, loaded = project()
            foot_yaw_sum += yaws
            mu_x_sum += mux
            for k in range(2):
                if loaded[k]:
                    loaded_mu_sum[k] += mux[k]
                    loaded_steps[k] += 1
        policy.update_ground_pick_phase(CONTROL_DT)
        policy.update_behavior(CONTROL_DT)
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        quat = data.qpos[adr + 3:adr + 7].astype(np.float32)
        v_world = np.array(data.qvel[adr:adr + 3], dtype=np.float32)
        v_body_sum += policy.quat_rotate_inverse(quat, v_world)[:2]
        yaw_sum += float(data.qvel[adr + 5])
        z = float(data.xpos[trunk][2])
        min_height = min(min_height, z)
        if fell_at is None and z < FALL_HEIGHT:
            fell_at = i * CONTROL_DT
        if rec is not None:
            rec.maybe_capture(i, data)

    travelled = data.xpos[trunk][:2] - start_xy
    print()
    print(f"ran {steps} control steps ({steps * CONTROL_DT:.1f} s at {1 / CONTROL_DT:.0f} Hz)")
    print(f"  travelled:    dx={travelled[0]:+.3f} m  dy={travelled[1]:+.3f} m")
    print(f"  body speed:   fwd={v_body_sum[0] / steps:+.3f} lat={v_body_sum[1] / steps:+.3f} m/s "
          f"(commanded fwd={args.lin_vel_x:+.3f} lat={args.lin_vel_y:+.3f})")
    print(f"  yaw rate:     {yaw_sum / steps:+.3f} rad/s, {yaw_sum * CONTROL_DT:+.2f} rad total")
    print(f"  trunk height: {data.xpos[trunk][2]:.3f} m at the end, {min_height:.3f} m at the lowest")
    print(f"  fell:         {'yes, at %.1f s' % fell_at if fell_at is not None else 'no'}")
    if not args.static:
        print(f"  foot yaw:     mean |yaw| L={foot_yaw_sum[0] / steps:.1f} deg R={foot_yaw_sum[1] / steps:.1f} deg "
              f"(0 = blade along world X)")
        print(f"  eff. mu_x:    mean L={mu_x_sum[0] / steps:.3f} R={mu_x_sum[1] / steps:.3f} "
              f"(along-blade {args.mu_along} when the foot points along X)")
        if args.rule == "load":
            lm = [loaded_mu_sum[k] / max(loaded_steps[k], 1) for k in range(2)]
            print(f"  loaded feet:  L {100 * loaded_steps[0] / steps:.0f}% of steps at mean mu {lm[0]:.3f}, "
                  f"R {100 * loaded_steps[1] / steps:.0f}% at {lm[1]:.3f} "
                  f"(pushing across the blade -> {args.mu_across}, along -> {args.mu_along})")
    if rec is not None:
        n = rec.write(args.video)
        print(f"  video:        {args.video} ({n} frames)")


if __name__ == "__main__":
    main()
