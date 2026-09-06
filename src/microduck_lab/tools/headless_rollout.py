#!/usr/bin/env python3
"""Run an ONNX policy in CPU MuJoCo with no viewer, and report what happened.

`infer_policy.py` is the interactive path: it opens a `mujoco.viewer` window and
reads the keyboard. On a headless box (no DISPLAY, no GL) that cannot start, so
this drives the same `PolicyInference` object on a fixed velocity command and
prints a rollout summary instead. Same 50 Hz / decimation-4 cadence, same
observation contract -- only the window and the keyboard are gone.

    uv run python -m microduck_lab.tools.headless_rollout --walking ../microduck/policies/alpha_walking.onnx \
        --new-cmd-obs --lin-vel-x 0.15 --seconds 10

`--video out.mp4` renders offscreen, which needs a GL stack MuJoCo can open
(`MUJOCO_GL=egl` with a GPU, `osmesa` where that is installed). It is off by
default because a CPU-only box has neither.
"""

import argparse
import os

import numpy as np


import mujoco  # noqa: E402
from microduck_lab.sim.upstream import (  # noqa: E402
    PolicyInference, MICRODUCK_XML, MICRODUCK_BALL_XML, MICRODUCK_ROLLERS_XML,
)

# infer_policy.py:882 overrides the XML's 0.002 s so that decimation 4 lands on
# the 50 Hz the policies were trained at. Same override here or the policy runs
# at 125 Hz against a model that never saw it.
CONTROL_TIMESTEP = 0.005

# The trunk stands at z = 0.12 m (the STAND keyframe); a fall puts it well under.
FALL_HEIGHT = 0.07


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walking", type=str, default=None, help="Path to the walking policy ONNX")
    p.add_argument("--standing", type=str, default=None, help="Path to the standing policy ONNX")
    p.add_argument("--sitstand", type=str, default=None, help="Path to the sitstand policy ONNX")
    p.add_argument("--ground-pick", type=str, default=None, help="Path to the ground-pick policy ONNX")
    p.add_argument("--roulade", type=str, default=None, help="Path to the roulade policy ONNX")
    p.add_argument("--kick-left", type=str, default=None, help="Path to the left-foot kick ONNX")
    p.add_argument("--kick-right", type=str, default=None, help="Path to the right-foot kick ONNX")
    p.add_argument("--roller", action="store_true", help="Use the roller-skate scene and wheel friction")
    p.add_argument("--do", action="append", default=[], metavar="NAME@T",
                   help="Fire a behaviour at sim time T seconds. NAME is one of "
                        "roulade, kick_left, kick_right, ground_pick, sit. Repeatable: "
                        "--do sit@2 --do sit@6 sits then stands again.")
    p.add_argument("--xml", type=str, default=None, help="Scene XML (default: the walk scene)")
    p.add_argument("--seconds", type=float, default=10.0, help="Rollout length in sim seconds")
    p.add_argument("--lin-vel-x", type=float, default=0.0)
    p.add_argument("--lin-vel-y", type=float, default=0.0)
    p.add_argument("--ang-vel-z", type=float, default=0.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--new-cmd-obs", action="store_true",
                   help="61-dim observation with the unified 13D command block. The policies "
                        "vendored in the microduck repo are all 61-dim, so they need this.")
    p.add_argument("--foot-friction", type=float, default=None,
                   help="Override sliding friction on the foot collision geoms. Needed for "
                        "the ice scene: MuJoCo takes the MAX of the two contacting geoms, "
                        "so an icy floor under grippy feet is not icy at all.")
    p.add_argument("--current-limit", type=float, default=1.75,
                   help="XL330 firmware current limit in A; clamps actuator force. 0 disables.")
    p.add_argument("--video", type=str, default=None, help="Write an mp4 here (needs offscreen GL)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if not args.walking and not args.standing and not args.sitstand:
        p.error("give at least one policy: --walking, --standing and/or --sitstand")

    # "roulade@2.5" -> (2.5, "roulade"). Sorted so the loop can pop them in order.
    script = []
    for item in args.do:
        name, _, when = item.partition("@")
        script.append((float(when or 0.0), name))
    script.sort()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.xml:
        xml = args.xml
    elif args.roller:
        xml = os.path.join(repo, MICRODUCK_ROLLERS_XML)
    elif args.kick_left or args.kick_right:
        # The kick policies need something to kick; that scene carries the ball.
        xml = os.path.join(repo, MICRODUCK_BALL_XML)
    else:
        xml = os.path.join(repo, MICRODUCK_XML)
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = CONTROL_TIMESTEP
    data = mujoco.MjData(model)

    # The XL330 saturates current at ~1.75 A, so torque saturates at kt * I_max.
    # Clipping the position actuators' output force reproduces the saturation the
    # policy trained against. infer_policy.py does the same thing.
    if args.current_limit and args.current_limit > 0:
        from bam.model import load_model
        kt = load_model(motor_name="xl330", model="m6").kt.value
        torque_limit = kt * args.current_limit
        model.actuator_forcerange[:, 0] = -torque_limit
        model.actuator_forcerange[:, 1] = torque_limit
        model.actuator_forcelimited[:] = 1

    policy = PolicyInference(
        model, data,
        walking_onnx_path=args.walking,
        standing_onnx_path=args.standing,
        sitstand_onnx_path=args.sitstand,
        ground_pick_onnx_path=args.ground_pick,
        roulade_onnx_path=args.roulade,
        kick_left_onnx_path=args.kick_left,
        kick_right_onnx_path=args.kick_right,
        action_scale=args.action_scale,
        new_cmd_obs=args.new_cmd_obs,
        # The class defaults this to False, but main() passes True (it is the
        # inverse of --raw-accelerometer). Left at the class default the policy
        # gets the raw accelerometer where it expects projected gravity, and
        # falls over within a couple of seconds.
        use_projected_gravity=True,
    )
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)
    policy.vel_max_x, policy.vel_min_x = 0.3, -0.3
    policy.vel_max_y, policy.vel_min_y = 0.2, -0.2
    policy.vel_max_ang = 1.5

    if args.foot_friction is not None:
        import re as _re
        n = 0
        for g in range(model.ngeom):
            gname = mujoco.mj_name2id and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gname and _re.match(r"^(left|right)_foot_collision$", gname):
                model.geom_friction[g, 0] = args.foot_friction
                n += 1
        print(f"foot friction: {args.foot_friction} on {n} geoms")

    # Wheel bearing friction, set here rather than in the XML because a non-zero
    # frictionloss in the model breaks training (infer_policy.py does the same).
    if args.roller:
        import re as _re
        for j in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jname and _re.match(r"^passive_.*", jname):
                model.dof_frictionloss[model.jnt_dofadr[j]] = 0.003

    # Start standing in the pose the policy treats as its zero, exactly as
    # infer_policy.py does: trunk at 125 mm, upright, joints at DEFAULT_POSE.
    # The scene's STAND keyframe is a different (older) pose -- starting there
    # hands the policy a body it never saw and it falls over.
    freejoint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[freejoint]
    data.qpos[adr + 0] = 0.0
    data.qpos[adr + 1] = 0.0
    data.qpos[adr + 2] = 0.125
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    obs = policy.get_observations()
    print(f"scene:       {xml}")
    print(f"observation: {obs.size}-dim, {policy.n_joints} joints, {model.nu} actuators")
    print(f"command:     vx={args.lin_vel_x} vy={args.lin_vel_y} wz={args.ang_vel_z}")
    print(f"current lim: {args.current_limit} A")

    decimation = 4
    control_dt = decimation * model.opt.timestep
    steps = int(round(args.seconds / control_dt))

    renderer = None
    frames = []
    cam = None
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        every = max(1, int(round(1.0 / (args.fps * control_dt))))
        # The scene defines no camera, so the default free camera sits still and
        # a walking duck leaves the frame in a couple of seconds. Track the trunk.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        cam.distance = 0.9
        cam.azimuth = 130.0
        cam.elevation = -15.0

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    trunk_qvel = int(model.jnt_dofadr[freejoint])
    # Body-frame forward/lateral speed and yaw rate, averaged over the rollout.
    # A pure spin command moves the trunk nowhere, so world displacement alone
    # would report it as "did nothing" -- the yaw total is what shows the turn.
    v_body_sum = np.zeros(2)
    yaw_sum = 0.0
    start_xy = data.xpos[trunk][:2].copy()
    min_height = float("inf")
    fell_at = None

    for i in range(steps):
        t = i * control_dt
        while script and script[0][0] <= t:
            _, name = script.pop(0)
            if name == "sit":
                policy.toggle_sit()
            elif name == "ground_pick":
                policy.trigger_ground_pick()
            else:
                policy.trigger_behavior(name)

        # Episodic policies advance and hand control back on a timer. Without
        # these two the duck would stay in the trick policy forever.
        policy.update_ground_pick_phase(control_dt)
        policy.update_behavior(control_dt)

        action = policy.infer()
        policy.apply_action(action)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        quat = data.qpos[adr + 3:adr + 7].astype(np.float32)
        v_world = np.array(data.qvel[trunk_qvel:trunk_qvel + 3], dtype=np.float32)
        v_body_sum += policy.quat_rotate_inverse(quat, v_world)[:2]
        yaw_sum += float(data.qvel[trunk_qvel + 5])

        z = float(data.xpos[trunk][2])
        min_height = min(min_height, z)
        if fell_at is None and z < FALL_HEIGHT:
            fell_at = i * control_dt

        if renderer is not None and i % every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    end_xy = data.xpos[trunk][:2].copy()
    travelled = end_xy - start_xy
    elapsed = steps * control_dt

    print()
    print(f"ran {steps} control steps ({elapsed:.1f} s at {1/control_dt:.0f} Hz)")
    print(f"  travelled:    dx={travelled[0]:+.3f} m  dy={travelled[1]:+.3f} m")
    print(f"  body speed:   fwd={v_body_sum[0]/steps:+.3f} lat={v_body_sum[1]/steps:+.3f} m/s "
          f"(commanded fwd={args.lin_vel_x:+.3f} lat={args.lin_vel_y:+.3f})")
    print(f"  yaw rate:     {yaw_sum/steps:+.3f} rad/s (commanded {args.ang_vel_z:+.3f}), "
          f"{yaw_sum*control_dt:+.2f} rad total")
    print(f"  trunk height: {data.xpos[trunk][2]:.3f} m at the end, {min_height:.3f} m at the lowest")
    print(f"  fell:         {'yes, at %.1f s' % fell_at if fell_at is not None else 'no'}")

    if renderer is not None and frames:
        import imageio.v3 as iio
        iio.imwrite(args.video, np.stack(frames), fps=args.fps)
        print(f"  video:        {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
