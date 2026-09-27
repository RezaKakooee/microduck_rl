"""Measure (and film) a walking policy stepping sideways.

Runs an exported 61D ONNX through a fixed command script on the deployment
rehearsal path of `scripts/infer_policy.py`: upstream `scene.xml`, BAM XL330
M6 motors set up by its own `load_bam_model`, 5 ms physics, decimation 4.
The ONNX runs alone (no switch to a standing policy): a policy trained with
standing envs has to stand by itself. Reports per segment how far the duck
moved sideways, how fast, and how much it turned. A policy that "strafes" by
turning shows up as yaw drift.

    python -m microduck_lab.tasks.walking.sideways --onnx sideways_v1.onnx
    MUJOCO_GL=egl python -m microduck_lab.tasks.walking.sideways --onnx sideways_v1.onnx \\
        --video videos/sideways/sideways_v1_try1.mp4
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab import paths
from microduck_lab.sim import upstream  # noqa: F401  (puts scripts/ on the path)
import infer_policy as ip  # noqa: E402

# (name, seconds, vx, vy, wz)
SCRIPT = (
    ("stand", 2.0, 0.0, 0.0, 0.0),
    ("left", 6.0, 0.0, 0.2, 0.0),
    ("stop", 2.0, 0.0, 0.0, 0.0),
    ("right", 6.0, 0.0, -0.2, 0.0),
    ("stop", 2.0, 0.0, 0.0, 0.0),
    ("forward", 4.0, 0.3, 0.0, 0.0),
    ("stop", 2.0, 0.0, 0.0, 0.0),
)


def build():
    """scene.xml plus painted 10 cm lines, then BAM exactly as infer_policy."""
    spec = mujoco.MjSpec.from_file(str(paths.REPO / ip.MICRODUCK_XML))
    for i in range(-10, 11):
        for name, size, pos in ((f"line_x_{i}", (1.0, 0.001, 0.0003), (0, i * 0.1, 0.0003)),
                                (f"line_y_{i}", (0.001, 1.0, 0.0003), (i * 0.1, 0, 0.0003))):
            g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                                        rgba=(0.9, 0.9, 0.9, 1))
            g.contype = g.conaffinity = 0
    bam = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
    kt, R = bam.kt.value, bam.R.value
    limit = bam.actuator.vin * kt / R
    names = []
    for act in spec.actuators:
        tgt = act.target if isinstance(act.target, str) else act.target.name
        if tgt.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited, act.forcerange, act.ctrllimited = True, (-limit, limit), False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for j in spec.joints:
            if j.name == tgt:
                j.damping = np.zeros((3, 1))
                j.frictionloss = 0.0
                j.solref_friction = ip.BAM_STIFF_SOLREF_FRICTION
                j.solimp_friction = ip.BAM_STIFF_SOLIMP_FRICTION
    model = spec.compile()
    model.opt.timestep = 0.005
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    data = mujoco.MjData(model)
    from bam.mujoco import MujocoController
    ctrl = MujocoController(bam, names, model, data, vin_drop_gain=0.1, vin_min=ip.BAM_VIN_MIN)
    return model, data, ctrl


SLOW = (
    ("stand", 2.0, 0.0, 0.0, 0.0),
    ("left", 6.0, 0.0, 0.1, 0.0),
    ("stop", 2.0, 0.0, 0.0, 0.0),
    ("right", 6.0, 0.0, -0.1, 0.0),
    ("stop", 2.0, 0.0, 0.0, 0.0),
)


def run(onnx, video=None, script=SCRIPT):
    m, d, ctrl = build()
    with contextlib.redirect_stdout(io.StringIO()):
        pol = ip.PolicyInference(m, d, walking_onnx_path=str(onnx), bam_ctrl=ctrl,
                                 new_cmd_obs=True, use_projected_gravity=True)
    pol.vel_max_x, pol.vel_min_x, pol.vel_max_y, pol.vel_min_y, pol.vel_max_ang = 0.4, -0.4, 0.3, -0.3, 1.5
    root = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    d.qpos[root:root + 7] = (0, 0, 0.125, 1, 0, 0, 0)
    d.qpos[pol.joint_qpos_indices] = pol.default_pose
    ctrl.reset(d.qpos)
    pol.set_position_targets(pol.default_pose)
    mujoco.mj_forward(m, d)
    pos = lambda: d.qpos[root:root + 3].copy()
    def yaw():
        qw, qx, qy, qz = d.qpos[root + 3:root + 7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    writer = renderer = None
    if video:
        import imageio.v2 as imageio
        if Path(video).exists():
            raise SystemExit(f"{video} exists; videos are never overwritten")
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(m, 720, 1280)
        writer = imageio.get_writer(video, fps=25, codec="libx264", quality=None, macro_block_size=8,
                                    output_params=["-crf", "22", "-movflags", "+faststart"])

    # Feet: is she stepping or sliding? Per foot, its site height above where
    # it stood, whether its sole touches the floor, and its horizontal speed.
    site = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
    sole = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("left_foot_collision", "right_foot_collision")]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rest_z = [float(d.site_xpos[i][2]) for i in site]

    def touching(g):
        return any({c.geom1, c.geom2} == {g, floor} for c in d.contact[:d.ncon])

    segments, fell, tick = [], None, 0
    for name, seconds, vx, vy, wz in script:
        with contextlib.redirect_stdout(io.StringIO()):
            pol.set_vel_cmd(vx, vy, wz)
        p0, yaw0 = pos(), yaw()
        lift, air, slip = [[], []], [0, 0], [[], []]
        peaks, cur = [[], []], [None, None]       # peak height of each swing (lift-off to touch-down)
        prev = [d.site_xpos[i][:2].copy() for i in site]
        for _ in range(int(round(seconds / 0.02))):
            pol.apply_action(pol.infer())
            for _ in range(4):
                ctrl.update()
                mujoco.mj_step(m, d)
            for k in (0, 1):
                xy = d.site_xpos[site[k]][:2].copy()
                lift[k].append(float(d.site_xpos[site[k]][2]) - rest_z[k])
                h = float(d.site_xpos[site[k]][2]) - rest_z[k]
                if touching(sole[k]):
                    slip[k].append(float(np.linalg.norm(xy - prev[k])) / 0.02)
                    if cur[k] is not None:
                        peaks[k].append(cur[k])
                        cur[k] = None
                else:
                    air[k] += 1
                    cur[k] = h if cur[k] is None else max(cur[k], h)
                prev[k] = xy
            if fell is None and d.xmat[trunk][8] < 0.5:
                fell = f"fell during '{name}'"
            if writer is not None and tick % 2 == 0:
                c = mujoco.MjvCamera()
                c.lookat[:] = (*pos()[:2], 0.08)
                c.distance, c.azimuth, c.elevation = 0.9, 180.0, -25.0   # from the front
                renderer.update_scene(d, c)
                writer.append_data(renderer.render())
            tick += 1
        dd = pos() - p0
        c, s_ = np.cos(-yaw0), np.sin(-yaw0)
        fwd, side = c * dd[0] - s_ * dd[1], s_ * dd[0] + c * dd[1]
        turn = float(np.degrees(np.arctan2(np.sin(yaw() - yaw0), np.cos(yaw() - yaw0))))
        ticks = int(round(seconds / 0.02))
        segments.append(dict(segment=name, command=[vx, vy, wz], seconds=seconds,
                             forward_mm=round(1000 * fwd), sideways_mm=round(1000 * side),
                             sideways_speed_mm_s=round(1000 * side / seconds, 1),
                             turned_deg=round(turn, 1),
                             # 95th percentile foot height above its resting height
                             foot_lift_mm=[round(1000 * float(np.percentile(l, 95)), 1) for l in lift],
                             foot_air_pct=[round(100 * a / ticks) for a in air],
                             # mean peak height per swing, like training's Metrics/peak_height_mean
                             swing_peak_mm=[round(1000 * float(np.mean(p)), 1) if p else 0.0 for p in peaks],
                             swings=[len(p) for p in peaks],
                             foot_slip_mm_s=[round(1000 * float(np.mean(v)), 1) if v else 0.0 for v in slip]))
    if writer is not None:
        writer.close()
    return dict(onnx=str(onnx), fell=fell, segments=segments)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--video")
    ap.add_argument("--report")
    ap.add_argument("--slow", action="store_true", help="0.1 m/s sideways instead of 0.2")
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    r = run(a.onnx, a.video, SLOW if a.slow else SCRIPT)
    text = json.dumps(r, indent=1)
    if a.report:
        Path(a.report).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
