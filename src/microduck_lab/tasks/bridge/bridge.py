#!/usr/bin/env python3
"""Suspension bridge: walk the Microduck across a soft, floating rope net bridge.

The bridge deck is a physically loose, flexible, simple-rope net ("طناب ساده")
suspended between elevated approach and exit platforms. Rather than a rigid plank,
the deck is modeled as an articulated series of independently compliant rungs,
each with multi-axis compliance:
  - Vertical sag/deflection (slide joint j_z) that sinks 35-55 mm under the duck's
    ~800 g weight, creating a moving catenary trough underfoot.
  - Roll compliance (hinge joint j_roll) tilting the walkway laterally under off-center steps.
  - Pitch compliance (hinge joint j_pitch) flexing as the duck pushes off and lands.
  - Natural twisted hemp rope cordage, knots, and vertical suspension hangers.

The pretrained walking policy (alpha_walking.onnx) provides locomotion at 50 Hz, guided by
a closed-loop lateral tracking steering controller that counteracts bridge oscillations.

Usage:
    .venv/bin/python -m microduck_lab.tasks.bridge.bridge --center-kz 160.0 --sag 0.035
    .venv/bin/python -m microduck_lab.tasks.bridge.bridge --sweep-kz 200.0 160.0 130.0
    sbatch -M cluster src/microduck_lab/tasks/bridge/bridge.sbatch
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

# Ensure EGL platform is configured if MUJOCO_GL=egl
if os.environ.get("MUJOCO_GL") == "egl" and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION, FALL_HEIGHT

WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")
TEMPLATE = paths.model("scene_beam.xml")

# Default bridge geometry
DEFAULT_START_X = 0.00   # start of suspended span (m)
DEFAULT_LENGTH = 2.00    # length of suspended span (m)
DEFAULT_WIDTH = 0.44     # total deck width (m)
DECK_HEIGHT = 0.18       # height above floor (m)
DEFAULT_SAG = 0.035      # baseline static catenary sag (m)
DEFAULT_NUM_RUNGS = 24   # number of articulated rungs along span

# Compliance parameters for soft rope net ("طناب ساده")
DEFAULT_CENTER_KZ = 160.0    # vertical stiffness at span center (N/m) -> ~45-50 mm deflection under 8 N
DEFAULT_CENTER_KR = 16.0     # roll rotational stiffness at center (N*m/rad)
DEFAULT_CENTER_KP = 22.0     # pitch rotational stiffness at center (N*m/rad)
DEFAULT_DAMPING_RATIO = 1.4  # critical damping ratio
DEFAULT_SPEED = 0.28         # commanded walking speed (m/s)
SETTLE_S = 1.0               # stand still settling duration (s)


def bridge_xml(
    start_x=DEFAULT_START_X,
    end_x=None,
    length=None,
    half_width=None,
    width=None,
    height=DECK_HEIGHT,
    sag=DEFAULT_SAG,
    num_rungs=DEFAULT_NUM_RUNGS,
    center_kz=DEFAULT_CENTER_KZ,
    center_kr=DEFAULT_CENTER_KR,
    center_kp=DEFAULT_CENTER_KP,
    damping_ratio=DEFAULT_DAMPING_RATIO,
    landing_len=0.40,
    soft=True,
    sway=True,
    **kwargs,
):
    """Procedural XML for a soft, articulated dynamic rope suspension bridge."""
    if length is None:
        if end_x is not None:
            length = end_x - start_x
        else:
            length = DEFAULT_LENGTH

    if width is None:
        if half_width is not None:
            width = 2.0 * half_width
        else:
            width = DEFAULT_WIDTH

    end_x = start_x + length
    half_w = width / 2.0
    dx = length / num_rungs
    tower_h = height + 0.35  # tall pylons (0.53 m) to clear the duck's head (0.43 m)

    # Material & cordage color tokens
    wood = "0.38 0.28 0.18 1"
    plank = "0.48 0.36 0.22 1"
    rope = "0.72 0.58 0.38 1"       # natural simple twisted rope ("طناب ساده")
    rope_dark = "0.58 0.44 0.28 1"  # twisted cord strands
    knot = "0.50 0.38 0.22 1"       # rope knots
    cable = "0.40 0.32 0.20 1"      # suspension cable

    # Contact compliance settings for soft rope surface:
    # contype="1" conaffinity="0": Collides with robot feet (which have conaffinity=1),
    # but does NOT collide with adjacent rungs, avoiding internal constraint explosion.
    if soft:
        rope_contact = 'contype="1" conaffinity="0" friction="1.8 0.01 0.001" solref="0.015 1.0" solimp="0.85 0.98 0.01 0.5 2"'
    else:
        rope_contact = 'contype="1" conaffinity="0" friction="1.2 0.005 0.0001"'

    fixed = []

    # 1. Approach Landing (Solid timber platform at start)
    entry_cx = start_x - landing_len / 2.0
    fixed.append(
        f'<geom name="landing_entry" type="box" pos="{entry_cx:.4f} 0 {height-0.006:.4f}" '
        f'size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.006" rgba="{plank}" friction="1.8 0.01 0.001"/>'
    )
    fixed.append(
        f'<geom name="landing_entry_base" type="box" pos="{entry_cx:.4f} 0 {(height-0.012)/2:.4f}" '
        f'size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.012)/2:.4f}" rgba="{wood}"/>'
    )

    # 2. Exit Landing (Solid timber finish platform)
    exit_cx = end_x + landing_len / 2.0
    fixed.append(
        f'<geom name="landing_exit" type="box" pos="{exit_cx:.4f} 0 {height-0.006:.4f}" '
        f'size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.006" rgba="{plank}" friction="1.8 0.01 0.001"/>'
    )
    fixed.append(
        f'<geom name="landing_exit_base" type="box" pos="{exit_cx:.4f} 0 {(height-0.012)/2:.4f}" '
        f'size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.012)/2:.4f}" rgba="{wood}"/>'
    )

    # 3. Pylons / Towers (Entry and Exit support structures)
    for x in (start_x, end_x):
        for s in (-1, 1):
            fixed.append(
                f'<geom name="pylon_{x:+.2f}_{s:+d}" type="capsule" '
                f'fromto="{x:.4f} {s*half_w:.4f} 0 {x:.4f} {s*half_w:.4f} {tower_h:.4f}" '
                f'size="0.016" rgba="{wood}"/>'
            )
        fixed.append(
            f'<geom name="pylon_top_{x:+.2f}" type="capsule" '
            f'fromto="{x:.4f} {-half_w:.4f} {tower_h:.4f} {x:.4f} {half_w:.4f} {tower_h:.4f}" '
            f'size="0.012" rgba="{wood}" contype="0" conaffinity="0"/>'
        )

    # 4. Overhead Main Catenary Suspension Cables
    c_steps = 30
    for s in (-1, 1):
        for i in range(c_steps):
            u0, u1 = i / c_steps, (i + 1) / c_steps
            x0 = start_x + u0 * length
            x1 = start_x + u1 * length
            z0 = tower_h - 0.01 - 0.12 * 4.0 * u0 * (1.0 - u0)
            z1 = tower_h - 0.01 - 0.12 * 4.0 * u1 * (1.0 - u1)
            fixed.append(
                f'<geom name="cable_{s:+d}_{i}" type="capsule" '
                f'fromto="{x0:.4f} {s*half_w:.4f} {z0:.4f} {x1:.4f} {s*half_w:.4f} {z1:.4f}" '
                f'size="0.006" rgba="{cable}"/>'
            )

    fixed_xml = '<body name="bridge_anchors">\n' + "\n".join(fixed) + "\n</body>\n"

    # 5. Articulated Soft Rope Net Walkway (Independently Compliant Rungs)
    rungs = []
    rung_mass = 0.06

    for i in range(num_rungs):
        x = start_x + (i + 0.5) * dx
        u = (i + 0.5) / num_rungs

        # Static baseline catenary droop
        static_droop = sag * 4.0 * u * (1.0 - u)
        z0 = height - static_droop

        # Parabolic compliance distribution: softest at center, firmer at bank anchors
        parabola = 4.0 * u * (1.0 - u)
        if sway:
            kz = center_kz + (1.0 - parabola) * 350.0
            kr = center_kr + (1.0 - parabola) * 12.0
            kp = center_kp + (1.0 - parabola) * 15.0
        else:
            kz, kr, kp = 2500.0, 80.0, 80.0

        # Critical damping matched to combined robot/rung mass
        dz = 2.0 * damping_ratio * np.sqrt(kz * (rung_mass + 0.05))
        dr = 2.0 * damping_ratio * np.sqrt(kr * 0.015)
        dp = 2.0 * damping_ratio * np.sqrt(kp * 0.015)

        # Vertical hanger ropes connecting rung edges to overhead catenary cable
        cable_z = tower_h - 0.01 - 0.12 * 4.0 * u * (1.0 - u)
        hanger_h = cable_z - z0
        h_L = f'<geom name="h_L_{i}" type="capsule" fromto="0 {half_w:.4f} 0 0 {half_w:.4f} {hanger_h:.4f}" size="0.003" rgba="{rope_dark}"/>'
        h_R = f'<geom name="h_R_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {-half_w:.4f} {hanger_h:.4f}" size="0.003" rgba="{rope_dark}"/>'

        # Main walkable tread surface: flat base
        tread = f'<geom name="tread_{i}" type="box" pos="0 0 0" size="{dx*0.48:.4f} {half_w:.4f} 0.006" mass="{rung_mass*0.5:.4f}" rgba="{rope}" {rope_contact}/>'
        cross_r = f'<geom name="cross_{i}" type="capsule" fromto="0 {-half_w:.4f} 0.006 0 {half_w:.4f} 0.006" size="0.006" mass="{rung_mass*0.2:.4f}" rgba="{rope}" contype="0" conaffinity="0"/>'

        # Leading and trailing edge capsules: rounded transition between adjacent rungs prevents toe tripping
        edge_f = f'<geom name="edge_f_{i}" type="capsule" fromto="{dx*0.44:.4f} {-half_w:.4f} 0.002 {dx*0.44:.4f} {half_w:.4f} 0.002" size="0.005" mass="{rung_mass*0.1:.4f}" rgba="{rope_dark}" {rope_contact}/>'
        edge_r = f'<geom name="edge_r_{i}" type="capsule" fromto="{-dx*0.44:.4f} {-half_w:.4f} 0.002 {-dx*0.44:.4f} {half_w:.4f} 0.002" size="0.005" mass="{rung_mass*0.1:.4f}" rgba="{rope_dark}" {rope_contact}/>'

        # Raised side rope curbs (guides feet back towards center and prevents lateral slip)
        curb_L = f'<geom name="curb_L_{i}" type="capsule" fromto="{-dx*0.46:.4f} {half_w-0.015:.4f} 0.012 {dx*0.46:.4f} {half_w-0.015:.4f} 0.012" size="0.010" rgba="{rope_dark}" contype="1" conaffinity="0" friction="1.5 0.01 0.001"/>'
        curb_R = f'<geom name="curb_R_{i}" type="capsule" fromto="{-dx*0.46:.4f} {-half_w+0.015:.4f} 0.012 {dx*0.46:.4f} {-half_w+0.015:.4f} 0.012" size="0.010" rgba="{rope_dark}" contype="1" conaffinity="0" friction="1.5 0.01 0.001"/>'

        # Visual knots and rope cord detailing ("طناب ساده")
        decor = []
        for y_str in np.linspace(-half_w + 0.03, half_w - 0.03, 7):
            decor.append(f'<geom name="k_{i}_{y_str:+.2f}" type="sphere" pos="0 {y_str:.4f} 0.006" size="0.008" rgba="{knot}" contype="0" conaffinity="0"/>')
            decor.append(f'<geom name="rib_{i}_{y_str:+.2f}" type="capsule" fromto="{-dx*0.46:.4f} {y_str:.4f} 0.006 {dx*0.46:.4f} {y_str:.4f} 0.006" size="0.003" rgba="{rope_dark}" contype="0" conaffinity="0"/>')

        b_xml = f"""
        <body name="soft_rung_{i}" pos="{x:.4f} 0 {z0:.4f}">
            <joint name="j_z_{i}" type="slide" axis="0 0 1" stiffness="{kz:.1f}" damping="{dz:.2f}" range="-0.15 0.05"/>
            <joint name="j_pitch_{i}" type="hinge" axis="0 1 0" stiffness="{kp:.1f}" damping="{dp:.2f}" range="-0.35 0.35"/>
            <joint name="j_roll_{i}" type="hinge" axis="1 0 0" stiffness="{kr:.1f}" damping="{dr:.2f}" range="-0.30 0.30"/>
            {tread}
            {edge_f}
            {edge_r}
            {curb_L}
            {curb_R}
            {cross_r}
            {h_L}
            {h_R}
            {' '.join(decor)}
        </body>
        """
        rungs.append(b_xml)

    # Soft inter-rung coupling: creates a continuous catenary deflection bowl ("طناب ساده")
    eq_lines = ["<equality>"]
    for i in range(num_rungs - 1):
        eq_lines.append(
            f'    <joint joint1="j_z_{i}" joint2="j_z_{i+1}" polycoef="0 1 0 0 0" solref="0.035 1.0" solimp="0.80 0.95 0.001"/>'
        )
        eq_lines.append(
            f'    <joint joint1="j_pitch_{i}" joint2="j_pitch_{i+1}" polycoef="0 1 0 0 0" solref="0.045 1.0" solimp="0.80 0.95 0.001"/>'
        )
    eq_lines.append("</equality>")
    equality_xml = "\n".join(eq_lines)

    body_xml = fixed_xml + "\n".join(rungs)
    if kwargs.get("return_equality", False):
        return body_xml, equality_xml
    return body_xml


def write_scene(bridge_xml_str, target_file=None, equality_xml_str=""):
    """Embed bridge XML into scene_beam template and write to file."""
    with open(TEMPLATE) as f:
        template = f.read()

    needle = '<geom name="beam" type="box" size="1.0000 0.0500 0.0200" pos="0.7000 0 0.0200" rgba="0.85 0.65 0.35 1" />'
    if needle not in template:
        raise RuntimeError(f"Could not find beam geom placeholder in {TEMPLATE}")

    scene_content = template.replace(needle, bridge_xml_str)
    if equality_xml_str:
        scene_content = scene_content.replace("</mujoco>", f"{equality_xml_str}\n</mujoco>")

    if target_file is None:
        target_file = str(paths.model("scene_bridge_active.xml"))

    Path(target_file).parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w") as f:
        f.write(scene_content)
    return target_file


def run(
    sag=DEFAULT_SAG,
    width=DEFAULT_WIDTH,
    length=DEFAULT_LENGTH,
    height=DECK_HEIGHT,
    speed=DEFAULT_SPEED,
    seconds=24.0,
    center_kz=DEFAULT_CENTER_KZ,
    center_kr=DEFAULT_CENTER_KR,
    center_kp=DEFAULT_CENTER_KP,
    num_rungs=DEFAULT_NUM_RUNGS,
    k_lat=7.0,
    k_yaw=3.5,
    max_aim=0.15,
    max_turn=0.45,
    soft=True,
    sway=True,
    walking=WALKING_ONNX,
    video=None,
    report=None,
    trace=None,
    verbose=True,
):
    """Execute one suspension bridge crossing run. Returns result dict."""
    say = print if verbose else (lambda *a, **k: None)
    quiet = contextlib.redirect_stdout(io.StringIO())

    half_w = width / 2.0
    start_x = DEFAULT_START_X  # 0.00 m
    end_x = start_x + length   # 2.00 m

    b_xml, eq_xml = bridge_xml(
        start_x=start_x,
        length=length,
        width=width,
        height=height,
        sag=sag,
        num_rungs=num_rungs,
        center_kz=center_kz,
        center_kr=center_kr,
        center_kp=center_kp,
        soft=soft,
        sway=sway,
        return_equality=True,
    )
    scene_file = write_scene(b_xml, equality_xml_str=eq_xml)

    try:
        model, data = duck_sim.load_scene(scene_file)
        with quiet:
            policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=walking)

        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        floor = gid("floor")

        # Spawn duck on solid approach landing: x = -0.20 m, y = 0.0, heading +x
        data.qpos[adr + 0] = -0.20
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = height + 0.125
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        say(
            f"Bridge: length={length:.2f} m (x=[{start_x:+.2f}, {end_x:+.2f}]), "
            f"width={width * 1000:.0f} mm, baseline sag={sag * 1000:.0f} mm, "
            f"center_kz={center_kz:.1f} N/m (soft rope net), sway={sway}"
        )
        finish_x = end_x - 0.10
        say(f"Target speed: {speed:.2f} m/s; start x=-0.20 m; finish line at x={finish_x:+.2f} m")

        rec = None
        if video:
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            try:
                rec = duck_sim.Recorder(model, fps=25, distance=1.2, azimuth=120.0, elevation=-14.0)
            except Exception as e:
                say(f"Warning: Offscreen renderer initialization failed ({e}). Video recording skipped.")

        steps = int(round(seconds / CONTROL_DT))
        res = {
            "sag_mm": sag * 1000,
            "width_mm": width * 1000,
            "length_m": length,
            "center_kz": center_kz,
            "num_rungs": num_rungs,
            "soft_net": soft,
            "sway": sway,
            "speed_cmd": speed,
            "distance_m": 0.0,
            "walk_time_s": 0.0,
            "max_lat_error_mm": 0.0,
            "max_yaw_deg": 0.0,
            "max_sink_mm": 0.0,
            "max_rung_roll_deg": 0.0,
            "max_rung_pitch_deg": 0.0,
            "max_duck_roll_deg": 0.0,
            "max_duck_pitch_deg": 0.0,
            "deepest_net_penetration_mm": 0.0,
            "success": False,
            "cross_time_s": None,
            "fail_time_s": None,
            "fail_reason": None,
            "min_trunk_z": float("inf"),
            "max_trunk_z": float("-inf"),
        }

        phase = "settle"
        walk_t0 = None
        x0 = -0.20
        turn = 0.0
        rows = []
        max_penetration = 0.0
        max_sink = 0.0
        max_rung_roll = 0.0
        max_rung_pitch = 0.0
        max_duck_roll = 0.0
        max_duck_pitch = 0.0

        with quiet:
            policy.set_vel_cmd(0.0, 0.0, 0.0)

        for i in range(steps):
            t = i * CONTROL_DT
            x = float(data.qpos[adr])
            y = float(data.qpos[adr + 1])
            z = float(data.qpos[adr + 2])

            qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            duck_roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
            duck_pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))

            max_duck_roll = max(max_duck_roll, abs(duck_roll))
            max_duck_pitch = max(max_duck_pitch, abs(duck_pitch))

            res["min_trunk_z"] = min(res["min_trunk_z"], z)
            res["max_trunk_z"] = max(res["max_trunk_z"], z)

            # Query dynamic bridge rung deflections
            for r in range(num_rungs):
                j_zid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_z_{r}")
                if j_zid >= 0:
                    sink = -float(data.qpos[model.jnt_qposadr[j_zid]])
                    if sink > max_sink:
                        max_sink = sink

                j_rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_roll_{r}")
                if j_rid >= 0:
                    r_roll = abs(float(data.qpos[model.jnt_qposadr[j_rid]]))
                    if r_roll > max_rung_roll:
                        max_rung_roll = r_roll

                j_pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_pitch_{r}")
                if j_pid >= 0:
                    r_pitch = abs(float(data.qpos[model.jnt_qposadr[j_pid]]))
                    if r_pitch > max_rung_pitch:
                        max_rung_pitch = r_pitch

            # Contact penetrations
            for c in range(data.ncon):
                d_val = data.contact[c].dist
                if d_val < 0:
                    max_penetration = max(max_penetration, -d_val)

            if phase == "settle" and t >= SETTLE_S:
                phase = "walk"
                walk_t0 = t
                x0 = x
                say(f"t={t:4.1f}s: Started walking from x={x:+.3f} m, y={y * 1000:+.1f} mm")

            if phase == "walk":
                aim = float(np.clip(-k_lat * y, -max_aim, max_aim))
                turn = float(np.clip(k_yaw * (aim - yaw), -max_turn, max_turn))
                if t - walk_t0 < 0.4:
                    turn = 0.0  # establish steady forward gait before turning
                with quiet:
                    policy.set_vel_cmd(speed, 0.0, turn)

                res["distance_m"] = x - x0
                res["walk_time_s"] = t - walk_t0
                res["max_lat_error_mm"] = max(res["max_lat_error_mm"], abs(y) * 1000)
                res["max_yaw_deg"] = max(res["max_yaw_deg"], abs(np.degrees(yaw)))

                if x >= finish_x and not res["success"]:
                    res["success"] = True
                    res["cross_time_s"] = t - walk_t0
                    phase = "done"
                    with quiet:
                        policy.set_vel_cmd(0.0, 0.0, 0.0)
                    say(
                        f"t={t:4.1f}s: SUCCESS! Crossed finish line! Distance={res['distance_m']:.2f} m "
                        f"in {res['cross_time_s']:.1f} s | Max sink={max_sink*1000:.1f} mm, "
                        f"Rung roll={np.degrees(max_rung_roll):.1f}°"
                    )

            policy.apply_action(policy.infer())
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            rows.append([t, x, y, z, yaw, turn, max_sink, max_rung_roll])

            if rec is not None:
                rec.maybe_capture(i, data)

            # Failure checks
            if phase in ("settle", "walk"):
                fail = None
                for c in range(data.ncon):
                    con = data.contact[c]
                    g1, g2 = con.geom1, con.geom2
                    if floor in (g1, g2):
                        other = g2 if g1 == floor else g1
                        b_id = model.geom_bodyid[other]
                        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or ""
                        if b_name.startswith("he_") or b_name == "trunk_base":
                            fail = "Robot touched the floor below the bridge"
                            break

                # Tumble / fall checks
                if abs(duck_roll) > np.radians(45.0) or abs(duck_pitch) > np.radians(45.0):
                    fail = f"Robot tipped over (roll={np.degrees(duck_roll):.1f}°, pitch={np.degrees(duck_pitch):.1f}°)"

                if z < 0.10:
                    fail = f"Trunk dropped below recovery height (z={z:.3f} m)"

                if abs(y) > half_w + 0.06:
                    fail = f"Lateral bridge boundary breached (y={y * 1000:+.1f} mm)"

                if fail:
                    res["fail_reason"] = fail
                    res["fail_time_s"] = t
                    say(f"t={t:4.1f}s: Failed - {fail}")
                    break

        if rec is not None:
            rec.write(video)

    finally:
        if os.path.exists(scene_file):
            os.remove(scene_file)

    res["final_x"] = x
    res["final_y_mm"] = y * 1000
    res["max_sink_mm"] = max_sink * 1000
    res["max_rung_roll_deg"] = float(np.degrees(max_rung_roll))
    res["max_rung_pitch_deg"] = float(np.degrees(max_rung_pitch))
    res["max_duck_roll_deg"] = float(np.degrees(max_duck_roll))
    res["max_duck_pitch_deg"] = float(np.degrees(max_duck_pitch))
    res["deepest_net_penetration_mm"] = max_penetration * 1000

    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(res, indent=2) + "\n")

    if trace:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace, trajectory=np.array(rows))

    say(json.dumps(res, indent=2))
    return res


def evaluate(
    center_kz=DEFAULT_CENTER_KZ,
    sag=DEFAULT_SAG,
    speed=DEFAULT_SPEED,
    seconds=28.0,
    soft=True,
    sway=True,
    video=None,
    report=None,
):
    """Automated evaluation helper for testing."""
    return run(
        sag=sag,
        center_kz=center_kz,
        speed=speed,
        seconds=seconds,
        soft=soft,
        sway=sway,
        video=video,
        report=report,
        verbose=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--center-kz", type=float, default=DEFAULT_CENTER_KZ, help="Bridge center vertical stiffness (N/m)")
    ap.add_argument("--sag", type=float, default=DEFAULT_SAG, help="Bridge static baseline catenary sag in metres")
    ap.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="Bridge deck width in metres")
    ap.add_argument("--length", type=float, default=DEFAULT_LENGTH, help="Bridge span length in metres")
    ap.add_argument("--num-rungs", type=int, default=DEFAULT_NUM_RUNGS, help="Number of articulated rungs along span")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Forward commanded speed in m/s")
    ap.add_argument("--seconds", type=float, default=24.0, help="Simulation duration in seconds")
    ap.add_argument("--no-soft", dest="soft", action="store_false", help="Disable soft contact compliance on the net")
    ap.add_argument("--no-sway", dest="sway", action="store_false", help="Disable dynamic compliant suspension joints")
    ap.add_argument("--sweep-kz", nargs="+", type=float, help="Sweep across multiple center stiffness values (N/m)")
    ap.add_argument("--sweep", nargs="+", type=float, help="Sweep across multiple sag values (metres)")
    ap.add_argument("--video", help="Path to save MP4 video recording")
    ap.add_argument("--report", help="Path to save JSON report")
    ap.add_argument("--trace", help="Path to save NPZ trajectory trace")
    args = ap.parse_args()

    if args.sweep_kz:
        results = []
        for kz in args.sweep_kz:
            print(f"\n{'=' * 20} Testing Center Stiffness kz = {kz:.1f} N/m {'=' * 20}")
            r = run(
                sag=args.sag,
                width=args.width,
                length=args.length,
                speed=args.speed,
                seconds=args.seconds,
                center_kz=kz,
                num_rungs=args.num_rungs,
                soft=args.soft,
                sway=args.sway,
                verbose=True,
            )
            results.append(r)
        print("\nStiffness Sweep Summary:")
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"kz {r['center_kz']:5.1f} N/m | {status} | Dist: {r['distance_m']:.2f} m | "
                f"Time: {r['cross_time_s'] if r['cross_time_s'] else r['walk_time_s']:.1f} s | "
                f"Max sink: {r['max_sink_mm']:.1f} mm | Rung roll: {r['max_rung_roll_deg']:.1f}° | "
                f"Duck roll: {r['max_duck_roll_deg']:.1f}°"
            )
    elif args.sweep:
        results = []
        for s in args.sweep:
            print(f"\n{'=' * 20} Testing sag = {s * 1000:.0f} mm {'=' * 20}")
            r = run(
                sag=s,
                width=args.width,
                length=args.length,
                speed=args.speed,
                seconds=args.seconds,
                center_kz=args.center_kz,
                num_rungs=args.num_rungs,
                soft=args.soft,
                sway=args.sway,
                verbose=True,
            )
            results.append(r)
        print("\nSag Sweep Summary:")
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"Sag {r['sag_mm']:3.0f} mm | {status} | Dist: {r['distance_m']:.2f} m | "
                f"Time: {r['cross_time_s'] if r['cross_time_s'] else r['walk_time_s']:.1f} s | "
                f"Max sink: {r['max_sink_mm']:.1f} mm | Rung roll: {r['max_rung_roll_deg']:.1f}°"
            )
    else:
        run(
            sag=args.sag,
            width=args.width,
            length=args.length,
            speed=args.speed,
            seconds=args.seconds,
            center_kz=args.center_kz,
            num_rungs=args.num_rungs,
            soft=args.soft,
            sway=args.sway,
            video=args.video,
            report=args.report,
            trace=args.trace,
            verbose=True,
        )


if __name__ == "__main__":
    main()
