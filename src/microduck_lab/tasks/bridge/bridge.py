#!/usr/bin/env python3
"""Suspension bridge: walk the Microduck across a loose, sagging rope bridge.

The walking deck is constructed from a knotted rope mesh adapted from
src/microduck_lab/tasks/crawl/net.py, suspended between elevated approach and exit
platforms. The bridge features high catenary suspension cables, vertical suspender
hangers, knotted cross and longitudinal ropes, a deep central sag, and soft
compliant contact physics so the robot's feet sink into the net under body weight.

The pretrained walking policy (alpha_walking.onnx) provides locomotion. A closed-loop
steering controller feeds trunk lateral offset and yaw error into the policy twist
command to keep the duck tracking the bridge centreline through the sag.

    uv run python -m microduck_lab.tasks.bridge.bridge --sag 0.06
    uv run python -m microduck_lab.tasks.bridge.bridge --sag 0.06 --soft
    uv run python -m microduck_lab.tasks.bridge.bridge --sweep 0.03 0.04 0.05 0.06
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \
        src/microduck_lab/tasks/bridge/bridge.py --sag 0.06 --video videos/bridge/bridge_sag06.mp4

--video requires a GPU with EGL; simulation runs on CPU.
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION, FALL_HEIGHT

WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")
TEMPLATE = paths.model("scene_beam.xml")

# Default bridge geometry
BRIDGE_START_X = -0.25   # start of suspended span
BRIDGE_END_X = 1.75     # end of suspended span (2.0 m span)
HALF_WIDTH = 0.22       # half-width of deck (0.44 m total)
DECK_HEIGHT = 0.18      # height above floor
DEFAULT_SAG = 0.06      # 60 mm central catenary sag (loose droop)
ROPE_RADIUS = 0.0035    # soft braided rope thickness (m)
KNOT_RADIUS = 0.0055    # knot sphere radius (m)
TIMBER_RADIUS = 0.012   # perimeter timber thickness
SETTLE_S = 1.0          # stand still settling duration
UPRIGHT_MARGIN = 0.07   # trunk must be within deck - sag - margin to be upright


def bridge_xml(start_x=BRIDGE_START_X, end_x=BRIDGE_END_X, half_width=HALF_WIDTH,
               height=DECK_HEIGHT, sag=DEFAULT_SAG, soft=True, sway=False,
               rope_spacing=0.025, landing_len=0.30):
    """Procedural XML for a loose, soft rope suspension bridge."""
    fixed_parts = []
    deck_parts = []

    # Soft contact compliance settings:
    # solref: time constant = 0.04s, damping ratio = 1.0 (soft, non-oscillating compliance)
    # solimp: dmin=0.6, dmax=0.95, width=0.015m (15 mm cushion buffer before full resistance)
    # margin: 8 mm soft contact envelope where foot sinks into the rope mesh
    if soft:
        soft_attr = 'solref="0.04 1.0" solimp="0.6 0.95 0.015 0.5 2" margin="0.008" friction="1.3 0.01 0.0002"'
    else:
        soft_attr = 'friction="1.1 0.005 0.0001"'

    def capsule(parts_list, name, a, b, radius, rgba, extra=""):
        points = " ".join(f"{v:.7f}" for v in [*a, *b])
        parts_list.append(f'<geom name="{name}" type="capsule" fromto="{points}" size="{radius}" rgba="{rgba}" {extra}/>')

    def box(parts_list, name, pos, size, rgba, extra=""):
        pos_str = " ".join(f"{v:.7f}" for v in pos)
        size_str = " ".join(f"{v:.7f}" for v in size)
        parts_list.append(f'<geom name="{name}" type="box" pos="{pos_str}" size="{size_str}" rgba="{rgba}" {extra}/>')

    wood = "0.38 0.28 0.18 1"
    plank_wood = "0.48 0.36 0.22 1"
    rope = "0.15 0.17 0.11 1"
    cable = "0.22 0.20 0.16 1"
    metal = "0.35 0.35 0.38 1"

    # 1. Solid Approach & Exit Platforms (Landing Decks)
    entry_cx = start_x - landing_len / 2.0
    box(fixed_parts, "landing_entry", (entry_cx, 0.0, height - 0.01), (landing_len / 2.0, half_width + 0.02, 0.01), plank_wood)
    box(fixed_parts, "landing_entry_base", (entry_cx, 0.0, (height - 0.02) / 2.0), (landing_len / 2.0, half_width + 0.01, (height - 0.02) / 2.0), wood)

    exit_cx = end_x + landing_len / 2.0
    box(fixed_parts, "landing_exit", (exit_cx, 0.0, height - 0.01), (landing_len / 2.0, half_width + 0.02, 0.01), plank_wood)
    box(fixed_parts, "landing_exit_base", (exit_cx, 0.0, (height - 0.02) / 2.0), (landing_len / 2.0, half_width + 0.01, (height - 0.02) / 2.0), wood)

    # 2. Suspension Towers (Pylons at entry and exit banks)
    tower_h = height + 0.20
    for x in (start_x, end_x):
        for side in (-1, 1):
            y = side * half_width
            key = f"{x:+.2f}_{y:+.2f}".replace(".", "").replace("+", "p").replace("-", "m")
            capsule(fixed_parts, f"tower_post_{key}", (x, y, 0.0), (x, y, tower_h), 0.016, wood)
            fixed_parts.append(f'<geom name="tower_cap_{key}" type="cylinder" pos="{x} {y} {tower_h + 0.005}" size="0.019 0.006" rgba="{metal}"/>')
        capsule(fixed_parts, f"tower_cross_{x:+.2f}".replace(".", "").replace("+", "p").replace("-", "m"),
                (x, -half_width, tower_h - 0.01), (x, half_width, tower_h - 0.01), 0.011, wood)

    # 3. Main Suspension Cables (Left and Right Deep Catenary Curves)
    cable_steps = 30
    cable_xs = np.linspace(start_x, end_x, cable_steps + 1)
    cable_sag = max(0.08, sag * 1.5)

    def cable_z(u):
        return tower_h - 0.01 - cable_sag * 4.0 * u * (1.0 - u)

    for side in (-1, 1):
        y = side * half_width
        s_tag = "L" if side > 0 else "R"
        for i in range(cable_steps):
            u0, u1 = i / cable_steps, (i + 1) / cable_steps
            p0 = (cable_xs[i], y, cable_z(u0))
            p1 = (cable_xs[i + 1], y, cable_z(u1))
            capsule(deck_parts, f"main_cable_{s_tag}_{i}", p0, p1, 0.0040, cable)

    # 4. Knotted Rope Mesh Deck with Deep Sag & Soft Compliant Physics
    xs = np.linspace(start_x, end_x, round((end_x - start_x) / rope_spacing) + 1)
    ys = np.linspace(-half_width, half_width, round(2 * half_width / rope_spacing) + 1)

    def deck_point(x, y):
        u = (x - start_x) / (end_x - start_x)
        v = (y + half_width) / (2.0 * half_width)
        # 3D catenary hammock curve: longitudinal sag * lateral concave trough
        z = height - sag * np.sin(np.pi * u) * np.sin(np.pi * v)
        return (x, y, z)

    # Cross ropes (soft compliant)
    for i, x in enumerate(xs):
        for j in range(len(ys) - 1):
            capsule(deck_parts, f"net_cross_{i}_{j}", deck_point(x, ys[j]), deck_point(x, ys[j + 1]),
                    ROPE_RADIUS, rope, extra=soft_attr)

    # Longitudinal ropes (soft compliant)
    for j, y in enumerate(ys):
        for i in range(len(xs) - 1):
            capsule(deck_parts, f"net_long_{i}_{j}", deck_point(xs[i], y), deck_point(xs[i + 1], y),
                    ROPE_RADIUS, rope, extra=soft_attr)

    # Knots at grid intersections (soft compliant spheres)
    for i in range(0, len(xs), 2):
        for j in range(0, len(ys), 2):
            pos_str = " ".join(f"{v:.7f}" for v in deck_point(xs[i], ys[j]))
            deck_parts.append(
                f'<geom name="net_knot_{i}_{j}" type="sphere" pos="{pos_str}" '
                f'size="{KNOT_RADIUS:.5f}" rgba="{rope}" {soft_attr}/>'
            )

    # Timber edge curbs along the deck sides
    for side in (-1, 1):
        y = side * half_width
        s_tag = "L" if side > 0 else "R"
        for i in range(len(xs) - 1):
            capsule(deck_parts, f"edge_beam_{s_tag}_{i}", deck_point(xs[i], y), deck_point(xs[i + 1], y),
                    TIMBER_RADIUS, wood)

    # 5. Vertical Suspenders (Hanger ropes connecting main cable to deck)
    hanger_every = max(1, len(xs) // 10)
    for i in range(0, len(xs), hanger_every):
        u = (xs[i] - start_x) / (end_x - start_x)
        cz = cable_z(u)
        for side in (-1, 1):
            y = side * half_width
            s_tag = "L" if side > 0 else "R"
            dz = deck_point(xs[i], y)[2]
            capsule(deck_parts, f"hanger_{s_tag}_{i}", (xs[i], y, dz), (xs[i], y, cz), 0.0025, rope)

    # 6. Assembly: Fixed Anchors Body + Suspended Deck Body
    fixed_xml = '<body name="bridge_anchors">\n' + "\n".join(fixed_parts) + "\n</body>\n"

    center_x = (start_x + end_x) / 2.0
    joint_xml = ""
    if sway:
        joint_xml = (
            f'<joint name="bridge_roll" type="hinge" pos="{center_x:.4f} 0 {height - sag / 2:.4f}" '
            f'axis="1 0 0" stiffness="40.0" damping="3.0" range="-0.15 0.15"/>\n'
            f'<joint name="bridge_pitch" type="hinge" pos="{center_x:.4f} 0 {height - sag / 2:.4f}" '
            f'axis="0 1 0" stiffness="60.0" damping="4.0" range="-0.15 0.15"/>\n'
        )

    deck_xml = f'<body name="bridge_deck">\n{joint_xml}' + "\n".join(deck_parts) + "\n</body>\n"
    return fixed_xml + deck_xml


def write_scene(bridge_xml_str, target_file=None):
    """Embed bridge XML into scene_beam template and write to file."""
    with open(TEMPLATE) as f:
        template = f.read()

    needle = '<geom name="beam" type="box" size="1.0000 0.0500 0.0200" pos="0.7000 0 0.0200" rgba="0.85 0.65 0.35 1" />'
    if needle not in template:
        raise RuntimeError(f"Could not find beam geom placeholder in {TEMPLATE}")

    scene_content = template.replace(needle, bridge_xml_str)
    if target_file is None:
        target_file = str(paths.model("scene_bridge_active.xml"))

    Path(target_file).parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w") as f:
        f.write(scene_content)
    return target_file


def run(sag=DEFAULT_SAG, width=2.0 * HALF_WIDTH, length=BRIDGE_END_X - BRIDGE_START_X,
        height=DECK_HEIGHT, speed=0.25, seconds=24.0, k_lat=2.2, k_yaw=1.2,
        max_aim=0.35, max_turn=0.8, soft=True, sway=False, walking=WALKING_ONNX, video=None,
        report=None, trace=None, verbose=True):
    """Execute one suspension bridge crossing run. Returns result dict."""
    say = print if verbose else (lambda *a, **k: None)
    quiet = contextlib.redirect_stdout(io.StringIO())

    half_w = width / 2.0
    start_x = -0.20
    end_x = start_x + length

    b_xml = bridge_xml(start_x=start_x, end_x=end_x, half_width=half_w, height=height,
                       sag=sag, soft=soft, sway=sway)
    scene_file = write_scene(b_xml)

    try:
        model, data = duck_sim.load_scene(scene_file)
        with quiet:
            policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=walking)

        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        floor = gid("floor")

        # Spawn duck on approach landing: x = 0.0, y = 0.0, heading +x
        data.qpos[adr + 0] = 0.0
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = height + 0.125
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        say(f"Bridge: length={length:.2f} m (x=[{start_x:+.2f}, {end_x:+.2f}]), "
            f"width={width * 1000:.0f} mm, sag={sag * 1000:.0f} mm, height={height * 1000:.0f} mm, "
            f"soft_net={soft}, sway={sway}")
        say(f"Target speed: {speed:.2f} m/s; finish line at x={end_x - 0.10:+.2f} m")

        rec = None
        if video:
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            rec = duck_sim.Recorder(model, fps=25, distance=1.1, azimuth=120.0, elevation=-14.0)

        finish_x = end_x - 0.10
        steps = int(round(seconds / CONTROL_DT))
        res = {
            "sag_mm": sag * 1000, "width_mm": width * 1000, "length_m": length,
            "soft_net": soft, "sway": sway, "speed_cmd": speed, "distance_m": 0.0,
            "walk_time_s": 0.0, "max_lat_error_mm": 0.0, "max_yaw_deg": 0.0,
            "deepest_net_penetration_mm": 0.0, "success": False, "cross_time_s": None,
            "fail_time_s": None, "fail_reason": None,
            "min_trunk_z": float("inf"), "max_trunk_z": float("-inf"),
        }

        phase = "settle"
        walk_t0 = None
        x0 = 0.0
        turn = 0.0
        rows = []
        max_penetration = 0.0

        with quiet:
            policy.set_vel_cmd(0.0, 0.0, 0.0)

        for i in range(steps):
            t = i * CONTROL_DT
            x, y = float(data.qpos[adr]), float(data.qpos[adr + 1])
            z = float(data.qpos[adr + 2])
            yaw = duck_sim.trunk_yaw(data, adr)

            res["min_trunk_z"] = min(res["min_trunk_z"], z)
            res["max_trunk_z"] = max(res["max_trunk_z"], z)

            # Measure soft net penetration
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
                    say(f"t={t:4.1f}s: Crossed finish line! Distance={res['distance_m']:.2f} m "
                        f"in {res['cross_time_s']:.1f} s (y={y * 1000:+.1f} mm)")

            policy.apply_action(policy.infer())
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            rows.append([t, x, y, z, yaw, turn])

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

                if z < height - sag - UPRIGHT_MARGIN:
                    fail = f"Trunk dropped below upright height (z={z:.3f} m)"

                if abs(y) > half_w + 0.05:
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
    res["deepest_net_penetration_mm"] = max_penetration * 1000

    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(res, indent=2) + "\n")

    if trace:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace, trajectory=np.array(rows))

    say(json.dumps(res, indent=2))
    return res


def evaluate(sag=DEFAULT_SAG, soft=True, sway=False, video=None, report=None):
    """Automated evaluation helper for testing."""
    return run(sag=sag, soft=soft, sway=sway, video=video, report=report, verbose=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sag", type=float, default=DEFAULT_SAG, help="Bridge central sag in metres")
    ap.add_argument("--width", type=float, default=2.0 * HALF_WIDTH, help="Bridge deck width in metres")
    ap.add_argument("--length", type=float, default=BRIDGE_END_X - BRIDGE_START_X, help="Bridge span length in metres")
    ap.add_argument("--speed", type=float, default=0.25, help="Forward commanded speed in m/s")
    ap.add_argument("--seconds", type=float, default=24.0, help="Simulation duration in seconds")
    ap.add_argument("--no-soft", dest="soft", action="store_false", help="Disable soft contact compliance on the net")
    ap.add_argument("--sway", action="store_true", help="Enable dynamic compliant suspension joints")
    ap.add_argument("--sweep", nargs="+", type=float, help="Sweep across multiple sag values (metres)")
    ap.add_argument("--video", help="Path to save MP4 video recording")
    ap.add_argument("--report", help="Path to save JSON report")
    ap.add_argument("--trace", help="Path to save NPZ trajectory trace")
    args = ap.parse_args()

    if args.sweep:
        results = []
        for s in args.sweep:
            print(f"\n{'=' * 20} Testing sag = {s * 1000:.0f} mm {'=' * 20}")
            r = run(sag=s, width=args.width, length=args.length, speed=args.speed,
                    seconds=args.seconds, soft=args.soft, sway=args.sway, verbose=True)
            results.append(r)
        print("\nSweep Summary:")
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            print(f"Sag {r['sag_mm']:3.0f} mm | {status} | Dist: {r['distance_m']:.2f} m | "
                  f"Time: {r['cross_time_s'] if r['cross_time_s'] else r['walk_time_s']:.1f} s | "
                  f"Max lat: {r['max_lat_error_mm']:.1f} mm | Soft sink: {r['deepest_net_penetration_mm']:.1f} mm")
    else:
        run(sag=args.sag, width=args.width, length=args.length, speed=args.speed,
            seconds=args.seconds, soft=args.soft, sway=args.sway, video=args.video,
            report=args.report, trace=args.trace, verbose=True)


if __name__ == "__main__":
    main()
