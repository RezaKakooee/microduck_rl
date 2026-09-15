import os
import sys
import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION

WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")
TEMPLATE = paths.model("scene_beam.xml")

def build_soft_bridge_xml(
    start_x=0.0,
    length=2.0,
    width=0.44,
    height=0.18,
    sag=0.035,
    num_rungs=24,
    center_kz=160.0,
    center_kr=2.2,
    center_kp=3.0,
    damping_ratio=1.4,
    landing_len=0.40,
):
    end_x = start_x + length
    half_w = width / 2.0
    dx = length / num_rungs
    tower_h = height + 0.22

    wood = "0.38 0.28 0.18 1"
    plank = "0.48 0.36 0.22 1"
    rope = "0.72 0.58 0.38 1"
    rope_dark = "0.58 0.44 0.28 1"
    knot = "0.50 0.38 0.22 1"
    cable = "0.40 0.32 0.20 1"

    rope_contact = 'friction="1.8 0.01 0.001" solref="0.015 1.0" solimp="0.85 0.98 0.01 0.5 2"'

    fixed = []
    # Approach Landing: from start_x - landing_len to start_x
    entry_cx = start_x - landing_len / 2.0
    fixed.append(f'<geom name="landing_entry" type="box" pos="{entry_cx:.4f} 0 {height-0.006:.4f}" size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.006" rgba="{plank}" friction="1.8 0.01 0.001"/>')
    fixed.append(f'<geom name="landing_entry_base" type="box" pos="{entry_cx:.4f} 0 {(height-0.012)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.012)/2:.4f}" rgba="{wood}"/>')

    # Exit Landing: from end_x to end_x + landing_len
    exit_cx = end_x + landing_len / 2.0
    fixed.append(f'<geom name="landing_exit" type="box" pos="{exit_cx:.4f} 0 {height-0.006:.4f}" size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.006" rgba="{plank}" friction="1.8 0.01 0.001"/>')
    fixed.append(f'<geom name="landing_exit_base" type="box" pos="{exit_cx:.4f} 0 {(height-0.012)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.012)/2:.4f}" rgba="{wood}"/>')

    # Towers
    for x in (start_x, end_x):
        for s in (-1, 1):
            fixed.append(f'<geom name="pylon_{x:.2f}_{s}" type="capsule" fromto="{x:.4f} {s*half_w:.4f} 0 {x:.4f} {s*half_w:.4f} {tower_h:.4f}" size="0.016" rgba="{wood}"/>')
        fixed.append(f'<geom name="pylon_top_{x:.2f}" type="capsule" fromto="{x:.4f} {-half_w:.4f} {tower_h:.4f} {x:.4f} {half_w:.4f} {tower_h:.4f}" size="0.012" rgba="{wood}"/>')

    # Overhead cables
    c_steps = 30
    for s in (-1, 1):
        for i in range(c_steps):
            u0, u1 = i / c_steps, (i + 1) / c_steps
            x0 = start_x + u0 * length
            x1 = start_x + u1 * length
            z0 = tower_h - 0.01 - 0.12 * 4.0 * u0 * (1.0 - u0)
            z1 = tower_h - 0.01 - 0.12 * 4.0 * u1 * (1.0 - u1)
            fixed.append(f'<geom name="cable_{s}_{i}" type="capsule" fromto="{x0:.4f} {s*half_w:.4f} {z0:.4f} {x1:.4f} {s*half_w:.4f} {z1:.4f}" size="0.006" rgba="{cable}"/>')

    fixed_xml = '<body name="bridge_anchors">\n' + '\n'.join(fixed) + '\n</body>\n'

    rungs = []
    rung_mass = 0.06

    for i in range(num_rungs):
        x = start_x + (i + 0.5) * dx
        u = (i + 0.5) / num_rungs
        static_droop = sag * 4.0 * u * (1.0 - u)
        z0 = height - static_droop

        parabola = 4.0 * u * (1.0 - u)
        kz = center_kz + (1.0 - parabola) * 350.0
        kr = center_kr + (1.0 - parabola) * 3.5
        kp = center_kp + (1.0 - parabola) * 4.0

        dz = 2.0 * damping_ratio * np.sqrt(kz * (rung_mass + 0.05))
        dr = 2.0 * damping_ratio * np.sqrt(kr * 0.015)
        dp = 2.0 * damping_ratio * np.sqrt(kp * 0.015)

        cable_z = tower_h - 0.01 - 0.12 * 4.0 * u * (1.0 - u)
        hanger_h = cable_z - z0
        h_L = f'<geom name="h_L_{i}" type="capsule" fromto="0 {half_w:.4f} 0 0 {half_w:.4f} {hanger_h:.4f}" size="0.003" rgba="{rope_dark}"/>'
        h_R = f'<geom name="h_R_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {-half_w:.4f} {hanger_h:.4f}" size="0.003" rgba="{rope_dark}"/>'

        # Main walkable tread surface: flat, flush, continuous
        tread = f'<geom name="tread_{i}" type="box" pos="0 0 0" size="{dx*0.495:.4f} {half_w:.4f} 0.006" mass="{rung_mass*0.6:.4f}" rgba="{rope}" {rope_contact}/>'
        cross_r = f'<geom name="cross_{i}" type="capsule" fromto="0 {-half_w:.4f} 0.006 0 {half_w:.4f} 0.006" size="0.006" mass="{rung_mass*0.4:.4f}" rgba="{rope}" contype="0" conaffinity="0"/>'

        # Visual knots and cord ribs
        decor = []
        for y_str in np.linspace(-half_w + 0.03, half_w - 0.03, 7):
            decor.append(f'<geom name="k_{i}_{y_str:+.2f}" type="sphere" pos="0 {y_str:.4f} 0.006" size="0.008" rgba="{knot}" contype="0" conaffinity="0"/>')
            decor.append(f'<geom name="rib_{i}_{y_str:+.2f}" type="capsule" fromto="{-dx*0.48:.4f} {y_str:.4f} 0.006 {dx*0.48:.4f} {y_str:.4f} 0.006" size="0.003" rgba="{rope_dark}" contype="0" conaffinity="0"/>')

        b_xml = f'''
        <body name="soft_rung_{i}" pos="{x:.4f} 0 {z0:.4f}">
            <joint name="j_z_{i}" type="slide" axis="0 0 1" stiffness="{kz:.1f}" damping="{dz:.2f}" range="-0.15 0.05"/>
            <joint name="j_pitch_{i}" type="hinge" axis="0 1 0" stiffness="{kp:.1f}" damping="{dp:.2f}" range="-0.35 0.35"/>
            <joint name="j_roll_{i}" type="hinge" axis="1 0 0" stiffness="{kr:.1f}" damping="{dr:.2f}" range="-0.30 0.30"/>
            {tread}
            {cross_r}
            {h_L}
            {h_R}
            {' '.join(decor)}
        </body>
        '''
        rungs.append(b_xml)

    return fixed_xml + '\n'.join(rungs)

def run_simulation(center_kz=180.0, speed=0.25):
    b_xml = build_soft_bridge_xml(start_x=0.0, length=2.0, center_kz=center_kz)
    with open(TEMPLATE) as f:
        tmpl = f.read()
    needle = '<geom name="beam" type="box" size="1.0000 0.0500 0.0200" pos="0.7000 0 0.0200" rgba="0.85 0.65 0.35 1" />'
    scene_str = tmpl.replace(needle, b_xml)
    scene_path = paths.model("scene_test_bridge_active.xml")
    with open(scene_path, "w") as f:
        f.write(scene_str)

    try:
        model, data = duck_sim.load_scene(str(scene_path))
        policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=WALKING_ONNX)

        # Position duck on approach landing
        data.qpos[adr + 0] = -0.20
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = 0.18 + 0.125
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        max_sink = 0.0
        max_rung_roll = 0.0
        success = False
        finish_x = 1.90

        print(f"=== Starting Run: center_kz={center_kz:.1f} N/m, speed={speed:.2f} m/s ===", flush=True)

        for step in range(500): # 10s
            t = step * CONTROL_DT
            x = float(data.qpos[adr])
            y = float(data.qpos[adr + 1])
            z = float(data.qpos[adr + 2])
            yaw = duck_sim.trunk_yaw(data, adr)
            qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
            duck_roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))

            for r in range(24):
                j_zid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_z_{r}")
                if j_zid >= 0:
                    sink = -float(data.qpos[model.jnt_qposadr[j_zid]])
                    max_sink = max(max_sink, sink)
                j_rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_roll_{r}")
                if j_rid >= 0:
                    max_rung_roll = max(max_rung_roll, abs(float(data.qpos[model.jnt_qposadr[j_rid]])))

            if t < 1.0:
                policy.set_vel_cmd(0.0, 0.0, 0.0)
            else:
                aim = float(np.clip(-1.8 * y, -0.25, 0.25))
                turn = float(np.clip(1.0 * (aim - yaw), -0.45, 0.45))
                if t < 1.4:
                    turn = 0.0
                policy.set_vel_cmd(speed, 0.0, turn)

            policy.apply_action(policy.infer())
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            if step % 50 == 0:
                print(f"t={t:4.1f}s | x={x:+.3f}m y={y*1000:+4.0f}mm z={z:+.3f}m | sink={max_sink*1000:4.1f}mm rung_roll={np.degrees(max_rung_roll):4.1f}° duck_roll={np.degrees(duck_roll):+4.1f}°", flush=True)

            if x >= finish_x and not success:
                print(f"--> SUCCESS! Crossed bridge! x={x:.2f}m at t={t:.1f}s | max_sink={max_sink*1000:.1f}mm, rung_roll={np.degrees(max_rung_roll):.1f}°", flush=True)
                success = True
                break

            if t > 1.0 and (abs(duck_roll) > np.radians(45.0) or z < 0.10):
                print(f"--> FELL at t={t:.1f}s: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m, duck_roll={np.degrees(duck_roll):.1f}°", flush=True)
                break

        print(f"FINAL: success={success}, max_sink={max_sink*1000:.1f}mm, max_rung_roll={np.degrees(max_rung_roll):.1f}°\n", flush=True)
        return success
    finally:
        if os.path.exists(scene_path):
            os.remove(scene_path)

if __name__ == "__main__":
    run_simulation(center_kz=180.0, speed=0.25)
