import os
import sys
import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION

WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")
TEMPLATE = paths.model("scene_beam.xml")

def build_bridge_xml(
    start_x=-0.20,
    length=2.0,
    width=0.44,
    height=0.18,
    static_sag=0.035,
    num_rungs=24,
    center_kz=180.0,
    center_kr=2.0,
    center_kp=2.5,
    damping_ratio=1.2,
):
    end_x = start_x + length
    half_w = width / 2.0
    dx = length / num_rungs
    landing_len = 0.40
    tower_h = height + 0.22
    
    wood = "0.38 0.28 0.18 1"
    plank = "0.48 0.36 0.22 1"
    rope = "0.72 0.58 0.38 1"       # hemp rope
    rope_dark = "0.58 0.44 0.28 1"  # twisted cord
    knot = "0.50 0.38 0.22 1"       # knot
    cable = "0.40 0.32 0.20 1"      # suspension cable
    
    fixed = []
    
    # Approach Landing: from start_x - landing_len to start_x
    entry_cx = start_x - landing_len / 2.0
    fixed.append(f'<geom name="landing_entry" type="box" pos="{entry_cx:.4f} 0 {height-0.01:.4f}" size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.01" rgba="{plank}" friction="1.8 0.01 0.001"/>')
    fixed.append(f'<geom name="landing_entry_base" type="box" pos="{entry_cx:.4f} 0 {(height-0.02)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.02)/2:.4f}" rgba="{wood}"/>')

    # Exit Landing: from end_x to end_x + landing_len
    exit_cx = end_x + landing_len / 2.0
    fixed.append(f'<geom name="landing_exit" type="box" pos="{exit_cx:.4f} 0 {height-0.01:.4f}" size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.01" rgba="{plank}" friction="1.8 0.01 0.001"/>')
    fixed.append(f'<geom name="landing_exit_base" type="box" pos="{exit_cx:.4f} 0 {(height-0.02)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.02)/2:.4f}" rgba="{wood}"/>')

    # Towers (Pylons)
    for x in (start_x, end_x):
        for s in (-1, 1):
            fixed.append(f'<geom name="pylon_{x:.2f}_{s}" type="capsule" fromto="{x:.4f} {s*half_w:.4f} 0 {x:.4f} {s*half_w:.4f} {tower_h:.4f}" size="0.016" rgba="{wood}"/>')
        fixed.append(f'<geom name="pylon_top_{x:.2f}" type="capsule" fromto="{x:.4f} {-half_w:.4f} {tower_h:.4f} {x:.4f} {half_w:.4f} {tower_h:.4f}" size="0.012" rgba="{wood}"/>')

    # Overhead suspension cables
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

    # Dynamic Soft Rope Net Walkway
    rungs = []
    rung_mass = 0.06
    rope_contact = 'friction="1.8 0.01 0.001" solref="0.015 1.0" solimp="0.85 0.98 0.01 0.5 2"'
    
    for i in range(num_rungs):
        x = start_x + (i + 0.5) * dx
        u = (i + 0.5) / num_rungs
        
        sag = static_sag * 4.0 * u * (1.0 - u)
        z0 = height - sag
        
        parabola = 4.0 * u * (1.0 - u)
        kz = center_kz + (1.0 - parabola) * 350.0
        kr = center_kr + (1.0 - parabola) * 3.5
        kp = center_kp + (1.0 - parabola) * 4.0
        
        dz = 2.0 * damping_ratio * np.sqrt(kz * (rung_mass + 0.04))
        dr = 2.0 * damping_ratio * np.sqrt(kr * 0.01)
        dp = 2.0 * damping_ratio * np.sqrt(kp * 0.01)
        
        cable_z = tower_h - 0.01 - 0.12 * 4.0 * u * (1.0 - u)
        h_L = f'<geom name="h_L_{i}" type="capsule" fromto="0 {half_w:.4f} 0 0 {half_w:.4f} {cable_z - z0:.4f}" size="0.003" rgba="{rope_dark}"/>'
        h_R = f'<geom name="h_R_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {-half_w:.4f} {cable_z - z0:.4f}" size="0.003" rgba="{rope_dark}"/>'
        
        cross_r = f'<geom name="cross_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {half_w:.4f} 0" size="0.007" mass="{rung_mass*0.3:.4f}" rgba="{rope}" {rope_contact}/>'
        tread = f'<geom name="tread_{i}" type="box" pos="0 0 -0.004" size="{dx*0.49:.4f} {half_w:.4f} 0.004" mass="{rung_mass*0.4:.4f}" rgba="{rope}" {rope_contact}/>'
        
        knots = []
        for y_str in np.linspace(-half_w + 0.03, half_w - 0.03, 7):
            knots.append(f'<geom name="k_{i}_{y_str:+.2f}" type="sphere" pos="0 {y_str:.4f} 0.002" size="0.009" mass="{rung_mass*0.03:.4f}" rgba="{knot}" {rope_contact}/>')
            knots.append(f'<geom name="rib_{i}_{y_str:+.2f}" type="capsule" fromto="{-dx*0.48:.4f} {y_str:.4f} 0 {dx*0.48:.4f} {y_str:.4f} 0" size="0.004" rgba="{rope_dark}" {rope_contact}/>')

        b_xml = f'''
        <body name="soft_rung_{i}" pos="{x:.4f} 0 {z0:.4f}">
            <joint name="j_z_{i}" type="slide" axis="0 0 1" stiffness="{kz:.1f}" damping="{dz:.2f}" range="-0.14 0.04"/>
            <joint name="j_pitch_{i}" type="hinge" axis="0 1 0" stiffness="{kp:.1f}" damping="{dp:.2f}" range="-0.35 0.35"/>
            <joint name="j_roll_{i}" type="hinge" axis="1 0 0" stiffness="{kr:.1f}" damping="{dr:.2f}" range="-0.30 0.30"/>
            {cross_r}
            {tread}
            {h_L}
            {h_R}
            {' '.join(knots)}
        </body>
        '''
        rungs.append(b_xml)
        
    return fixed_xml + '\n'.join(rungs)

def test_bridge_run():
    xml_str = build_bridge_xml(center_kz=180.0)
    with open(TEMPLATE) as f:
        tmpl = f.read()
    needle = '<geom name="beam" type="box" size="1.0000 0.0500 0.0200" pos="0.7000 0 0.0200" rgba="0.85 0.65 0.35 1" />'
    scene_str = tmpl.replace(needle, xml_str)
    
    scene_path = paths.model("scene_verify_active.xml")
    with open(scene_path, "w") as f:
        f.write(scene_str)
        
    try:
        model, data = duck_sim.load_scene(str(scene_path))
        policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=WALKING_ONNX)
        
        # Spawn duck as bridge.py did: x = 0.0, y = 0.0, z = height + 0.125
        data.qpos[adr + 0] = 0.0
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = 0.18 + 0.125
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        
        speed = 0.25
        k_lat = 2.2
        k_yaw = 1.2
        max_aim = 0.35
        max_turn = 0.8
        
        max_sink = 0.0
        max_roll = 0.0
        max_pitch = 0.0
        success = False
        finish_x = 1.65
        out_file = "/home2/reza/.gemini/antigravity-ide/brain/6dd2253f-9596-47e0-bc95-17574bbd4ff8/scratch/verify_out.txt"
        
        with open(out_file, "w") as out_f:
            out_f.write("Starting rollout from x=0.0 on articulated loose rope net...\n")
            
            for step in range(550): # 11 seconds
                t = step * CONTROL_DT
                x = float(data.qpos[adr])
                y = float(data.qpos[adr + 1])
                z = float(data.qpos[adr + 2])
                qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
                yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
                roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
                pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
                
                for r in range(24):
                    q_z = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_z_{r}")]
                    sink = -data.qpos[q_z]
                    if sink > max_sink:
                        max_sink = sink
                    q_r = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_roll_{r}")]
                    max_roll = max(max_roll, abs(data.qpos[q_r]))
                    q_p = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_pitch_{r}")]
                    max_pitch = max(max_pitch, abs(data.qpos[q_p]))
                    
                if t < 1.0:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
                else:
                    aim = float(np.clip(-k_lat * y, -max_aim, max_aim))
                    turn = float(np.clip(k_yaw * (aim - yaw), -max_turn, max_turn))
                    policy.set_vel_cmd(speed, 0.0, turn)
                    
                policy.apply_action(policy.infer())
                for _ in range(DECIMATION):
                    mujoco.mj_step(model, data)
                    
                if step % 25 == 0:
                    line = f"t={t:4.1f}s | x={x:+.3f}m y={y*1000:+4.0f}mm z={z:+.3f}m | max_sink={max_sink*1000:4.1f}mm net_roll={np.degrees(max_roll):4.1f}° | duck_roll={np.degrees(roll):+5.1f}° duck_pitch={np.degrees(pitch):+5.1f}°\n"
                    out_f.write(line)
                    out_f.flush()
                    
                if x >= finish_x and not success:
                    out_f.write(f"SUCCESS! Reached finish line x={x:.2f}m at t={t:.1f}s!\n")
                    success = True
                    break
                    
                if t > 1.0 and z < 0.09:
                    out_f.write(f"FELL at t={t:.2f}s: x={x:.3f}m, z={z:.3f}m\n")
                    break
                    
            out_f.write(f"Result: success={success}, max_sink={max_sink*1000:.1f}mm, max_net_roll={np.degrees(max_roll):.1f}°\n")
    finally:
        if os.path.exists(scene_path):
            os.remove(scene_path)

if __name__ == "__main__":
    test_bridge_run()
