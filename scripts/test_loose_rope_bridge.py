import os
import sys
sys.path.insert(0, "/home2/reza/microduck_rl")
sys.path.insert(0, "/home2/reza/microduck_rl/scripts")
import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION

WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")
TEMPLATE = paths.model("scene_beam.xml")

def generate_loose_rope_xml(
    span=1.6,
    width=0.38,
    num_rungs=24,
    height=0.18,
    static_sag=0.035,
    center_kz=150.0,
    center_kr=1.8,
    center_kp=2.2,
    damping_ratio=0.8,
):
    """
    Generates a procedural MuJoCo XML for a soft, physically compliant
    loose-rope suspension bridge ("طناب ساده").
    """
    half_w = width / 2.0
    start_x = 0.0
    end_x = start_x + span
    dx = span / num_rungs
    landing_len = 0.40
    tower_h = height + 0.22
    
    # Natural hemp/jute rope color palette ("طناب ساده")
    wood_col = "0.38 0.28 0.18 1"
    plank_col = "0.48 0.36 0.22 1"
    rope_col = "0.72 0.58 0.38 1"       # hemp rope
    rope_dark = "0.62 0.48 0.30 1"      # twisted core
    knot_col = "0.55 0.42 0.25 1"       # rope knot
    cable_col = "0.45 0.35 0.22 1"      # thick overhead cable
    
    fixed_parts = []
    
    # 1. Approach & Exit Timber Platforms
    # Landing entry: x from -landing_len to 0
    fixed_parts.append(f'<geom name="landing_entry" type="box" pos="{-landing_len/2:.4f} 0 {height-0.01:.4f}" size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.01" rgba="{plank_col}" friction="1.6 0.01 0.001"/>')
    fixed_parts.append(f'<geom name="landing_entry_base" type="box" pos="{-landing_len/2:.4f} 0 {(height-0.02)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.02)/2:.4f}" rgba="{wood_col}"/>')
    
    # Landing exit: x from span to span + landing_len
    fixed_parts.append(f'<geom name="landing_exit" type="box" pos="{end_x+landing_len/2:.4f} 0 {height-0.01:.4f}" size="{landing_len/2:.4f} {half_w+0.04:.4f} 0.01" rgba="{plank_col}" friction="1.6 0.01 0.001"/>')
    fixed_parts.append(f'<geom name="landing_exit_base" type="box" pos="{end_x+landing_len/2:.4f} 0 {(height-0.02)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} {(height-0.02)/2:.4f}" rgba="{wood_col}"/>')
    
    # 2. Suspension Towers (Pylons)
    for x in (start_x, end_x):
        for s in (-1, 1):
            fixed_parts.append(f'<geom name="pylon_{x:.2f}_{s}" type="capsule" fromto="{x:.4f} {s*half_w:.4f} 0 {x:.4f} {s*half_w:.4f} {tower_h:.4f}" size="0.016" rgba="{wood_col}"/>')
        fixed_parts.append(f'<geom name="pylon_cross_{x:.2f}" type="capsule" fromto="{x:.4f} {-half_w:.4f} {tower_h:.4f} {x:.4f} {half_w:.4f} {tower_h:.4f}" size="0.012" rgba="{wood_col}"/>')

    # 3. Main Catenary Suspension Cables (Overhead)
    c_steps = 28
    for s in (-1, 1):
        for i in range(c_steps):
            u0, u1 = i / c_steps, (i + 1) / c_steps
            x0 = start_x + u0 * span
            x1 = start_x + u1 * span
            z0 = tower_h - 0.01 - 0.11 * 4.0 * u0 * (1.0 - u0)
            z1 = tower_h - 0.01 - 0.11 * 4.0 * u1 * (1.0 - u1)
            fixed_parts.append(f'<geom name="cable_{s}_{i}" type="capsule" fromto="{x0:.4f} {s*half_w:.4f} {z0:.4f} {x1:.4f} {s*half_w:.4f} {z1:.4f}" size="0.006" rgba="{cable_col}"/>')

    fixed_xml = '<body name="bridge_anchors">\n' + '\n'.join(fixed_parts) + '\n</body>\n'

    # 4. Articulated Loose-Rope Net Walkway (Multi-body dynamic compliance)
    rung_xml = []
    rung_mass = 0.05  # light rope mass per segment (~50g)
    
    # Soft rope contact properties
    rope_contact = 'friction="1.8 0.01 0.001" solref="0.015 1.0" solimp="0.85 0.98 0.01 0.5 2"'
    
    # We place rungs from x = dx*0.5 to span - dx*0.5
    for i in range(num_rungs):
        x = start_x + (i + 0.5) * dx
        u = x / span
        
        # Initial catenary sag of the loose net
        sag = static_sag * 4.0 * u * (1.0 - u)
        z0 = height - sag
        
        # Parabolic stiffness distribution: softest at center, stiffer near anchors
        parabola = 4.0 * u * (1.0 - u)
        kz = center_kz + (1.0 - parabola) * 350.0
        kr = center_kr + (1.0 - parabola) * 3.5
        kp = center_kp + (1.0 - parabola) * 4.0
        
        # Damping calculated from damping_ratio
        dz = 2.0 * damping_ratio * np.sqrt(kz * (rung_mass + 0.04))
        dr = 2.0 * damping_ratio * np.sqrt(kr * 0.01)
        dp = 2.0 * damping_ratio * np.sqrt(kp * 0.01)
        
        # Hanger ropes to overhead cable
        cable_z = tower_h - 0.01 - 0.11 * 4.0 * u * (1.0 - u)
        h_L = f'<geom name="h_L_{i}" type="capsule" fromto="0 {half_w:.4f} 0 0 {half_w:.4f} {cable_z - z0:.4f}" size="0.003" rgba="{rope_dark}"/>'
        h_R = f'<geom name="h_R_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {-half_w:.4f} {cable_z - z0:.4f}" size="0.003" rgba="{rope_dark}"/>'

        # Transverse rope (cross-rung)
        cross_rope = f'<geom name="cross_rope_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {half_w:.4f} 0" size="0.007" mass="{rung_mass*0.3:.4f}" rgba="{rope_col}" {rope_contact}/>'
        
        # Flexible woven net segment (tread): covers dx span so foot never falls into empty void
        tread = f'<geom name="mesh_net_{i}" type="box" pos="0 0 -0.004" size="{dx*0.49:.4f} {half_w:.4f} 0.004" mass="{rung_mass*0.4:.4f}" rgba="{rope_col}" {rope_contact}/>'
        
        # Knots & longitudinal rope strands along the net
        knots = []
        num_strands = 7
        for y_str in np.linspace(-half_w + 0.03, half_w - 0.03, num_strands):
            # Rope knot sphere
            knots.append(f'<geom name="knot_{i}_{y_str:+.2f}" type="sphere" pos="0 {y_str:.4f} 0.002" size="0.009" mass="{rung_mass*0.03:.4f}" rgba="{knot_col}" {rope_contact}/>')
            # Longitudinal rope rib
            knots.append(f'<geom name="rib_{i}_{y_str:+.2f}" type="capsule" fromto="{-dx*0.48:.4f} {y_str:.4f} 0 {dx*0.48:.4f} {y_str:.4f} 0" size="0.004" rgba="{rope_dark}" {rope_contact}/>')

        b_xml = f'''
        <body name="net_seg_{i}" pos="{x:.4f} 0 {z0:.4f}">
            <joint name="j_z_{i}" type="slide" axis="0 0 1" stiffness="{kz:.1f}" damping="{dz:.2f}" range="-0.14 0.04"/>
            <joint name="j_pitch_{i}" type="hinge" axis="0 1 0" stiffness="{kp:.1f}" damping="{dp:.2f}" range="-0.35 0.35"/>
            <joint name="j_roll_{i}" type="hinge" axis="1 0 0" stiffness="{kr:.1f}" damping="{dr:.2f}" range="-0.30 0.30"/>
            {cross_rope}
            {tread}
            {h_L}
            {h_R}
            {' '.join(knots)}
        </body>
        '''
        rung_xml.append(b_xml)
        
    return fixed_xml + '\n'.join(rung_xml)

def run_test(speed=0.22, kz=160.0):
    xml_str = generate_loose_rope_xml(center_kz=kz)
    with open(TEMPLATE) as f:
        tmpl = f.read()
    needle = '<geom name="beam" type="box" size="1.0000 0.0500 0.0200" pos="0.7000 0 0.0200" rgba="0.85 0.65 0.35 1" />'
    scene_str = tmpl.replace(needle, xml_str)
    
    test_scene = paths.model("scene_test_loose.xml")
    with open(test_scene, "w") as f:
        f.write(scene_str)
        
    try:
        model, data = duck_sim.load_scene(str(test_scene))
        policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=WALKING_ONNX)
        
        # Start duck at x = 0.05 (already on the bridge net entrance!)
        # Or at x = -0.10 on the landing platform
        data.qpos[adr + 0] = -0.10
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = 0.18 + 0.115
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        
        seconds = 14.0
        steps = int(round(seconds / CONTROL_DT))
        k_lat, k_yaw = 4.0, 2.0
        max_aim = 0.20
        max_turn = 0.6
        
        max_sink = 0.0
        max_roll = 0.0
        max_pitch = 0.0
        success = False
        
        for step in range(steps):
            t = step * CONTROL_DT
            x = float(data.qpos[adr])
            y = float(data.qpos[adr + 1])
            z = float(data.qpos[adr + 2])
            yaw = duck_sim.trunk_yaw(data, adr)
            
            # Measure net deformation
            for r in range(24):
                q_z = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_z_{r}")]
                sink = -data.qpos[q_z]
                if sink > max_sink:
                    max_sink = sink
                q_r = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_roll_{r}")]
                max_roll = max(max_roll, abs(data.qpos[q_r]))
                q_p = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_pitch_{r}")]
                max_pitch = max(max_pitch, abs(data.qpos[q_p]))
                
            if t < 0.8:
                policy.set_vel_cmd(0.0, 0.0, 0.0)
            elif t < 1.5:
                # Gentle start straight ahead
                policy.set_vel_cmd(speed, 0.0, 0.0)
            else:
                aim = float(np.clip(-k_lat * y, -max_aim, max_aim))
                turn = float(np.clip(k_yaw * (aim - yaw), -max_turn, max_turn))
                policy.set_vel_cmd(speed, 0.0, turn)
                
            policy.apply_action(policy.infer())
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
                
            if step % 25 == 0:
                print(f"t={t:4.1f}s | x={x:+.3f}m y={y*1000:+4.0f}mm z={z:+.3f}m | sink={max_sink*1000:4.1f}mm roll={np.degrees(max_roll):4.1f}° pitch={np.degrees(max_pitch):4.1f}°")
                
            if x >= 1.50 and not success:
                print(f"SUCCESS! Crossed bridge in {t:.1f}s!")
                success = True
                break
                
            if t > 1.5 and z < 0.09:
                print(f"FELL at t={t:.2f}s: x={x:.3f}m, z={z:.3f}m")
                break
                
        return success, max_sink, max_roll
    finally:
        if os.path.exists(test_scene):
            os.remove(test_scene)

if __name__ == "__main__":
    run_test()
