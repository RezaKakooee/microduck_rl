import os
import sys
import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.sim.duck_sim import CONTROL_DT, DECIMATION

WALKING_ONNX = os.path.join(os.path.dirname(duck_sim.REPO), "microduck", "policies", "alpha_walking.onnx")
TEMPLATE = paths.model("scene_beam.xml")

def build_loose_net_xml(span=1.6, width=0.38, num_rungs=20, height=0.18, center_kz=180.0, center_kr=2.0, center_kp=2.5):
    """Generate dynamic loose rope net XML."""
    half_w = width / 2.0
    start_x = 0.0
    end_x = start_x + span
    dx = span / (num_rungs - 1)
    
    wood_col = "0.40 0.30 0.20 1"
    plank_col = "0.50 0.38 0.24 1"
    rope_col = "0.72 0.58 0.38 1"       # hemp rope
    knot_col = "0.60 0.45 0.28 1"       # rope knot
    cable_col = "0.45 0.36 0.25 1"      # thick suspension rope
    
    parts = []
    # Approach & Exit platforms
    landing_len = 0.40
    # Landing entry ends exactly at x = 0.0 (from -landing_len to 0.0)
    entry_cx = -landing_len / 2.0
    parts.append(f'<geom name="landing_entry" type="box" pos="{entry_cx:.4f} 0 {height-0.01:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} 0.01" rgba="{plank_col}" friction="1.5 0.01 0.001"/>')
    parts.append(f'<geom name="landing_entry_base" type="box" pos="{entry_cx:.4f} 0 {(height-0.02)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.02:.4f} {(height-0.02)/2:.4f}" rgba="{wood_col}"/>')
    
    # Landing exit starts at span (from span to span + landing_len)
    exit_cx = end_x + landing_len / 2.0
    parts.append(f'<geom name="landing_exit" type="box" pos="{exit_cx:.4f} 0 {height-0.01:.4f}" size="{landing_len/2:.4f} {half_w+0.03:.4f} 0.01" rgba="{plank_col}" friction="1.5 0.01 0.001"/>')
    parts.append(f'<geom name="landing_exit_base" type="box" pos="{exit_cx:.4f} 0 {(height-0.02)/2:.4f}" size="{landing_len/2:.4f} {half_w+0.02:.4f} {(height-0.02)/2:.4f}" rgba="{wood_col}"/>')
    
    # Towers & Cables
    tower_h = height + 0.22
    for x in (start_x, end_x):
        for s in (-1, 1):
            parts.append(f'<geom name="tower_{x:.2f}_{s}" type="capsule" fromto="{x:.4f} {s*half_w:.4f} 0 {x:.4f} {s*half_w:.4f} {tower_h:.4f}" size="0.016" rgba="{wood_col}"/>')
        parts.append(f'<geom name="tower_top_{x:.2f}" type="capsule" fromto="{x:.4f} {-half_w:.4f} {tower_h:.4f} {x:.4f} {half_w:.4f} {tower_h:.4f}" size="0.012" rgba="{wood_col}"/>')
    
    # Catenary top cables
    for s in (-1, 1):
        for i in range(24):
            u0, u1 = i/24.0, (i+1)/24.0
            x0 = start_x + u0 * span
            x1 = start_x + u1 * span
            z0 = tower_h - 0.01 - 0.10 * 4.0 * u0 * (1.0 - u0)
            z1 = tower_h - 0.01 - 0.10 * 4.0 * u1 * (1.0 - u1)
            parts.append(f'<geom name="cable_{s}_{i}" type="capsule" fromto="{x0:.4f} {s*half_w:.4f} {z0:.4f} {x1:.4f} {s*half_w:.4f} {z1:.4f}" size="0.005" rgba="{cable_col}"/>')

    fixed_xml = '<body name="bridge_fixed">\n' + '\n'.join(parts) + '\n</body>\n'

    # Dynamic Rungs with flexible surface
    # Rung 0 is at x = dx/2, last rung is at span - dx/2
    # This leaves clean clearance to the landing platforms!
    rungs_xml = []
    rung_xs = np.linspace(dx * 0.5, span - dx * 0.5, num_rungs)
    tread_len = dx * 0.98  # no overlap between neighboring rungs!
    rung_mass = 0.06       # light rope
    
    for i, x in enumerate(rung_xs):
        u = x / span
        sag = 0.035 * 4.0 * u * (1.0 - u) # initial static sag
        z = height - sag
        
        # Soft compliance at center
        parabola = 4.0 * u * (1.0 - u)
        kz = center_kz + (1.0 - parabola) * 350.0
        kr = center_kr + (1.0 - parabola) * 3.5
        kp = center_kp + (1.0 - parabola) * 4.5
        
        dz = 2.0 * 1.6 * np.sqrt(kz * (rung_mass + 0.04))
        dr = 2.0 * 1.6 * np.sqrt(kr * 0.01)
        dp = 2.0 * 1.6 * np.sqrt(kp * 0.01)
        
        rope_contact = 'friction="1.6 0.01 0.001" solref="0.015 1.0" solimp="0.85 0.98 0.01 0.5 2"'
        
        # Cross rope
        r_str = f'<geom name="cross_rope_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {half_w:.4f} 0" size="0.007" mass="{rung_mass*0.35:.4f}" rgba="{rope_col}" {rope_contact}/>'
        
        # Continuous woven rope tread (non-overlapping)
        tread = f'<geom name="mesh_tread_{i}" type="box" pos="0 0 -0.004" size="{tread_len/2:.4f} {half_w:.4f} 0.004" mass="{rung_mass*0.35:.4f}" rgba="{rope_col}" {rope_contact}/>'
        
        # Knots along the cross rope
        knots = []
        for y_k in np.linspace(-half_w + 0.03, half_w - 0.03, 7):
            knots.append(f'<geom name="knot_{i}_{y_k:+.2f}" type="sphere" pos="0 {y_k:.4f} 0.002" size="0.009" mass="{rung_mass*0.04:.4f}" rgba="{knot_col}" {rope_contact}/>')
            
        # Hanger ropes to top cable
        cable_z = tower_h - 0.01 - 0.10 * 4.0 * u * (1.0 - u)
        h_L = f'<geom name="h_L_{i}" type="capsule" fromto="0 {half_w:.4f} 0 0 {half_w:.4f} {cable_z - z:.4f}" size="0.003" rgba="{rope_col}"/>'
        h_R = f'<geom name="h_R_{i}" type="capsule" fromto="0 {-half_w:.4f} 0 0 {-half_w:.4f} {cable_z - z:.4f}" size="0.003" rgba="{rope_col}"/>'

        rung_b = f'''
        <body name="rung_{i}" pos="{x:.4f} 0 {z:.4f}">
            <joint name="j_z_{i}" type="slide" axis="0 0 1" stiffness="{kz:.1f}" damping="{dz:.2f}" range="-0.12 0.04"/>
            <joint name="j_pitch_{i}" type="hinge" axis="0 1 0" stiffness="{kp:.1f}" damping="{dp:.2f}" range="-0.30 0.30"/>
            <joint name="j_roll_{i}" type="hinge" axis="1 0 0" stiffness="{kr:.1f}" damping="{dr:.2f}" range="-0.25 0.25"/>
            {r_str}
            {tread}
            {h_L}
            {h_R}
            {' '.join(knots)}
        </body>
        '''
        rungs_xml.append(rung_b)
        
    return fixed_xml + '\n'.join(rungs_xml)

def test_walk():
    xml_bridge = build_loose_net_xml()
    with open(TEMPLATE) as f:
        tmpl = f.read()
    needle = '<geom name="beam" type="box" size="1.0000 0.0500 0.0200" pos="0.7000 0 0.0200" rgba="0.85 0.65 0.35 1" />'
    scene_str = tmpl.replace(needle, xml_bridge)
    
    test_scene = paths.model("scene_test_bridge.xml")
    with open(test_scene, "w") as f:
        f.write(scene_str)
        
    try:
        model, data = duck_sim.load_scene(str(test_scene))
        policy, adr = duck_sim.make_policy(model, data, walking_onnx_path=WALKING_ONNX)
        
        # Init duck on landing platform
        data.qpos[adr + 0] = -0.15
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = 0.18 + 0.125
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        
        speed = 0.22
        k_lat, k_yaw = 2.2, 1.2
        
        print("Starting rollout...")
        for step in range(350): # 7.0 seconds
            t = step * CONTROL_DT
            x = float(data.qpos[adr])
            y = float(data.qpos[adr + 1])
            z = float(data.qpos[adr + 2])
            yaw = duck_sim.trunk_yaw(data, adr)
            
            if t < 1.0:
                policy.set_vel_cmd(0.0, 0.0, 0.0)
            else:
                aim = float(np.clip(-k_lat * y, -0.35, 0.35))
                turn = float(np.clip(k_yaw * (aim - yaw), -0.8, 0.8))
                policy.set_vel_cmd(speed, 0.0, turn)
                
            policy.apply_action(policy.infer())
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
                
            if step % 25 == 0:
                sinks = [-data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_z_{r}")]] for r in range(20)]
                max_s = max(sinks)
                active_r = int(np.argmax(sinks))
                print(f"t={t:4.2f}s | x={x:+.3f}m y={y*1000:+4.0f}mm z={z:+.3f}m | max_sink={max_s*1000:4.1f}mm (rung {active_r:02d})")
                
            if t > 1.0 and z < 0.08:
                print(f"Duck fell at t={t:.2f}s (x={x:.3f}m, z={z:.3f}m)")
                return False
                
        print(f"Completed! Final x = {x:.3f}m")
        return True
    finally:
        if os.path.exists(test_scene):
            os.remove(test_scene)

if __name__ == "__main__":
    test_walk()
