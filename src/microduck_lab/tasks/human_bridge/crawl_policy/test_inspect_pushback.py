"""Inspect exact contact forces (Fx, Fy, Fz in world frame) and joint states during pushback."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

story = Story(verbose=False)
she = story.she
he = story.he
m, d = story.world.model, story.world.data
J = {name: i for i, name in enumerate(SERVOS)}

# Fast-forward to t=6.5s
while story.world.t < 6.8:
    story.tick()

print(f"Inspecting pushback starting at t={story.world.t:.2f}s:")
print(f"{'t':6s} {'x':7s} {'vx':7s} {'LF_x':7s} {'RF_x':7s} {'L_hp':6s} {'R_hp':6s} {'L_kn':6s} {'R_kn':6s} {'Major contacts (body, other, Fx, Fz)'}")

for step in range(80): # 1.6s, up to t=8.4s
    story.tick()
    t = story.world.t
    pos = she.pos()
    qvel_x = d.qvel[she.dof] # world linear vx of she trunk
    
    # Joint positions
    qpos = d.qpos[she.adr + 7 : she.adr + 7 + 14]
    l_hp, r_hp = qpos[J['left_hip_pitch']], qpos[J['right_hip_pitch']]
    l_kn, r_kn = qpos[J['left_knee']], qpos[J['right_knee']]
    
    foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
    foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')
    lf_x = d.xpos[foot_l, 0]
    rf_x = d.xpos[foot_r, 0]
    
    # Calculate contact forces on Sister in world frame
    major_con = []
    total_fx = 0.0
    for c_i in range(d.ncon):
        c = d.contact[c_i]
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
        if b1.startswith('she_') or b2.startswith('she_'):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c_i, f6)
            # Contact frame: f6[0] is normal force along contact normal (c.frame[:3])
            # In world coordinates, force on body 2 is R_contact * f6[:3], on body 1 is -R_contact * f6[:3]
            # c.frame is a 3x3 rotation matrix flattened (row-major: normal, tan1, tan2)
            c_R = c.frame.reshape(3, 3)
            # Force vector in world coordinates acting on geom2:
            f_world_on_2 = c_R.T @ f6[:3]
            f_world_on_she = -f_world_on_2 if b1.startswith('she_') else f_world_on_2
            total_fx += f_world_on_she[0]
            if abs(f_world_on_she[0]) > 0.3 or abs(f_world_on_she[2]) > 1.0:
                she_b = b1 if b1.startswith('she_') else b2
                other_b = b2 if b1.startswith('she_') else b1
                major_con.append(f"{she_b[:12]}<->{other_b[:12]}(Fx={f_world_on_she[0]:+.2f}N,Fz={f_world_on_she[2]:+.2f}N,x={c.pos[0]:.2f})")
    
    if step % 5 == 0:
        print(f"{t:6.2f} {pos[0]:+6.3f} {qvel_x:+6.3f} {lf_x:+6.3f} {rf_x:+6.3f} {l_hp:+5.2f} {r_hp:+5.2f} {l_kn:+5.2f} {r_kn:+5.2f} TotFx={total_fx:+5.2f}N | {', '.join(major_con)}")
