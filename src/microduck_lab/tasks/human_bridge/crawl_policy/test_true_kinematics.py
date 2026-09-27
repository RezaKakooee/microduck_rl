"""Measure exact foot dx, dy, dz for each joint using true she.qidx mapping."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

s = Story(verbose=False)
m, d = s.world.model, s.world.data
J = {name: i for i, name in enumerate(SERVOS)}

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# Base qpos
q0 = s.she.data.qpos[s.she.qidx].copy()
q0[J['left_hip_pitch']] = -0.90 # forward crawl pose
q0[J['right_hip_pitch']] = 0.90
q0[J['left_hip_roll']] = -0.30
q0[J['right_hip_roll']] = 0.30
d.qpos[s.she.qidx] = q0
mujoco.mj_forward(m, d)

pos_l0 = d.xpos[foot_l].copy()
pos_r0 = d.xpos[foot_r].copy()

print(f"Base foot positions (world): LF = {np.round(pos_l0, 4)}, RF = {np.round(pos_r0, 4)}")
print(f"Trunk position (world): {np.round(d.xpos[s.she.trunk], 4)}")
print("\nEffect of +0.20 rad change on each joint:")

for j_name in SERVOS:
    q = q0.copy()
    q[J[j_name]] += 0.20
    d.qpos[s.she.qidx] = q
    mujoco.mj_forward(m, d)
    
    pos_l = d.xpos[foot_l].copy()
    pos_r = d.xpos[foot_r].copy()
    
    dl = (pos_l - pos_l0) * 1000 # mm
    dr = (pos_r - pos_r0) * 1000 # mm
    
    if np.linalg.norm(dl) > 0.1:
        print(f"  {j_name:16s} -> LF dx={dl[0]:+6.1f}mm dy={dl[1]:+6.1f}mm dz={dl[2]:+6.1f}mm")
    if np.linalg.norm(dr) > 0.1:
        print(f"  {j_name:16s} -> RF dx={dr[0]:+6.1f}mm dy={dr[1]:+6.1f}mm dz={dr[2]:+6.1f}mm")
