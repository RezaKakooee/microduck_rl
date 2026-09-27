"""Test exact direction of motion (dx, dy, dz in world frame) for each joint."""
import sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

w = World(L.design())
she = w.ducks['she']
m, d = w.model, w.data
J = {name: i for i, name in enumerate(SERVOS)}

rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'neck_pitch': 0.241, 'head_pitch': -0.135}.items():
    c_base[J[name]] = val

she.place((0.0, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
d.qpos[she.adr + 7 : she.adr + 7 + 14] = c_base
mujoco.mj_forward(m, d)

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

pos_l0 = d.xpos[foot_l].copy()
pos_r0 = d.xpos[foot_r].copy()

print(f"Base foot positions: LF={np.round(pos_l0, 4)}, RF={np.round(pos_r0, 4)}")

delta = 0.20
for side, j_name, foot_id, pos_0 in [
    ('left', 'hip_yaw', foot_l, pos_l0),
    ('left', 'hip_roll', foot_l, pos_l0),
    ('left', 'hip_pitch', foot_l, pos_l0),
    ('left', 'knee', foot_l, pos_l0),
    ('left', 'ankle', foot_l, pos_l0),
    ('right', 'hip_yaw', foot_r, pos_r0),
    ('right', 'hip_roll', foot_r, pos_r0),
    ('right', 'hip_pitch', foot_r, pos_r0),
    ('right', 'knee', foot_r, pos_r0),
    ('right', 'ankle', foot_r, pos_r0),
]:
    full_name = f"{side}_{j_name}"
    q = c_base.copy()
    q[J[full_name]] += delta
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = q
    mujoco.mj_forward(m, d)
    pos_new = d.xpos[foot_id].copy()
    diff = (pos_new - pos_0) * 1000 # mm
    print(f"+0.20 rad on {full_name:16s} -> dx={diff[0]:+6.1f}mm dy={diff[1]:+6.1f}mm dz={diff[2]:+6.1f}mm")
