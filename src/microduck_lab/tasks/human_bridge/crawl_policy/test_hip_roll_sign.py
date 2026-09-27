"""Check which sign of hip_roll adducts (brings feet toward y=0) on Sister."""
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

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')

she.place((0.0, 0.0, L.near_z + 0.05), q=np.zeros(len(SERVOS)))
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# Base joint targets from crawl_gait at t=0
q_base = np.zeros(len(SERVOS))
for k, name in enumerate(JOINTS):
    off, amp, ph = p[3 * k : 3 * k + 3]
    q_base[J['left_' + name]] = off
    q_base[J['right_' + name]] = -off

print(f"Base targets: left_hp={q_base[J['left_hip_pitch']]:.2f}, left_kn={q_base[J['left_knee']]:.2f}, left_an={q_base[J['left_ankle']]:.2f}")
print(f"             right_hp={q_base[J['right_hip_pitch']]:.2f}, right_kn={q_base[J['right_knee']]:.2f}, right_an={q_base[J['right_ankle']]:.2f}")

for l_hr in [-0.35, -0.20, 0.0, +0.20, +0.35]:
    q = q_base.copy()
    q[J['left_hip_roll']] = l_hr
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = q
    mujoco.mj_forward(m, d)
    lf_y = d.xpos[foot_l, 1]
    print(f"left_hip_roll = {l_hr:+.2f} -> LF world Y = {lf_y*1000:+.1f} mm")

for r_hr in [-0.35, -0.20, 0.0, +0.20, +0.35]:
    q = q_base.copy()
    q[J['right_hip_roll']] = r_hr
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = q
    mujoco.mj_forward(m, d)
    rf_y = d.xpos[foot_r, 1]
    print(f"right_hip_roll = {r_hr:+.2f} -> RF world Y = {rf_y*1000:+.1f} mm")
