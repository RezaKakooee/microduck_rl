"""Map foot Y position in world frame as a function of hip_roll and knee."""
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
mujoco.mj_forward(m, d)

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

print("Mapping hip_roll values to foot Y position:")
for hr_l in [-0.38, -0.30, -0.20, -0.10, 0.00, +0.10, +0.20, +0.30, +0.38]:
    q = c_base.copy()
    q[J['left_hip_roll']] = hr_l
    q[J['right_hip_roll']] = -hr_l # symmetrical
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = q
    mujoco.mj_forward(m, d)
    lf_y = d.xpos[foot_l, 1]
    rf_y = d.xpos[foot_r, 1]
    span = abs(rf_y - lf_y)
    print(f"left_hr={hr_l:+.2f} right_hr={-hr_l:+.2f} -> LF_y={lf_y:+.4f} RF_y={rf_y:+.4f} span={span*1000:.1f}mm")
