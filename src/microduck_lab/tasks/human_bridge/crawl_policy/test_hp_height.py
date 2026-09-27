"""Check hip_pitch sign for lifting foot higher in Z_world."""
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
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')

rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c = np.zeros(len(SERVOS))
c[J['left_knee']] = -0.60

she.place((-0.10, 0.0, L.near_z + 0.05), q=c)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz

for hp in [-1.5, -1.0, -0.5, 0.0, 0.5]:
    c_test = c.copy()
    c_test[J['left_hip_pitch']] = hp
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = c_test
    mujoco.mj_forward(m, d)
    print(f"left_hip_pitch={hp:+.2f} -> LF world pos: x={d.xpos[foot_l, 0]:+.3f}, z={d.xpos[foot_l, 2]:.3f}")
