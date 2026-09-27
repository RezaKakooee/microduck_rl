"""Check jaw height in world coordinates for neck/head pitch."""
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
jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_mouth_jaw')
head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_jaw_soft')

rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c = np.zeros(len(SERVOS))
she.place((-0.15, 0.0, L.near_z + 0.05), q=c)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz

print(f"Ledge height near_z = {L.near_z:.3f} m")
for np_val in [-0.5, -0.2, 0.0, 0.2, 0.5]:
    for hp_val in [-0.5, 0.0, 0.5]:
        c_test = c.copy()
        c_test[J['neck_pitch']] = np_val
        c_test[J['head_pitch']] = hp_val
        d.qpos[she.adr + 7 : she.adr + 7 + 14] = c_test
        mujoco.mj_forward(m, d)
        z_jaw = d.xpos[jaw, 2]
        z_head = d.xpos[head, 2]
        print(f"neck={np_val:+.2f} head={hp_val:+.2f} -> jaw_z={z_jaw:.3f} head_z={z_head:.3f}")
