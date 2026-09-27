"""Check ankle angle to plant rubber soles onto Brother's back."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK, HoldStraight

w = World(L.design())
he, she = w.ducks['he'], w.ducks['she']
m, d = w.model, w.data
J = {name: i for i, name in enumerate(SERVOS)}
sole_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'she_left_foot_collision')
sole_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'she_right_foot_collision')

from scipy.spatial.transform import Rotation as R
rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c = np.zeros(len(SERVOS))
c[J['left_hip_pitch']] = -1.20
c[J['right_hip_pitch']] = 1.20
c[J['left_knee']] = 0.50
c[J['right_knee']] = -0.50
c[J['left_hip_roll']] = -0.25
c[J['right_hip_roll']] = 0.25

she.place((0.02, 0.0, L.near_z + 0.03), q=c)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz

for an in [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2]:
    c_test = c.copy()
    c_test[J['left_ankle']] = an
    c_test[J['right_ankle']] = -an
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = c_test
    mujoco.mj_forward(m, d)
    z_sole_l = d.geom_xpos[sole_l, 2]
    z_knee_l = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_leg'), 2]
    print(f"ankle={an:+.2f} -> sole_z={z_sole_l:.3f}, knee_z={z_knee_l:.3f} | diff(sole - knee)={(z_sole_l - z_knee_l)*1000:+.1f}mm")
