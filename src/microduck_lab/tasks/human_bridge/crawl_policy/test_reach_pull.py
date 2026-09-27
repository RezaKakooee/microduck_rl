"""Test extending legs onto Brother and pulling across the notch."""
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
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK, HoldStraight

w = World(L.design())
he, she = w.ducks['he'], w.ducks['she']
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
hold_he = HoldStraight(he)

rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
J = {name: i for i, name in enumerate(SERVOS)}
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'neck_pitch': 0.241, 'head_pitch': -0.135}.items():
    c_base[J[name]] = val

she.place((-0.075, 0.0, L.near_z + 0.04), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)
w.start()

# Settle Sister at edge
for _ in range(100):
    hold_he()
    she.hold(c_base)
    w.step()

print(f"Settled at edge: trunk_pos={np.round(she.pos(), 3)}")

d, m = w.data, w.model
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# Now extend left leg fully onto Brother:
# hp = -0.90, kn = -1.00 (straight leg), an = -1.20
for step in range(50):
    u = step / 50.0
    hold_he()
    q = c_base.copy()
    # Left leg reaches out:
    q[J['left_hip_pitch']] = -1.20 * (1-u) + (-0.90) * u
    q[J['left_knee']] = -0.60 * (1-u) + (-1.00) * u
    q[J['left_ankle']] = 0.0 * (1-u) + (-1.20) * u
    she.hold(q)
    w.step()

lf = d.xpos[foot_l]
print(f"Left leg reached: LF=({lf[0]:.3f}, {lf[1]:.3f}, {lf[2]:.3f})")

# Check contacts between left foot and brother:
con = 0
for c_i in range(d.ncon):
    c = d.contact[c_i]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
    if ('she_ankle_left' in b1 or 'she_ankle_left' in b2) and (b1.startswith('he_') or b2.startswith('he_')):
        con += 1
print(f"Contacts between left foot and Brother: {con}")
