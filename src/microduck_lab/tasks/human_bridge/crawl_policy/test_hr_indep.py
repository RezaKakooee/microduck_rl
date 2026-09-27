"""Test left and right hip roll axes independently."""
import sys
from pathlib import Path
import numpy as np
import mujoco

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
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

c = np.zeros(len(SERVOS))
c[J['left_hip_pitch']] = -1.2
c[J['right_hip_pitch']] = 1.2
c[J['left_knee']] = -0.6
c[J['right_knee']] = 0.6

print("Left leg hip roll:")
for hr in [-0.3, 0.0, 0.3]:
    c_test = c.copy()
    c_test[J['left_hip_roll']] = hr
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = c_test
    mujoco.mj_forward(m, d)
    print(f"  left_hr={hr:+.2f} -> LF world Y = {d.xpos[foot_l, 1]*1000:+.1f} mm")

print("Right leg hip roll:")
for hr in [-0.3, 0.0, 0.3]:
    c_test = c.copy()
    c_test[J['right_hip_roll']] = hr
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = c_test
    mujoco.mj_forward(m, d)
    print(f"  right_hr={hr:+.2f} -> RF world Y = {d.xpos[foot_r, 1]*1000:+.1f} mm")
