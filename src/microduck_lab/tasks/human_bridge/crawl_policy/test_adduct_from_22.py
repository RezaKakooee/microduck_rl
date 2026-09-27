"""Test test_adduct.py starting from x = -0.22m (exact START_X)."""
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

# Start at x = -0.22 (same as Story)
she.place((-0.22, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)
w.start()

# Settle brother and sister
for _ in range(150):
    hold_he()
    she.hold(c_base)
    w.step()

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
t_start = w.t
d, m = w.data, w.model

print("Testing crawl from x=-0.22m with adducted hips:")
for step in range(350): # 7 seconds
    t = w.t - t_start
    hold_he()
    q = c_base.copy()
    for k, name in enumerate(('hip_pitch', 'knee', 'ankle')):
        idx = 0 if name == 'hip_pitch' else (1 if name == 'knee' else 3)
        off, amp, ph = p[3 * idx : 3 * idx + 3]
        wave = amp * np.sin(2.0 * np.pi * f * t + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)
    
    # Adducted hips (constant, no positive feedback):
    q[J['left_hip_roll']] = -0.30
    q[J['right_hip_roll']] = 0.30
    
    # Head lifted:
    q[J['neck_pitch']] = -0.50
    q[J['head_pitch']] = -0.50
    
    she.hold(q)
    w.step()

    if step % 25 == 0:
        pos = she.pos()
        r_roll = abs(float(she.data.xmat[she.trunk, 7]))
        print(f"t={t:.2f}s trunk=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) roll={r_roll:.3f}")

print(f"Final trunk pos: {np.round(she.pos(), 3)}")
