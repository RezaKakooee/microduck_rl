"""Test Brother HoldStraight stiffness with higher gain and limit."""
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
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK

class StrongHoldStraight:
    GAIN = 10.0 # Fast integral response
    LIMIT = 1.5  # Full joint authority

    def __init__(self, duck, dt=0.02):
        self.duck, self.dt = duck, dt
        self.extra = np.zeros(len(SERVOS))

    def __call__(self):
        err = PLANK - self.duck.q
        self.extra = np.clip(self.extra + self.GAIN * err * self.dt, -self.LIMIT, self.LIMIT)
        self.duck.hold(PLANK + self.extra)

p_orig = np.load('local_storage/gaits/crawl_gait.npy').copy()
J = {name: i for i, name in enumerate(SERVOS)}
rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'neck_pitch': 0.241, 'head_pitch': -0.135}.items():
    c_base[J[name]] = val

w = World(L.design())
he, she = w.ducks['he'], w.ducks['she']
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
hold_he = StrongHoldStraight(he)

# Place Sister at x = 0.00
she.place((0.00, 0.0, L.near_z + 0.03), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)
w.start()

# Settle
for _ in range(100):
    hold_he()
    she.hold(c_base)
    w.step()

x_start = she.pos()[0]
print(f"Settled start: x={x_start:.3f}, he_up={he.up():.3f}")

f_val = 2.09
for i in range(400): # 8 seconds
    t = i * 0.02
    hold_he()
    q = c_base.copy()
    
    for k, name in enumerate(('hip_pitch', 'knee', 'hip_roll', 'ankle')):
        off, amp, ph = p_orig[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * f_val * t + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)
        
    q[J['left_hip_roll']] = -0.20
    q[J['right_hip_roll']] = 0.20
    q[J['neck_pitch']] = -0.50
    q[J['head_pitch']] = -0.50
    
    she.hold(q)
    w.step()
    
    if i % 50 == 0:
        pos = she.pos()
        print(f"t={t:.2f}s pos={np.round(pos, 3)}, he_up={he.up():.3f}")

x_end = she.pos()[0]
print(f"Final pos: {np.round(she.pos(), 3)}, total dx = {(x_end - x_start)*1000:+.1f} mm, he_up={he.up():.3f}")
