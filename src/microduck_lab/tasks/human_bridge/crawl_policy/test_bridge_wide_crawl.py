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
he, she = w.ducks["he"], w.ducks["she"]

# Place brother on near step in PLANK
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
hold_he = HoldStraight(he)

# Sister crawl orientation (belly on ground, moving across gap in +X):
rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
J = {name: i for i, name in enumerate(SERVOS)}
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'left_hip_roll': 0.50, 'right_hip_roll': -0.50,
                  'neck_pitch': 0.241, 'head_pitch': -0.135}.items():
    c_base[J[name]] = val

she.place((-0.20, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)

w.start()

# Let brother settle
for _ in range(150):
    hold_he()
    she.hold(c_base)
    w.step()

print(f"Settled: he_up = {he.up():.3f}, sister pos = {np.round(she.pos(), 3)}")

p = np.load("local_storage/gaits/crawl_gait.npy").copy()
# Adjust hip_roll offset so feet walk on the side ledges clear of the notch
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
p[3 * 2] = 0.50  # hip_roll offset = 0.50 rad -> foot y = +/- 0.090 m

f = p[-3]
t_start = w.t
print("Sister starting wide crawl...")

for i in range(400): # 8 seconds
    t = w.t - t_start
    hold_he()
    q = c_base.copy()
    for k, name in enumerate(JOINTS):
        off, amp, ph = p[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * f * t + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)
    q[J['neck_pitch']] = p[-2]
    q[J['head_pitch']] = p[-1]

    she.hold(q)
    w.step()

    if i % 25 == 0:
        pos = np.round(she.pos(), 3)
        print(f"t={w.t:.2f}s pos={pos} he_up={he.up():.3f}")

print(f"Final t={w.t:.2f}s pos={np.round(she.pos(), 3)} he_up={he.up():.3f}")
