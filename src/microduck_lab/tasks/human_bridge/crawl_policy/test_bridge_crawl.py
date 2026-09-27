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

# Place brother on near step in PLANK and engage HoldStraight
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
hold_he = HoldStraight(he)

# Sister crawl orientation (belly on ground, facing across gap in +X):
rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
J = {name: i for i, name in enumerate(SERVOS)}
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'neck_pitch': -0.40, 'head_pitch': -0.30}.items():
    c_base[J[name]] = val

she.place((-0.20, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)

w.start()

# Let brother settle
print("Brother settling...")
for _ in range(150):
    hold_he()
    she.hold(c_base)
    w.step()

print(f"Brother settled: he_up = {he.up():.3f}, sister pos = {np.round(she.pos(), 3)}")

p = np.load("local_storage/gaits/crawl_gait.npy")
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
t_period = p[-2] + p[-1]
tuck_time = p[-2]

print("Sister crawling with crawl_baby_gait...")
t_start = w.t
for i in range(150): # 3 seconds
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
        # Check contact force on sister's feet/legs
        d = w.data
        contact_names = []
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            g1 = mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            f6 = np.zeros(6)
            mujoco.mj_contactForce(w.model, d, c_i, f6)
            if f6[0] > 0.5:
                contact_names.append(f"{g1}<->{g2}({f6[0]:.1f}N)")
        print(f"t={w.t:.2f}s x={she.pos()[0]:.3f} z={she.pos()[2]:.3f} contacts: {', '.join(contact_names[:4])}")

print(f"Final t={w.t:.2f}s pos={np.round(she.pos(), 3)}")
