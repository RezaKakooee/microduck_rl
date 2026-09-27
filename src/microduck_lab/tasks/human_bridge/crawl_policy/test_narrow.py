"""Test crawl gait with narrow legs centered on Brother."""
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

she.place((-0.15, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)
w.start()

# Settle
for _ in range(100):
    hold_he()
    she.hold(c_base)
    w.step()

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
t_start = w.t
d, m = w.data, w.model
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# KEEP LEGS NARROW: hip_roll off=0.05, amp=0.0
p[6] = 0.05  # small offset
p[7] = 0.00  # ZERO amplitude so feet stay centered on brother!

print("Starting narrow-leg crawl from x=-0.15...")
for step in range(350): # 7 seconds
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

    if step % 20 == 0:
        pos = she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        con_he = 0
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
            if (b1.startswith('she_') and b2.startswith('he_')) or (b2.startswith('she_') and b1.startswith('he_')):
                con_he += 1

        print(f"t={t:.2f}s trunk_x={pos[0]:.3f} z={pos[2]:.3f} | LF=({lf[0]:.3f},{lf[1]:.3f},{lf[2]:.3f}) RF=({rf[0]:.3f},{rf[1]:.3f},{rf[2]:.3f}) | con_brother={con_he}")

print(f"Final trunk pos: {np.round(she.pos(), 3)}")
