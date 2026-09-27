"""Inspect why Sister stalls at x=+0.015m on Brother's back in test_adduct.py."""
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
for _ in range(150):
    hold_he()
    she.hold(c_base)
    w.step()

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
f = p[-3]
t_start = w.t
d, m = w.data, w.model
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

print(f"Traverse inspection from t=2.0s to 7.0s:")
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
    
    q[J['left_hip_roll']] = -0.30
    q[J['right_hip_roll']] = 0.30
    q[J['neck_pitch']] = -0.50
    q[J['head_pitch']] = -0.50
    
    she.hold(q)
    w.step()

    if step >= 100 and step % 25 == 0:
        pos = she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        # Check contacts between sister and brother
        con_details = []
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
            if (b1.startswith('she_') and b2.startswith('he_')) or (b2.startswith('she_') and b1.startswith('he_')):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, c_i, f6)
                she_b = b1 if b1.startswith('she_') else b2
                he_b = b2 if b1.startswith('she_') else b1
                con_details.append(f"{she_b[:12]}<->{he_b[:12]}(fn={f6[0]:.1f}N,x={c.pos[0]:.2f},y={c.pos[1]:.2f})")
        print(f"t={t:.2f}s pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) LF=({lf[0]:.3f},{lf[1]:.3f},{lf[2]:.3f}) RF=({rf[0]:.3f},{rf[1]:.3f},{rf[2]:.3f})")
        print(f"   Contacts on Brother ({len(con_details)}): {', '.join(con_details)}")
