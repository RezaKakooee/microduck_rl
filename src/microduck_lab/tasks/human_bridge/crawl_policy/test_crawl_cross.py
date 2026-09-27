"""Test wide-leg crawl over the side ledges."""
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

# Increase hip_roll offset so feet are wider than notch (notch is +/- 0.085):
# JOINTS[2] is hip_roll: off, amp, ph = p[6:9]
p[6] = 0.65  # wider hip roll offset (was 0.276)
p[7] = 0.30  # smaller amplitude to keep feet wide

print("Starting wide-leg crawl from x=-0.15...")
for step in range(250): # 5 seconds
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

    if step % 15 == 0:
        pos = she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        con_notch = 0
        con_side = 0
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if 'near_step' in g1 or 'near_step' in g2:
                con_notch += 1
            if 'near_top_left' in g1 or 'near_top_left' in g2 or 'near_top_right' in g1 or 'near_top_right' in g2:
                con_side += 1

        print(f"t={t:.2f}s trunk_x={pos[0]:.3f} z={pos[2]:.3f} | LF=({lf[0]:.3f},{lf[1]:.3f},{lf[2]:.3f}) RF=({rf[0]:.3f},{rf[1]:.3f},{rf[2]:.3f}) | con_notch={con_notch} con_side={con_side}")

print(f"Final trunk pos: {np.round(she.pos(), 3)}")
