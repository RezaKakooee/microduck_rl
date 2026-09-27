"""Test crawl with lifted head and adducted hips to see if Sister passes the notch."""
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
                  'neck_pitch': 0.35, 'head_pitch': -0.15}.items():
    c_base[J[name]] = val

she.place((-0.22, 0.0, L.near_z + 0.05), q=c_base)
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

jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_mouth_jaw')
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

print("Testing crawl with neck_pitch=+0.35, head_pitch=-0.15:")
furthest_x = -0.22
for step in range(400): # 8 seconds
    t = w.t - t_start
    hold_he()
    q = c_base.copy()
    
    y_err = float(she.pos()[1])
    yaw_ctrl = float(np.clip(-2.0 * y_err, -0.30, 0.30))
    
    for k, name in enumerate(('hip_pitch', 'knee', 'ankle')):
        idx = 0 if name == 'hip_pitch' else (1 if name == 'knee' else 3)
        off, amp, ph = p[3 * idx : 3 * idx + 3]
        wave = amp * np.sin(2.0 * np.pi * f * t + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)
    
    q[J['left_hip_yaw']] = yaw_ctrl
    q[J['right_hip_yaw']] = yaw_ctrl
    q[J['left_hip_roll']] = -0.30
    q[J['right_hip_roll']] = 0.30
    
    # Lifted head:
    q[J['neck_pitch']] = 0.35
    q[J['head_pitch']] = -0.15
    
    she.hold(q)
    w.step()
    
    pos = she.pos()
    furthest_x = max(furthest_x, pos[0])
    if step % 25 == 0:
        jw = d.xpos[jaw]
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        # check jaw contact
        jaw_con = any(
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]) == 'she_mouth_jaw' or
             mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2]) == 'she_mouth_jaw')
            for i in range(d.ncon)
        )
        print(f"t={t:5.2f}s trunk=({pos[0]:+6.3f},{pos[1]:+6.3f},{pos[2]:.3f}) jaw_z={jw[2]:.3f} (jaw_touch={jaw_con}) LF_x={lf[0]:+6.3f} RF_x={rf[0]:+6.3f}")

print(f"\nFinal trunk pos: {np.round(she.pos(), 3)}, furthest_x = {furthest_x:.3f} m")
