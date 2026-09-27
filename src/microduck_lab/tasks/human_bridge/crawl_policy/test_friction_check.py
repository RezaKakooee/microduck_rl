"""Inspect foot contacts and forces on Brother's back."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story

# Let's inspect what happens in test_steered_crawl at t=5.0s
from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK, HoldStraight

w = World(L.design())
he, she = w.ducks['he'], w.ducks['she']
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
hold_he = HoldStraight(he)

from scipy.spatial.transform import Rotation as R
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

for _ in range(150):
    hold_he()
    she.hold(c_base)
    w.step()

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
f = p[-3]
t_start = w.t
d, m = w.data, w.model

# Run to t=5.0s
for step in range(250):
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
    q[J['neck_pitch']] = -0.50
    q[J['head_pitch']] = -0.50
    she.hold(q)
    w.step()

print(f"At t=5.0s, Sister pos: {np.round(she.pos(), 3)}")
print(f"Sister velocities (qvel): {np.round(d.qvel[she.dof : she.dof + 6], 3)}")

# Check contact forces:
for c_i in range(d.ncon):
    c = d.contact[c_i]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
    if (b1.startswith('she_') and b2.startswith('he_')) or (b2.startswith('she_') and b1.startswith('he_')):
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, c_i, f6)
        print(f"  {b1} <-> {b2}: pos={np.round(c.pos, 3)}, fn={f6[0]:.2f}N, friction=({f6[1]:.2f}, {f6[2]:.2f})")
