"""Analyze foot trajectory in body frame during crawl gait."""
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
p = np.load('local_storage/gaits/crawl_gait.npy')
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
J = {name: i for i, name in enumerate(SERVOS)}
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
trunk = she.trunk

c_base = np.zeros(len(SERVOS))
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'neck_pitch': 0.241, 'head_pitch': -0.135}.items():
    c_base[J[name]] = val

print("Foot position relative to trunk over 1 cycle (T=%.2fs):" % (1.0/f))
steps = 25
for step in range(steps):
    t = step / steps / f
    q = c_base.copy()
    for k, name in enumerate(JOINTS):
        off, amp, ph = p[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * f * t + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)
    
    # Set qpos directly and forward kinematics
    d.qpos[she.adr + 7 : she.adr + 7 + 14] = q
    mujoco.mj_forward(m, d)
    
    # Vector from trunk to foot in trunk body frame
    R_trunk = d.xmat[trunk].reshape(3, 3)
    p_rel = R_trunk.T @ (d.xpos[foot_l] - d.xpos[trunk])
    
    # In body frame: +X is chest/front, +Y is left, +Z is head/up
    # Since robot is lying down with legs at -Z in rest, let's see p_rel:
    print(f"t={t:5.3f}s | p_rel: X(chest)={p_rel[0]:+.3f}, Y(left)={p_rel[1]:+.3f}, Z(head)={p_rel[2]:+.3f} | q_hp={q[J['left_hip_pitch']]:+.2f} q_kn={q[J['left_knee']]:+.2f} q_an={q[J['left_ankle']]:+.2f}")
