"""Analyze foot trajectories during the crawl cycle for different hip_roll and hip_yaw."""
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

w = World(L.design())
she = w.ducks['she']
m, d = w.model, w.data
J = {name: i for i, name in enumerate(SERVOS)}

rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

c_base = np.zeros(len(SERVOS))
she.place((0.0, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(m, d)

# Test different hip_roll configurations
configs = [
    ("Default crawl_gait hip_roll", None, None),
    ("Constant hr = (-0.28, +0.28)", -0.28, +0.28),
    ("Constant hr = (-0.35, +0.35)", -0.35, +0.35),
    ("Inverted hr = (+0.28, -0.28)", +0.28, -0.28),
    ("Zero hr = (0.0, 0.0)", 0.0, 0.0),
]

for title, hr_l, hr_r in configs:
    print(f"\n--- {title} ---")
    min_lf_y, max_lf_y = 100, -100
    min_rf_y, max_rf_y = 100, -100
    min_lf_z, max_lf_z = 100, -100
    min_rf_z, max_rf_z = 100, -100
    
    for step in range(50):
        t = step / 50.0 / f # one full cycle
        q = c_base.copy()
        for k, name in enumerate(JOINTS):
            off, amp, ph = p[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * f * t + ph)
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)
            
        if hr_l is not None:
            q[J['left_hip_roll']] = hr_l
            q[J['right_hip_roll']] = hr_r
            
        d.qpos[she.adr + 7 : she.adr + 7 + 14] = q
        mujoco.mj_forward(m, d)
        
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        
        min_lf_y = min(min_lf_y, lf[1]); max_lf_y = max(max_lf_y, lf[1])
        min_rf_y = min(min_rf_y, rf[1]); max_rf_y = max(max_rf_y, rf[1])
        min_lf_z = min(min_lf_z, lf[2]); max_lf_z = max(max_lf_z, lf[2])
        min_rf_z = min(min_rf_z, rf[2]); max_rf_z = max(max_rf_z, rf[2])
        
    print(f"  LF Y range: [{min_lf_y*1000:+6.1f}, {max_lf_y*1000:+6.1f}] mm")
    print(f"  RF Y range: [{min_rf_y*1000:+6.1f}, {max_rf_y*1000:+6.1f}] mm")
    print(f"  LF Z range: [{min_lf_z*1000:+6.1f}, {max_lf_z*1000:+6.1f}] mm")
    print(f"  RF Z range: [{min_rf_z*1000:+6.1f}, {max_rf_z*1000:+6.1f}] mm")
