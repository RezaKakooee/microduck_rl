"""Systematic test of locomotion along Brother's back (fresh world per test)."""
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

def run_test(test_name, f_val, kn_off, kn_amp, hp_off, hp_amp, an_off, an_amp):
    w = World(L.design())
    he, she = w.ducks['he'], w.ducks['she']
    he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
    hold_he = HoldStraight(he)
    
    # Place Sister at x = 0.00 (just onto Brother)
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
    t0 = w.t
    
    for i in range(250): # 5 seconds
        t = w.t - t0
        hold_he()
        q = c_base.copy()
        
        w_hp = hp_amp * np.sin(2.0 * np.pi * f_val * t + p_orig[2])
        w_kn = kn_amp * np.sin(2.0 * np.pi * f_val * t + p_orig[5])
        w_an = an_amp * np.sin(2.0 * np.pi * f_val * t + p_orig[11])
        
        q[J['left_hip_pitch']] = hp_off + w_hp
        q[J['right_hip_pitch']] = -(hp_off - w_hp)
        q[J['left_knee']] = kn_off + w_kn
        q[J['right_knee']] = -(kn_off - w_kn)
        q[J['left_ankle']] = an_off + w_an
        q[J['right_ankle']] = -(an_off - w_an)
        
        q[J['left_hip_roll']] = -0.25
        q[J['right_hip_roll']] = 0.25
        q[J['neck_pitch']] = -0.50
        q[J['head_pitch']] = -0.50
        
        she.hold(q)
        w.step()
        
    x_end = she.pos()[0]
    dx = (x_end - x_start) * 1000.0
    print(f"{test_name:18s}: start={x_start:.3f}, end={x_end:.3f}, dx={dx:+.1f} mm, he_up={he.up():.3f}")

tests = [
    ("orig_gait", (2.09, 0.954, 1.179, -0.890, 1.231, -0.752, 0.887)),
    ("low_freq", (1.2, 0.954, 1.179, -0.890, 1.231, -0.752, 0.887)),
    ("high_freq", (3.0, 0.954, 1.179, -0.890, 1.231, -0.752, 0.887)),
    ("deep_knee", (2.0, 1.20, 1.40, -1.00, 1.00, -1.00, 0.80)),
    ("shallow_knee", (2.0, 0.60, 0.80, -1.00, 1.00, -0.50, 0.50)),
    ("extended_knee", (2.0, -0.50, 0.80, -1.00, 0.80, -0.50, 0.50)),
]

for name, params in tests:
    run_test(name, *params)
