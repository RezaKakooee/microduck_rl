"""Test head-first crawl gait on the bridge."""
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

# Head-first orientation: head points +X, belly down, legs at -X
rot = R.from_euler('y', 90, degrees=True)
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
J = {name: i for i, name in enumerate(SERVOS)}
for name, val in {'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
                  'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
                  'neck_pitch': 0.20, 'head_pitch': -0.20}.items():
    c_base[J[name]] = val

she.place((-0.25, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)
w.start()

# Settle
for _ in range(100):
    hold_he()
    she.hold(c_base)
    w.step()

x_start = she.pos()[0]
print(f"Settled start pos: {np.round(she.pos(), 3)}")

# Test a clean 2-phase crawl gait:
# Left and right legs alternate (180 deg out of phase).
# In each cycle of period T = 0.5s (2 Hz):
# Stance (push back): foot extends back, pushing ground -> propels robot forward (+X)
# Swing (lift & reach): foot lifts off ground and reaches forward to reset.

f = 2.5 # 2.5 Hz
d, m = w.data, w.model

for step in range(250): # 5 seconds
    t = step * 0.02
    hold_he()
    q = c_base.copy()
    
    for side, sgn, phase_offset in [('left_', 1.0, 0.0), ('right_', -1.0, np.pi)]:
        phi = (2.0 * np.pi * f * t + phase_offset) % (2.0 * np.pi)
        
        # Stance: phi in [0, pi]
        # Leg pushes backward: hip_pitch sweeps down, knee extends, pushing against ground
        if phi < np.pi:
            u = phi / np.pi # 0 -> 1
            # Push backward:
            hp = -1.00 + 0.50 * u   # hip presses into ground
            kn = -0.80 + 1.60 * u   # knee extends from -0.80 to +0.80 (pushes backward!)
            an = -0.20 - 0.40 * u   # ankle flexes
            hr = 0.15
        else:
            u = (phi - np.pi) / np.pi # 0 -> 1
            # Swing: lift leg and bring forward
            hp = -0.50 - 0.50 * np.sin(np.pi * u)  # lifted
            kn = 0.80 - 1.60 * u                   # knee flexes forward from +0.80 to -0.80
            an = -0.60 + 0.40 * u
            hr = 0.25                              # slightly wider during swing
            
        q[J[side + 'hip_pitch']] = sgn * hp
        q[J[side + 'knee']] = sgn * kn
        q[J[side + 'ankle']] = sgn * an
        q[J[side + 'hip_roll']] = sgn * hr
        
    q[J['neck_pitch']] = 0.20
    q[J['head_pitch']] = -0.20
    
    she.hold(q)
    w.step()
    
    if step % 25 == 0:
        pos = she.pos()
        print(f"t={t:.2f}s pos={np.round(pos, 3)} dx={1000*(pos[0]-x_start):+.1f}mm")

pos_final = she.pos()
dx_total = 1000 * (pos_final[0] - x_start)
print(f"Final pos: {np.round(pos_final, 3)} | Total dx = {dx_total:+.1f} mm")
