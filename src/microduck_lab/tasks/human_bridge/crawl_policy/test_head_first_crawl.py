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

# Place brother on near step in PLANK
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
hold_he = HoldStraight(he)

# Sister HEAD-FIRST orientation: head facing +X across gap, belly on ground, legs behind:
rot = R.from_euler('y', 90, degrees=True)
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

c_base = np.zeros(len(SERVOS))
J = {name: i for i, name in enumerate(SERVOS)}
c_base[J['neck_pitch']] = -0.35  # chin tucked up so head slides smooth
c_base[J['head_pitch']] = -0.20
c_base[J['left_hip_roll']] = 0.15
c_base[J['right_hip_roll']] = -0.15

she.place((-0.20, 0.0, L.near_z + 0.05), q=c_base)
she.data.qpos[she.adr + 3 : she.adr + 7] = quat_wxyz
mujoco.mj_forward(w.model, w.data)

w.start()

# Let brother settle
for _ in range(150):
    hold_he()
    she.hold(c_base)
    w.step()

print(f"Settled: he_up = {he.up():.3f}, sister pos = {np.round(she.pos(), 3)}")

# Crawl gait: alternating push and recovery
f = 2.0 # 2 Hz
t_start = w.t
print("Sister starting head-first crawl across bridge...")

for i in range(400): # 8 seconds
    t = w.t - t_start
    hold_he()
    
    q = c_base.copy()
    
    # Left leg phase phi_l in [0, 2*pi)
    phi_l = (2.0 * np.pi * f * t) % (2.0 * np.pi)
    phi_r = (phi_l + np.pi) % (2.0 * np.pi)
    
    for phi, side in [(phi_l, 'left_'), (phi_r, 'right_')]:
        # Stance (push back): phi in [0, pi]
        # Swing (recover forward): phi in [pi, 2*pi]
        if phi < np.pi:
            u = phi / np.pi # 0 to 1
            hp = -1.10 # firmly pressed down
            kn = 1.0 - 2.0 * u # sweeps from +1.0 to -1.0
            an = -0.20
        else:
            u = (phi - np.pi) / np.pi # 0 to 1
            hp = -0.20 + 0.60 * np.sin(np.pi * u) # lifted off ground
            kn = -1.0 + 2.0 * u # sweeps from -1.0 to +1.0
            an = 0.20
            
        sign = 1.0 if side == 'left_' else -1.0
        q[J[side + 'hip_pitch']] = sign * hp
        q[J[side + 'knee']] = sign * kn
        q[J[side + 'ankle']] = sign * an
        q[J[side + 'hip_roll']] = sign * 0.15
        
    q[J['neck_pitch']] = -0.35
    q[J['head_pitch']] = -0.20

    she.hold(q)
    w.step()

    if i % 25 == 0:
        pos = np.round(she.pos(), 3)
        print(f"t={w.t:.2f}s pos={pos} he_up={he.up():.3f}")

print(f"Final t={w.t:.2f}s pos={np.round(she.pos(), 3)} he_up={he.up():.3f}")
