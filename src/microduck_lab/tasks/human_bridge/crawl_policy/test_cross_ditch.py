"""Test extending over the ditch onto Brother's shins without retracting into ditch."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
J = {name: i for i, name in enumerate(SERVOS)}

story = Story(verbose=True)

def compute_targets(self, t: float):
    d = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()

    if self.phase in ("wait", "he_lies"):
        return q

    pos_x = d.pos()[0]
    r_roll = float(d.data.xmat[d.trunk, 7])
    roll_correct = float(np.clip(1.2 * r_roll, -0.25, 0.25))

    if pos_x < -0.09:
        # Phase 1: Fast crawl across near ledge
        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -0.25 + roll_correct
                q[J['right_hip_roll']] = 0.25 + roll_correct
            else:
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)
        q[J['neck_pitch']] = 0.241
        q[J['head_pitch']] = -0.135
    else:
        # Phase 2: Crossing the ditch onto Brother!
        # Reach feet over the ditch onto Brother's shins (x >= 0.00)
        # Alternate legs with forward-biased stroke:
        # Do not retract knee into the ditch! Clamp knee so it stays in front of brother's ankles!
        f_cross = 2.0
        for side, sgn, phi_offset in [('left_', 1.0, 0.0), ('right_', -1.0, np.pi)]:
            phi = (2.0 * np.pi * f_cross * u_phase + phi_offset) % (2.0 * np.pi)
            if phi < np.pi:
                # Stance on brother: pull body forward
                u = phi / np.pi
                hp = -1.20 - 0.20 * np.sin(np.pi * u)
                kn = -0.80 + 1.20 * u   # pulls from -0.80 (extended onto brother) to +0.40
                an = -0.80
            else:
                # Swing over ditch: reach forward onto brother
                u = (phi - np.pi) / np.pi
                hp = -0.80 - 0.40 * np.sin(np.pi * u)  # lifted
                kn = 0.40 - 1.20 * u                   # extends forward
                an = -0.20
                
            q[J[side + 'hip_pitch']] = sgn * hp
            q[J[side + 'knee']] = sgn * kn
            q[J[side + 'ankle']] = sgn * an
            q[J[side + 'hip_roll']] = sgn * 0.15 + roll_correct

        q[J['neck_pitch']] = 0.241
        q[J['head_pitch']] = -0.135

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting simulation with ditch-crossing transition...")
try:
    for i in range(500): # 10 seconds
        story.tick()
        if i % 25 == 0:
            print(f"t={story.world.t:5.2f}s {story.phase:7s} she_x={story.she.pos()[0]:.3f} z={story.she.pos()[2]:.3f} he_up={story.he.up():.3f}")
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m")
