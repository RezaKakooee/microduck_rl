"""Test Story with lifted head (+0.60, -0.60) step by step."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
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

    # Roll stabilization
    r_roll = float(d.data.xmat[d.trunk, 7])
    roll_correct = float(np.clip(1.2 * r_roll, -0.25, 0.25))

    for k, name in enumerate(JOINTS):
        off, amp, ph = self.params[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
        if name == 'hip_roll':
            q[J['left_hip_roll']] = -0.25 + roll_correct
            q[J['right_hip_roll']] = 0.25 + roll_correct
        else:
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)

    # LIFT HEAD HIGH IN AIR: positive neck_pitch, negative head_pitch
    q[J['neck_pitch']] = 0.60
    q[J['head_pitch']] = -0.60

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting simulation with head lifted high...")
try:
    for i in range(500): # 10 seconds
        story.tick()
        if i % 25 == 0:
            print(f"t={story.world.t:5.2f}s {story.phase:7s} she_x={story.she.pos()[0]:.3f} z={story.she.pos()[2]:.3f} he_up={story.he.up():.3f}")
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m")
