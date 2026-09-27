"""Test roll-stabilized crawl from START_X = -0.22m."""
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

    # Roll stabilization feedback:
    # data.xmat[trunk, 7] is trunk roll component
    r_roll = float(d.data.xmat[d.trunk, 7])
    roll_correct = float(np.clip(1.2 * r_roll, -0.25, 0.25))

    for k, name in enumerate(('hip_pitch', 'knee', 'ankle')):
        idx = 0 if name == 'hip_pitch' else (1 if name == 'knee' else 3)
        off, amp, ph = self.params[3 * idx : 3 * idx + 3]
        wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)

    # Centered hips with active roll stabilization:
    q[J['left_hip_roll']] = -0.28 + roll_correct
    q[J['right_hip_roll']] = 0.28 + roll_correct

    # Lifted head:
    q[J['neck_pitch']] = -0.50
    q[J['head_pitch']] = -0.50

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting roll-stabilized simulation from -0.22m...")
try:
    for i in range(500): # 10 seconds
        story.tick()
        if i % 25 == 0:
            r_roll = abs(float(story.she.data.xmat[story.she.trunk, 7]))
            print(f"t={story.world.t:5.2f}s {story.phase:7s} she_x={story.she.pos()[0]:.3f} z={story.she.pos()[2]:.3f} roll={r_roll:.3f} he_up={story.he.up():.3f}")
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m")
