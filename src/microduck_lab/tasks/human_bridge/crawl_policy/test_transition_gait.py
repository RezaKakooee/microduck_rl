"""Test seamless transition controller from near ledge across Brother."""
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

# Replace compute_targets dynamically to test our transition logic
def compute_targets(self, t: float):
    d = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()

    if self.phase in ("wait", "he_lies"):
        return q

    pos_x = d.pos()[0]
    
    # 1. Base crawl wave from fast crawl skill
    for k, name in enumerate(JOINTS):
        off, amp, ph = self.params[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
        q[J['left_' + name]] = off + wave
        q[J['right_' + name]] = -(off - wave)

    # 2. Adaptive hip roll and head pitch:
    if pos_x < -0.09:
        # Near ledge: wide and stable, slight head lift
        q[J['neck_pitch']] = -0.20
        q[J['head_pitch']] = -0.20
    else:
        # Reaching notch and Brother: narrow hip roll, strong head lift
        q[J['left_hip_roll']] = -0.15
        q[J['right_hip_roll']] = 0.15
        q[J['neck_pitch']] = -0.60
        q[J['head_pitch']] = -0.50

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting simulation...")
try:
    for i in range(500): # 10 seconds
        story.tick()
        if i % 25 == 0:
            print(f"t={story.world.t:5.2f}s {story.phase:7s} she_x={story.she.pos()[0]:.3f} z={story.she.pos()[2]:.3f} he_up={story.he.up():.3f}")
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m")
