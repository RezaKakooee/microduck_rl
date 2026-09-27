"""Test full crossing with hip_yaw=0.0 and symmetrical adducted hip_roll."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
J = {name: i for i, name in enumerate(SERVOS)}

story = Story(verbose=False)

def compute_targets(self, t: float):
    d = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()

    if self.phase in ("wait", "he_lies"):
        return q

    # Symmetrical crawl with adducted hip_roll and NO yaw steering
    for k, name in enumerate(JOINTS):
        off, amp, ph = self.params[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
        if name == 'hip_roll':
            q[J['left_hip_roll']] = -0.35
            q[J['right_hip_roll']] = 0.35
        else:
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)

    q[J['left_hip_yaw']] = 0.0
    q[J['right_hip_yaw']] = 0.0
    q[J['neck_pitch']] = 0.30
    q[J['head_pitch']] = -0.15

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting simulation with hip_yaw=0.0 and hip_roll=-0.35/+0.35...")
for i in range(1000): # 20 seconds
    try:
        story.tick()
    except RuntimeError as e:
        print(f"Halted at t={story.world.t:.2f}s: {e}")
        break
        
    if i % 50 == 0:
        pos = story.she.pos()
        rear = rearmost(story.she)
        r_roll = float(story.she.data.xmat[story.she.trunk, 7])
        print(f"t={story.world.t:5.2f}s pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f}")
        
    if story.phase == "across" and story.settle_time >= 3.0:
        print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
        break

rear_final = rearmost(story.she)
print(f"Final: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")
