"""Test full crossing with roll feedback sign fixed (or disabled)."""
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

def test_roll_gain(gain: float, name_label: str):
    print(f"\n================ Testing with roll_gain = {gain} ({name_label}) ================")
    story = Story(verbose=False)
    
    def compute_targets(self, t: float):
        d = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()

        if self.phase in ("wait", "he_lies"):
            return q

        # Roll correction
        r_roll = float(d.data.xmat[d.trunk, 7])
        roll_correct = float(np.clip(gain * r_roll, -0.25, 0.25))

        # Centerline steering
        y_err = float(d.pos()[1])
        yaw_ctrl = float(np.clip(-2.5 * y_err, -0.30, 0.30))

        # Crawl gait
        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -0.30 + roll_correct
                q[J['right_hip_roll']] = 0.30 + roll_correct
            else:
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)

        q[J['left_hip_yaw']] = yaw_ctrl
        q[J['right_hip_yaw']] = yaw_ctrl
        q[J['neck_pitch']] = 0.30
        q[J['head_pitch']] = -0.15

        return np.clip(q, d.lo, d.hi)

    story.controller.compute_targets = compute_targets.__get__(story.controller)

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
            print(f"t={story.world.t:5.2f}s she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f}")
            
        if story.phase == "across" and story.settle_time >= 3.0:
            print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
            break

    rear_final = rearmost(story.she)
    print(f"Result {name_label}: furthest_x = {story.furthest_x:.3f} m, final_x = {story.she.pos()[0]:.3f} m, rear = {rear_final:.3f} m, phase = {story.phase}")

test_roll_gain(0.0, "zero gain (no roll feedback)")
test_roll_gain(-1.2, "negative feedback (restoring)")
