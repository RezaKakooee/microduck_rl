"""Test active differential hip pitch roll stabilization during crossing."""
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

def test_roll_stabilizer(k_pitch: float):
    print(f"\n--- Testing with k_pitch = {k_pitch} ---")
    story = Story(verbose=False)
    
    def compute_targets(self, t: float):
        d = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()

        if self.phase in ("wait", "he_lies"):
            return q

        # Roll error: xmat[7] > 0 means tilted left
        r_roll = float(d.data.xmat[d.trunk, 7])
        # Angular velocity around world X (roll rate)
        w_roll = float(d.data.qvel[d.dof + 3])
        
        # Differential pitch stabilization:
        # If tilted left (r_roll > 0), left leg pushes DOWN (more negative pitch), right leg relaxes
        dp = float(np.clip(k_pitch * r_roll + 0.05 * w_roll, -0.30, 0.30))

        y_err = float(d.pos()[1])
        yaw_ctrl = float(np.clip(-2.0 * y_err, -0.20, 0.20))

        hp_bias = 0.30
        kn_bias = 0.00
        hr_val = 0.30

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -hr_val - 0.5 * yaw_ctrl
                q[J['right_hip_roll']] = hr_val - 0.5 * yaw_ctrl
            elif name == 'hip_pitch':
                # Apply differential pitch: dp pushes down on lower side
                q[J['left_' + name]] = (off + hp_bias - dp) + wave
                q[J['right_' + name]] = -((off + hp_bias + dp) - wave)
            elif name == 'knee':
                q[J['left_' + name]] = (off + kn_bias) + wave
                q[J['right_' + name]] = -((off + kn_bias) - wave)
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
            print(f"t={story.world.t:5.2f}s pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f}")
            
        if story.phase == "across" and story.settle_time >= 3.0:
            print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
            break

    rear_final = rearmost(story.she)
    print(f"Result k_pitch={k_pitch}: furthest_x = {story.furthest_x:.3f} m, rear = {rear_final:.3f} m, phase = {story.phase}")

for kp in [0.4, 0.8, 1.2]:
    test_roll_stabilizer(kp)
