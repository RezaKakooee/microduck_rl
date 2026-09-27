"""Test heading and lateral control to keep Sister strictly on Brother's centerline."""
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

def test_heading_gain(k_yaw: float, k_pos: float):
    print(f"\n--- Testing with k_yaw = {k_yaw}, k_pos = {k_pos} ---")
    story = Story(verbose=False)
    
    # Measure nominal heading at t=0
    yaw0 = story.she.yaw()
    
    def compute_targets(self, t: float):
        d = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()

        if self.phase in ("wait", "he_lies"):
            return q

        # Heading error relative to nominal
        yaw_curr = d.yaw()
        yaw_err = (yaw_curr - yaw0 + np.pi) % (2 * np.pi) - np.pi
        
        # Lateral position error (y should be 0.000m)
        y_err = float(d.pos()[1])
        
        # Combined steering signal
        steer = float(np.clip(-k_pos * y_err - k_yaw * yaw_err, -0.25, 0.25))

        # Roll stabilization (k_pitch = 0.4)
        r_roll = float(d.data.xmat[d.trunk, 7])
        dp = float(np.clip(0.4 * r_roll, -0.25, 0.25))

        hp_bias = 0.30
        kn_bias = 0.00
        hr_val = 0.30

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                # Differential hip roll to steer
                q[J['left_hip_roll']] = -hr_val - steer
                q[J['right_hip_roll']] = hr_val - steer
            elif name == 'hip_pitch':
                # Differential pitch: dp for roll, plus forward thrust
                q[J['left_' + name]] = (off + hp_bias - dp) + wave
                q[J['right_' + name]] = -((off + hp_bias + dp) - wave)
            elif name == 'knee':
                q[J['left_' + name]] = (off + kn_bias) + wave
                q[J['right_' + name]] = -((off + kn_bias) - wave)
            else:
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)

        # Differential hip yaw to steer
        q[J['left_hip_yaw']] = steer
        q[J['right_hip_yaw']] = steer
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
            yaw_deg = np.degrees((story.she.yaw() - yaw0 + np.pi) % (2 * np.pi) - np.pi)
            print(f"t={story.world.t:5.2f}s pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f} yaw={yaw_deg:+.1f}deg")
            
        if story.phase == "across" and story.settle_time >= 3.0:
            print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
            break

    rear_final = rearmost(story.she)
    print(f"Result k_yaw={k_yaw}, k_pos={k_pos}: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")

for k_yaw in [0.5, 1.0, 2.0]:
    for k_pos in [2.0, 4.0]:
        test_heading_gain(k_yaw, k_pos)
