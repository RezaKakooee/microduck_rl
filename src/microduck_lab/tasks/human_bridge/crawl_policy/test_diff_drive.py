"""Test differential drive steering (tank steering via hip pitch bias) to keep Sister centered."""
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

def test_steer(k_steer: float, k_roll: float, hp_bias: float = 0.30, kn_bias: float = 0.00, hr_val: float = 0.30):
    print(f"\n=== Testing k_steer = {k_steer:.2f}, k_roll = {k_roll:.2f}, hp_bias = {hp_bias:.2f} ===")
    story = Story(verbose=False)
    
    def compute_targets(self, t: float):
        d = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()

        if self.phase in ("wait", "he_lies"):
            return q

        # Lateral position error (y should be 0.000m)
        y_pos = float(d.pos()[1])
        # If y < 0 (left of center), left leg should push more, right less -> steer right
        # steer_bias > 0 when y < 0:
        steer = float(np.clip(-k_steer * y_pos, -0.25, 0.25))

        # Roll stabilization
        r_roll = float(d.data.xmat[d.trunk, 7])
        dp_roll = float(np.clip(k_roll * r_roll, -0.20, 0.20))

        # Combined pitch bias for left and right
        left_pitch_bias = hp_bias + steer - dp_roll
        right_pitch_bias = hp_bias - steer + dp_roll

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -hr_val
                q[J['right_hip_roll']] = hr_val
            elif name == 'hip_pitch':
                q[J['left_' + name]] = (off + left_pitch_bias) + wave
                q[J['right_' + name]] = -((off + right_pitch_bias) - wave)
            elif name == 'knee':
                q[J['left_' + name]] = (off + kn_bias) + wave
                q[J['right_' + name]] = -((off + kn_bias) - wave)
            else:
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)

        q[J['left_hip_yaw']] = 0.0
        q[J['right_hip_yaw']] = 0.0
        q[J['neck_pitch']] = 0.30
        q[J['head_pitch']] = -0.15

        return np.clip(q, d.lo, d.hi)

    story.controller.compute_targets = compute_targets.__get__(story.controller)

    for i in range(1250): # 25 seconds
        try:
            story.tick()
        except RuntimeError as e:
            print(f"Halted at t={story.world.t:.2f}s: {e}")
            break
            
        if i % 50 == 0:
            pos = story.she.pos()
            rear = rearmost(story.she)
            r_roll = float(story.she.data.xmat[story.she.trunk, 7])
            he_up = story.he.up()
            print(f"t={story.world.t:5.2f}s phase={story.phase:7s} she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f} he_up={he_up:.3f}")
            
        if story.phase == "across" and story.settle_time >= 3.0:
            print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
            break

    rear_final = rearmost(story.she)
    print(f"Result k_steer={k_steer}, k_roll={k_roll}: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")

for ks in [1.0, 2.0, 3.0, 4.0]:
    for kr in [0.0, 0.2, 0.4]:
        test_steer(ks, kr)
