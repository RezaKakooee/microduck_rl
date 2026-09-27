"""Test lifted-foot crossing over Brother's ankles onto his back."""
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

    # Roll stabilization
    r_roll = float(d.data.xmat[d.trunk, 7])
    roll_correct = float(np.clip(1.2 * r_roll, -0.25, 0.25))

    # Centerline steering
    y_err = float(d.pos()[1])
    yaw_ctrl = float(np.clip(-2.0 * y_err, -0.30, 0.30))

    pos_x = float(d.pos()[0])

    # Dynamic gait scaling
    for k, name in enumerate(JOINTS):
        off, amp, ph = self.params[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
        if name == 'hip_roll':
            # Adducted hips to stay on Brother's back
            q[J['left_hip_roll']] = -0.32 + roll_correct
            q[J['right_hip_roll']] = 0.32 + roll_correct
        elif name == 'hip_pitch':
            # When near the notch (x in [-0.12, 0.05]), lift foot higher to step over Brother's ankles
            hp_lift = 0.35 if -0.12 <= pos_x <= 0.05 else 0.0
            q[J['left_hip_pitch']] = (off + hp_lift) + wave
            q[J['right_hip_pitch']] = -((off + hp_lift) - wave)
        elif name == 'knee':
            # Extend knee forward when reaching over the gap
            kn_bias = 0.25 if -0.12 <= pos_x <= 0.05 else 0.0
            q[J['left_knee']] = (off + kn_bias) + wave
            q[J['right_knee']] = -((off + kn_bias) - wave)
        else:
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)

    q[J['left_hip_yaw']] = yaw_ctrl
    q[J['right_hip_yaw']] = yaw_ctrl

    # Lift head so beak does not drag on floor
    q[J['neck_pitch']] = 0.30
    q[J['head_pitch']] = -0.15

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting simulation...")
try:
    for i in range(700): # 14 seconds
        story.tick()
        if i % 25 == 0:
            pos = story.she.pos()
            he_up = story.he.up()
            print(f"t={story.world.t:5.2f}s phase={story.phase:7s} she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} furthest_x={story.furthest_x:+.3f} he_up={he_up:.3f}")
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m, phase = {story.phase}")
