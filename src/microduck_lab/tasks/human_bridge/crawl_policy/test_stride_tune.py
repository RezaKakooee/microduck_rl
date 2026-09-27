"""Search hip_pitch and knee offsets to propel Sister across Brother's back."""
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

def run_trial(hp_bias: float, kn_bias: float, hr_val: float):
    story = Story(verbose=False)
    
    def compute_targets(self, t: float):
        d = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()

        if self.phase in ("wait", "he_lies"):
            return q

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -hr_val
                q[J['right_hip_roll']] = hr_val
            elif name == 'hip_pitch':
                q[J['left_' + name]] = (off + hp_bias) + wave
                q[J['right_' + name]] = -((off + hp_bias) - wave)
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

    for i in range(500): # 10 seconds
        try:
            story.tick()
        except RuntimeError:
            break
        if story.phase == "across":
            break

    rear_final = rearmost(story.she)
    print(f"hp_bias={hp_bias:+.2f} kn_bias={kn_bias:+.2f} hr={hr_val:.2f} -> furthest_x={story.furthest_x:+.3f}m rear={rear_final:+.3f}m final_pos={np.round(story.she.pos(), 3)}")
    return story.furthest_x

print("Searching stride parameters:")
for hp_b in [0.0, 0.15, 0.30]:
    for kn_b in [-0.20, 0.0, +0.20]:
        for hr in [0.25, 0.30, 0.35]:
            run_trial(hp_b, kn_b, hr)
