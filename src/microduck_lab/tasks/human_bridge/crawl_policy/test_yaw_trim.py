"""Test left-right differential pitch trim to keep Sister strictly at y=0.000m."""
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

def test_trim(trim: float, hp_bias: float = 0.35, hr_val: float = 0.30):
    print(f"\n================ Running trim = {trim:+.3f} ================")
    story = Story(verbose=False)
    
    def check(self):
        she, he = self.she, self.he
        if abs(she.pos()[1]) > 0.18:
            raise RuntimeError(f"she drifted off bridge sideways: y={she.pos()[1]:.3f} m")
        if self.phase in ("cross", "across") and she.pos()[2] < L.far_z - 0.08:
            raise RuntimeError(f"she dropped into gap at x={she.pos()[0]:.3f}, z={she.pos()[2]:.3f} m")
        if self.phase in ("cross", "across") and abs(he.up()) > 0.35:
            raise RuntimeError(f"brother was knocked loose: he.up() = {he.up():.3f}")

    story.check = check.__get__(story)

    def compute_targets(self, t: float):
        d = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()
        if self.phase in ("wait", "he_lies"):
            return q

        left_hp = hp_bias + trim
        right_hp = hp_bias - trim

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -hr_val
                q[J['right_hip_roll']] = hr_val
            elif name == 'hip_pitch':
                q[J['left_' + name]] = (off + left_hp) + wave
                q[J['right_' + name]] = -((off + right_hp) - wave)
            elif name == 'knee':
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)
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
    print(f"Result trim={trim:+.3f}: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")

for trim in [-0.06, -0.04, -0.02, +0.02, +0.04, +0.06]:
    test_trim(trim)
