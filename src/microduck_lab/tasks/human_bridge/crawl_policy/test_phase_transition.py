"""Test deterministic edge transition across the notch."""
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

# State variable for the transition
transition_t0 = None

def compute_targets(self, t: float):
    global transition_t0
    d = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()

    if self.phase in ("wait", "he_lies"):
        return q

    pos_x = d.pos()[0]
    r_roll = float(d.data.xmat[d.trunk, 7])

    if pos_x < -0.085 and transition_t0 is None:
        # Phase 1: Fast crawl across near ledge
        for k, name in enumerate(('hip_pitch', 'knee', 'ankle')):
            idx = 0 if name == 'hip_pitch' else (1 if name == 'knee' else 3)
            off, amp, ph = self.params[3 * idx : 3 * idx + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)

        q[J['left_hip_roll']] = -0.28
        q[J['right_hip_roll']] = 0.28
        q[J['neck_pitch']] = -0.50
        q[J['head_pitch']] = -0.50

    elif pos_x < 0.02:
        # Phase 2: Notch transition - step onto Brother and pull!
        if transition_t0 is None:
            transition_t0 = t
            print(f">>> Triggered notch transition at t={t:.2f}s, x={pos_x:.3f}m")

        dt_trans = t - transition_t0
        # 1. First 0.3s: Extend both legs forward onto Brother's shins (x >= 0.02)
        if dt_trans < 0.35:
            u = dt_trans / 0.35
            # Straighten legs forward:
            hp = -1.20 * (1 - u) + (-0.80) * u
            kn = -0.60 * (1 - u) + (-0.90) * u  # knee straight forward
            an = 0.00 * (1 - u) + (-1.20) * u
            q[J['left_hip_pitch']] = hp
            q[J['right_hip_pitch']] = -hp
            q[J['left_knee']] = kn
            q[J['right_knee']] = -kn
            q[J['left_ankle']] = an
            q[J['right_ankle']] = -an
        else:
            # 2. Next 0.8s: Pull knees and hips to drag pelvis over the lip
            u = min(1.0, (dt_trans - 0.35) / 0.80)
            hp = -0.80 * (1 - u) + (-1.40) * u  # hip presses down
            kn = -0.90 * (1 - u) + (+1.20) * u  # knee bends to pull body forward!
            an = -1.20 * (1 - u) + (-0.50) * u
            q[J['left_hip_pitch']] = hp
            q[J['right_hip_pitch']] = -hp
            q[J['left_knee']] = kn
            q[J['right_knee']] = -kn
            q[J['left_ankle']] = an
            q[J['right_ankle']] = -an

        q[J['left_hip_roll']] = -0.25
        q[J['right_hip_roll']] = 0.25
        q[J['neck_pitch']] = -0.50
        q[J['head_pitch']] = -0.50

    else:
        # Phase 3: On Brother's back! Crawl along Brother
        u_bro = t - transition_t0 - 1.15
        for k, name in enumerate(('hip_pitch', 'knee', 'ankle')):
            idx = 0 if name == 'hip_pitch' else (1 if name == 'knee' else 3)
            off, amp, ph = self.params[3 * idx : 3 * idx + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_bro + ph)
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)

        q[J['left_hip_roll']] = -0.20
        q[J['right_hip_roll']] = 0.20
        q[J['neck_pitch']] = -0.50
        q[J['head_pitch']] = -0.50

    return np.clip(q, d.lo, d.hi)

story.controller.compute_targets = compute_targets.__get__(story.controller)

print("Starting simulation with deterministic transition...")
try:
    for i in range(500): # 10 seconds
        story.tick()
        if i % 25 == 0:
            print(f"t={story.world.t:5.2f}s {story.phase:7s} she_x={story.she.pos()[0]:.3f} z={story.she.pos()[2]:.3f} he_up={story.he.up():.3f}")
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m")
