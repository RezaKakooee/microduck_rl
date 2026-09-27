"""Inspect Sister and Brother contacts and state when Sister is on Brother's back."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

s = Story(verbose=False)
m, d = s.world.model, s.world.data
J = {name: i for i, name in enumerate(SERVOS)}

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')

def check(self):
    pass # don't halt
s.check = check.__get__(s)

def compute_targets(self, t: float):
    d_duck = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()
    if self.phase in ("wait", "he_lies"):
        return q

    # Baseline crawl parameters from test_crossing_free (which got furthest)
    hp_bias = 0.30
    hr_val = 0.30

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
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)
        else:
            q[J['left_' + name]] = off + wave
            q[J['right_' + name]] = -(off - wave)

    q[J['left_hip_yaw']] = 0.0
    q[J['right_hip_yaw']] = 0.0
    q[J['neck_pitch']] = 0.30
    q[J['head_pitch']] = -0.15
    return np.clip(q, d_duck.lo, d_duck.hi)

s.controller.compute_targets = compute_targets.__get__(s.controller)

# Run until t = 10s (when Sister is on Brother's back)
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

for i in range(1250): # 25 seconds
    try:
        s.tick()
    except RuntimeError as e:
        print(f"Halted at t={s.world.t:.2f}s: {e}")
        break

    if i % 50 == 0 and s.world.t >= 5.0:
        pos = s.she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        rear = rearmost(s.she)
        r_roll = float(s.she.data.xmat[s.she.trunk, 7])
        he_up = s.he.up()
        print(f"t={s.world.t:5.2f}s phase={s.phase:7s} she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} furthest_x={s.furthest_x:+.3f} roll={r_roll:+.2f} he_up={he_up:.3f}")

    if s.phase == "across" and s.settle_time >= 3.0:
        print(f">>> SUCCESS! Reached across and settled at t={s.world.t:.2f}s!")
        break

print(f"FINAL: furthest_x = {s.furthest_x:.3f} m, final_pos = {np.round(s.she.pos(), 3)}, rear = {rearmost(s.she):.3f} m, phase = {s.phase}")
