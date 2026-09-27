"""Inspect Sister's feet and joints when she reaches Brother's head (x=0.07m, t=11s)."""
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

def compute_targets(self, t: float):
    d_duck = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()
    if self.phase in ("wait", "he_lies"):
        return q

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

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# Run to t = 11.5s
for i in range(575):
    s.tick()

print(f"--- State at t={s.world.t:.2f}s ---")
pos = s.she.pos()
lf = d.xpos[foot_l]
rf = d.xpos[foot_r]
print(f"Trunk: {np.round(pos, 4)}, rear: {rearmost(s.she):.4f}")
print(f"LF: {np.round(lf, 4)}, RF: {np.round(rf, 4)}")

# Joint positions vs targets
q_act = d.qpos[s.she.qidx]
q_tgt = s.controller.target
print("\nJoint target vs actual:")
for name in SERVOS:
    idx = J[name]
    print(f"  {name:16s}: tgt={q_tgt[idx]:+6.2f}, act={q_act[idx]:+6.2f}, diff={q_act[idx]-q_tgt[idx]:+6.2f}")

# Contacts
print("\nContacts:")
for c_idx in range(d.ncon):
    con = d.contact[c_idx]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom1])
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom2])
    if 'she' in str(b1) or 'she' in str(b2):
        c_force = np.zeros(6)
        mujoco.mj_contactForce(m, d, c_idx, c_force)
        print(f"  {b1} touching {b2}: normal={c_force[0]:6.2f}N, pos=({con.pos[0]:+.3f}, {con.pos[1]:+.3f}, {con.pos[2]:.3f})")
