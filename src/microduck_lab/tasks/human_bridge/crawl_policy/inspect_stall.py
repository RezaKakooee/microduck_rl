"""Inspect why Sister stalls at x=0.079m."""
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

story = Story(verbose=False)
def check(self):
    pass
story.check = check.__get__(story)

hp_bias = 0.30
hr_val = 0.30

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
m, d = story.world.model, story.world.data
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

for i in range(600): # 12 seconds
    story.tick()
    t = story.world.t
    if t >= 10.0 and i % 10 == 0:
        pos = story.she.pos()
        fl = d.xpos[foot_l]
        fr = d.xpos[foot_r]
        r_roll = float(d.xmat[story.she.trunk, 7])
        print(f"t={t:5.2f}s she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} LF=({fl[0]:+.3f},{fl[1]:+.3f},{fl[2]:.3f}) RF=({fr[0]:+.3f},{fr[1]:+.3f},{fr[2]:.3f}) roll={r_roll:+.2f}")

print("\n--- Active contacts at t=12s ---")
for c_idx in range(d.ncon):
    con = d.contact[c_idx]
    g1, g2 = con.geom1, con.geom2
    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"g{g1}"
    n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"g{g2}"
    if "she" in n1 or "she" in n2:
        print(f"  {n1:25s} <-> {n2:25s} at pos=({con.pos[0]:+.3f},{con.pos[1]:+.3f},{con.pos[2]:.3f})")
