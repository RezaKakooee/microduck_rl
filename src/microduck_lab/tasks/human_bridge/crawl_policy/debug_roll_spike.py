"""Inspect contacts, velocities, and forces during the roll spike at t=10s."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
J = {name: i for i, name in enumerate(SERVOS)}

story = Story(verbose=False)
hr_val = 0.36
k_roll = 0.25
hp_bias = 0.30

def compute_targets(self, t: float):
    d = self.duck
    u_phase = t - self.t_start_phase
    q = self.target.copy()

    if self.phase in ("wait", "he_lies"):
        return q

    r_roll = float(d.data.xmat[d.trunk, 7])
    dp = float(np.clip(k_roll * r_roll, -0.25, 0.25))

    for k, name in enumerate(JOINTS):
        off, amp, ph = self.params[3 * k : 3 * k + 3]
        wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
        if name == 'hip_roll':
            q[J['left_hip_roll']] = -hr_val
            q[J['right_hip_roll']] = hr_val
        elif name == 'hip_pitch':
            q[J['left_' + name]] = (off + hp_bias - dp) + wave
            q[J['right_' + name]] = -((off + hp_bias + dp) - wave)
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

for step in range(600): # 12 seconds
    t = story.world.t
    try:
        story.tick()
    except RuntimeError as e:
        print(f"FAILED at t={t:.3f}s: {e}")
        # Print all active contacts on Sister
        print("\nActive contacts at failure:")
        for c_idx in range(d.ncon):
            con = d.contact[c_idx]
            g1, g2 = con.geom1, con.geom2
            name1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
            name2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"
            if "she" in name1 or "she" in name2:
                pos = con.pos
                dist = con.dist
                print(f"  {name1:25s} <-> {name2:25s} at pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}) dist={dist:+.4f}")
        break
        
    if t >= 9.80 and step % 2 == 0:
        pos = story.she.pos()
        r_roll = float(d.xmat[story.she.trunk, 7])
        ang_vel = d.qvel[story.she.dof + 3 : story.she.dof + 6]
        print(f"t={t:6.3f}s she_pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}) roll={r_roll:+.3f} w_roll={ang_vel[0]:+.3f}")
