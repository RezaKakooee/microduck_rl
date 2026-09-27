"""Test how knee extension and hip pitch affect foot reach along +X."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

s = Story(verbose=False)
m, d = s.world.model, s.world.data
J = {name: i for i, name in enumerate(SERVOS)}
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')

q0 = s.she.data.qpos[s.she.qidx].copy()
trunk_x0 = d.xpos[s.she.trunk, 0]

for hp in [-0.40, -0.20, 0.0, +0.20, +0.40]:
    for kn in [-0.60, -0.30, 0.0, +0.30]:
        q = q0.copy()
        q[J['left_hip_pitch']] = -0.89 + hp
        q[J['right_hip_pitch']] = 0.89 - hp
        q[J['left_knee']] = 0.95 + kn
        q[J['right_knee']] = -0.95 - kn
        q[J['left_hip_roll']] = -0.30
        q[J['right_hip_roll']] = 0.30
        d.qpos[s.she.qidx] = q
        mujoco.mj_forward(m, d)
        reach_x = (d.xpos[foot_l, 0] - d.xpos[s.she.trunk, 0]) * 1000 # mm
        foot_z = d.xpos[foot_l, 2] * 1000 # mm
        print(f"hp={hp:+.2f}, kn={kn:+.2f} -> reach_x={reach_x:+6.1f}mm, foot_z={foot_z:6.1f}mm")
