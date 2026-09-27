"""Calculate Sister foot Y position as a function of hip_roll and stride."""
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
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

print("Brother back width is 71 mm (half-width = 35.5 mm: y in [-0.0355, +0.0355])")
print("Measuring Sister foot positions at base pose for different hr_val:")

for hr in [-0.50, -0.30, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]:
    q = s.she.data.qpos[s.she.qidx].copy()
    q[J['left_hip_roll']] = -hr
    q[J['right_hip_roll']] = hr
    q[J['left_hip_pitch']] = -0.90 # hp_bias = 0.30
    q[J['right_hip_pitch']] = 0.90
    d.qpos[s.she.qidx] = q
    mujoco.mj_forward(m, d)
    # Check body and foot positions
    she_y = d.xpos[s.she.trunk, 1]
    fl_y = d.xpos[foot_l, 1] - she_y
    fr_y = d.xpos[foot_r, 1] - she_y
    span = abs(fl_y - fr_y) * 1000 # mm
    print(f"hr={hr:+.2f} -> LF_y={fl_y*1000:+5.1f}mm, RF_y={fr_y*1000:+5.1f}mm, total span={span:5.1f}mm (fits back: {span <= 71.0})")

