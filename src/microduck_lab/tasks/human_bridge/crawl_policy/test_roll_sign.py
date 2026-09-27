"""Check sign of xmat[7] when rolling left vs rolling right."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from scipy.spatial.transform import Rotation as R

s = Story(verbose=False)
m, d = s.world.model, s.world.data

rot0 = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

# Roll around world X axis:
# Positive roll in world: right-hand rule about +X (tilts +Y towards +Z, so left side goes up, right side goes down)
for roll_deg in [-10, -5, 0, +5, +10]:
    r_world = R.from_euler('x', roll_deg, degrees=True)
    r_comb = r_world * rot0
    q_xyzw = r_comb.as_quat()
    d.qpos[s.she.adr + 3 : s.she.adr + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    mujoco.mj_forward(m, d)
    val7 = float(d.xmat[s.she.trunk, 7])
    # Which side is lower in world Z:
    # +Y is left in world. If left is lower, z_left < z_right
    print(f"roll = {roll_deg:+3d} deg (world) -> xmat[7] = {val7:+.4f}")
