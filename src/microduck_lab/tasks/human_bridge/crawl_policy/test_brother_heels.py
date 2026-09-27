"""Check Brother's rearmost heel point when standing and when lying down."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK
from microduck_lab.tasks.human_bridge.crawl_policy.story import STAND_POSE

w = World(L.design())
he = w.ducks['he']
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, STAND_POSE)
mujoco.mj_forward(w.model, w.data)

soles = he.points(he.sole_geoms)
solids = he.points(he.solid)

print(f"Brother standing at x={L.gap_start - 0.005:.3f}:")
print(f"  Sole geoms X range: [{soles[:, 0].min():.4f}, {soles[:, 0].max():.4f}] m")
print(f"  Solid geoms X min (heel): {solids[:, 0].min():.4f} m")

# Now check Brother when in PLANK
he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, PLANK)
mujoco.mj_forward(w.model, w.data)
solids_plank = he.points(he.solid)
print(f"Brother in PLANK:")
print(f"  Solid geoms X min: {solids_plank[:, 0].min():.4f} m")
print(f"  Solid geoms X max (head): {solids_plank[:, 0].max():.4f} m")
