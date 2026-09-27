"""Print Brother's body positions when in plank pose."""
import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
s = Story(verbose=False)
for _ in range(250):
    s.tick() # let brother lie down into plank

m, d = s.world.model, s.world.data
print(f"Brother in plank (t={s.world.t:.2f}s, he_up={s.he.up():.3f}):")
for i in range(m.nbody):
    name = m.body(i).name
    if 'he_' in name:
        pos = d.xpos[i]
        print(f"  {name:25s}: x={pos[0]:+.3f}, y={pos[1]:+.3f}, z={pos[2]:.3f}")

print("\nLedge reference points:")
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
print(f"  near_edge = {L.near_edge:+.3f} (near_z={L.near_z:.3f})")
print(f"  gap_start = {L.gap_start:+.3f}")
print(f"  gap_end   = {L.gap_end:+.3f}")
print(f"  far_edge  = {L.far_edge:+.3f} (far_z={L.far_z:.3f}, shelf_z={L.shelf_z:.3f})")
