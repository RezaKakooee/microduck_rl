"""Test story with dx=0.07 (start_x = -0.15m)."""
import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

story = Story(dx=0.07, verbose=True)

print("Running story simulation with dx=0.07 (start_x = -0.15m)...")
try:
    for i in range(600): # 12 seconds
        story.tick()
        if i % 25 == 0:
            pos = story.she.pos()
            he_up = story.he.up()
            print(f"t={story.world.t:5.2f}s phase={story.phase:8s} | she_pos={np.round(pos, 3)} furthest_x={story.furthest_x:+.3f} | he_up={he_up:.3f}")
except Exception as e:
    print(f"Exception at t={story.world.t:.2f}s: {e}")

pos_final = story.she.pos()
print(f"\nSimulation ended at t={story.world.t:.2f}s")
print(f"Final Sister pos: {np.round(pos_final, 3)}")
print(f"Furthest x: {story.furthest_x:.3f} m (Far ledge starts at {L.far_edge:.3f} m)")
