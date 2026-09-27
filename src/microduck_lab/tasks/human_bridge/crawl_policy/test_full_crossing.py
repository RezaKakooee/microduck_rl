"""Full crossing simulation test with detailed logging across the entire bridge."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

story = Story(verbose=True)

print(f"Goal: rearmost(she) > {L.far_edge + 0.005:.3f} m (far ledge starts at {L.far_edge:.3f} m)")
print("Running full crossing test (25 seconds)...")

try:
    for i in range(1250): # 25 seconds @ 50 Hz (0.02s dt)
        story.tick()
        t = story.world.t
        if i % 25 == 0:
            pos = story.she.pos()
            rear = rearmost(story.she)
            he_up = story.he.up()
            r_roll = float(story.she.data.xmat[story.she.trunk, 7])
            print(f"t={t:5.2f}s phase={story.phase:7s} she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f} he_up={he_up:.3f}")
        if story.phase == "across" and story.settle_time >= 3.0:
            print(f">>> Reached across and settled at t={t:.2f}s!")
            break
except Exception as e:
    print(f"Halted at t={story.world.t:.2f}s: {e}")

rear_final = rearmost(story.she)
print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, current_x = {story.she.pos()[0]:.3f} m, rear = {rear_final:.3f} m")
print(f"Phase = {story.phase}, Success = {story.failure is None and story.phase == 'across' and rear_final > L.far_edge + 0.005}")
