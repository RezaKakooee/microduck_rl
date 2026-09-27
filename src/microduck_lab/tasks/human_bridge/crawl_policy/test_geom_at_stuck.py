"""Print exact geometry of Brother and Sister at t=7.5s."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story

story = Story(verbose=False)
while story.world.t < 7.5:
    story.tick()

he, she = story.he, story.she
m, d = story.world.model, story.world.data

print("Sister bodies at t=7.5s:")
for b in sorted(list(she.bodies)):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    pos = d.xpos[b]
    print(f"  {name:20s}: x={pos[0]:+.3f}, y={pos[1]:+.3f}, z={pos[2]:.3f}")

print("\nBrother bodies at t=7.5s:")
for b in sorted(list(he.bodies)):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    pos = d.xpos[b]
    print(f"  {name:20s}: x={pos[0]:+.3f}, y={pos[1]:+.3f}, z={pos[2]:.3f}")
