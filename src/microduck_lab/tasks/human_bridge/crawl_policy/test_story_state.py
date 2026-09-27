"""Inspect Brother and Sister state in Story at t=7.5s."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

story = Story(verbose=False)
while story.world.t < 7.5:
    story.tick()

he, she = story.he, story.she
m, d = story.world.model, story.world.data

print(f"At t={story.world.t:.2f}s:")
print(f"  Sister trunk pos: {np.round(she.pos(), 3)}")
print(f"  Brother trunk pos: {np.round(he.pos(), 3)}")
print(f"  Brother head pos: {np.round(d.xpos[he.head], 3)}")

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')
print(f"  Sister left foot: {np.round(d.xpos[foot_l], 3)}")
print(f"  Sister right foot: {np.round(d.xpos[foot_r], 3)}")

he_foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'he_ankle_left')
he_foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'he_ankle_right')
print(f"  Brother left foot: {np.round(d.xpos[he_foot_l], 3)}")
print(f"  Brother right foot: {np.round(d.xpos[he_foot_r], 3)}")

print("\nContacts:")
for c_i in range(d.ncon):
    c = d.contact[c_i]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
    f6 = np.zeros(6)
    mujoco.mj_contactForce(m, d, c_i, f6)
    if f6[0] > 0.1:
        print(f"  {b1}/{g1} <-> {b2}/{g2}: pos={np.round(c.pos, 3)}, fn={f6[0]:.2f}N")
