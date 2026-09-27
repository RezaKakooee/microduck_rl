"""Inspect every contact on Sister from t=5.5 to 6.5s in Story."""
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
while story.world.t < 5.8:
    story.tick()

he, she = story.he, story.she
m, d = story.world.model, story.world.data

print("Stepping through transition (t=5.8 to 6.5s):")
for step in range(35): # 0.7s
    story.tick()
    t = story.world.t
    pos = she.pos()
    
    # List all bodies in contact with Sister
    con_list = []
    for c_i in range(d.ncon):
        c = d.contact[c_i]
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if b1.startswith('she_') or b2.startswith('she_'):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c_i, f6)
            if f6[0] > 0.1:
                my_b = b1 if b1.startswith('she_') else b2
                other_b = b2 if b1.startswith('she_') else b1
                other_g = g2 if b1.startswith('she_') else g1
                con_list.append((my_b, other_b, other_g, round(f6[0], 2), round(float(c.pos[0]), 3)))

    if step % 5 == 0:
        print(f"\nt={t:.2f}s pos={np.round(pos, 3)}")
        for c_info in con_list:
            print(f"   {c_info[0]} <-> {c_info[1]}/{c_info[2]} fn={c_info[3]}N x={c_info[4]}")
