"""Inspect Brother's contacts when lying down with different shelf_len."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.scene import Layout
from microduck_lab.tasks.human_bridge.crawl_policy.world import World
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK
from microduck_lab.tasks.human_bridge.crawl_policy.story import STAND_POSE, smooth

for slen in [0.075, 0.085, 0.095, 0.105, 0.120, 0.150]:
    layout = Layout(shelf_len=slen, step_len=0.052)
    w = World(layout.design())
    he = w.ducks['he']
    he.stand_on(layout.gap_start - 0.005, 0.0, layout.step_z, STAND_POSE)
    w.start()
    he.hold(STAND_POSE)
    for _ in range(75): w.step()
    
    start = he.target.copy()
    for step_i in range(175):
        u = step_i * 0.02
        lie_progress = smooth(u / 2.0)
        he.hold((1.0 - lie_progress) * start + lie_progress * PLANK)
        w.step()
        
    m, d = w.model, w.data
    head_contacts = []
    for c_i in range(d.ncon):
        con = d.contact[c_i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        if 'he_' in str(g1) or 'he_' in str(g2):
            if 'shelf' in str(g1) or 'shelf' in str(g2) or 'far_top' in str(g1) or 'far_top' in str(g2):
                head_contacts.append(f"{g1} touching {g2} at x={con.pos[0]:.3f}, z={con.pos[2]:.3f}")
    print(f"\nshelf_len={slen:.3f} (far_edge={layout.far_edge:.3f}): he_up={he.up():+.3f}")
    for hc in set(head_contacts):
        print(f"   {hc}")
