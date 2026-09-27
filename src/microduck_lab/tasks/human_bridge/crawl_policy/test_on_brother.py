"""Inspect contacts and forces when Sister is on Brother at x=-0.023m."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

story = Story(verbose=False)
she, he = story.she, story.he
m, d = story.world.model, story.world.data
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# Fast forward to t=8.0s
while story.world.t < 8.0:
    story.tick()

print(f"Inspecting Sister on Brother starting at t={story.world.t:.2f}s:")
for step in range(100): # 2 seconds
    story.tick()
    t = story.world.t
    if step % 10 == 0:
        pos = she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        
        # Check all contacts on Sister
        con_list = []
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
            if b1.startswith('she_') or b2.startswith('she_'):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, c_i, f6)
                if f6[0] > 0.5:
                    she_b = b1 if b1.startswith('she_') else b2
                    other_b = b2 if b1.startswith('she_') else b1
                    con_list.append(f"{she_b[:10]}<->{other_b[:10]}(fn={f6[0]:.1f}N,x={c.pos[0]:.2f},y={c.pos[1]:.2f},z={c.pos[2]:.2f})")
                    
        print(f"\nt={t:5.2f}s pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) LF=({lf[0]:+.3f},{lf[1]:+.3f},{lf[2]:.3f}) RF=({rf[0]:+.3f},{rf[1]:+.3f},{rf[2]:.3f})")
        print(f"  Contacts ({len(con_list)}): {', '.join(con_list)}")
