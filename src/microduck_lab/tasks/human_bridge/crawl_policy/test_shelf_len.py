"""Test different shelf_len values to see how Brother settles and where Sister reaches."""
import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.scene import Layout
from microduck_lab.tasks.human_bridge.crawl_policy.world import World
from microduck_lab.tasks.human_bridge.crawl_policy.brother import LieDown, HoldStraight

from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK
from microduck_lab.tasks.human_bridge.crawl_policy.story import STAND_POSE

def smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)

for slen in [0.060, 0.075, 0.090, 0.105, 0.120, 0.150]:
    layout = Layout(shelf_len=slen, step_len=0.052)
    w = World(layout.design())
    he = w.ducks['he']
    he.stand_on(layout.gap_start - 0.005, 0.0, layout.step_z, STAND_POSE)
    w.start()
    he.hold(STAND_POSE)
    for _ in range(75): # 1.5s wait
        w.step()
        
    start = he.target.copy()
    for step_i in range(175): # 3.5s lie
        u = step_i * 0.02
        lie_progress = smooth(u / 2.0)
        he.hold((1.0 - lie_progress) * start + lie_progress * PLANK)
        w.step()
        
    hold = HoldStraight(he)
    for _ in range(75): # 1.5s hold
        hold()
        w.step()
        
    import mujoco
    mouth_body = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, 'he_mouth_jaw')
    print(f"shelf_len={slen:.3f} -> far_edge={layout.far_edge:.3f}: he_up={he.up():+.3f}, head_x={w.data.xpos[mouth_body, 0]:+.3f}, head_z={w.data.xpos[mouth_body, 2]:.3f}")
