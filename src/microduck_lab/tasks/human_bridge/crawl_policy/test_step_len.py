"""Test different step_len values to see Brother's lie-down stability and Sister's crossing."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.scene import Layout
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK, LieDown, HoldStraight
from microduck_lab.tasks.human_bridge.crawl_policy.story import STAND_POSE, START_X, CRAWL_BASE

for test_len in [0.076, 0.060, 0.050, 0.045, 0.040]:
    L = Layout(step_len=test_len)
    w = World(L.design())
    he, she = w.ducks['he'], w.ducks['she']
    
    # Stand Brother
    he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, STAND_POSE)
    w.start()
    
    # 1.5s wait in stand
    stable_stand = True
    for _ in range(75):
        he.hold(STAND_POSE)
        w.step()
        if abs(he.up() - 1.0) > 0.15:
            stable_stand = False
            break
            
    # Lie down
    lie = LieDown(he, he.target, w.t)
    stable_lie = True
    for _ in range(175): # 3.5s
        lie(w.t)
        w.step()
        
    hold = HoldStraight(he)
    for _ in range(50): # 1.0s settle
        hold()
        w.step()
        
    solids_he = he.points(he.solid)
    he_up = he.up()
    print(f"step_len={test_len*1000:4.1f}mm (near_edge={L.near_edge*1000:+5.1f}mm):")
    print(f"  stand_ok={stable_stand}, lie_ok={abs(he_up) < 0.10}, he_up={he_up:.3f}")
    print(f"  Brother rearmost solid: {solids_he[:, 0].min()*1000:+5.1f}mm")
    gap = solids_he[:, 0].min() - L.near_edge
    print(f"  Gap between near_edge and Brother: {gap*1000:4.1f}mm")
