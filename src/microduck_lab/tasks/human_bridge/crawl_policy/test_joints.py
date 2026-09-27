"""Check joint indices and right leg joint ranges."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

w = World(L.design())
she = w.ducks['she']
m, d = w.model, w.data
J = {name: i for i, name in enumerate(SERVOS)}
print("SERVOS:", SERVOS)
for name in SERVOS:
    idx = J[name]
    print(f"{name:20s}: idx={idx} lo={she.lo[idx]:.3f} hi={she.hi[idx]:.3f}")
