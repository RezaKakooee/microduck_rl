import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World, Brain
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

w = World(L.design())
she = w.ducks["she"]
b = Brain(she, stand_only=True)
print("Default pose:", np.round(b.default_pose, 4).tolist())
