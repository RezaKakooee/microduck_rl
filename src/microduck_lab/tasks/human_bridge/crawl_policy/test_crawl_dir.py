import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.crawl.crawl import Crawl, DT
from microduck_lab.tasks.crawl.expert import gait

p = np.load('local_storage/gaits/crawl_gait.npy')
c = Crawl()
for _ in range(int(3.0 / DT)):
    c.step(gait(c, p, c.t))

dist, dx, dy = c.travelled()
print(f"In crawl.py after 3.0s: dist={dist:.3f}m, dx={dx:+.3f}m, dy={dy:+.3f}m")
print(f"Trunk xpos: start={np.round(c.start, 3)}, current={np.round(c.data.xpos[c.trunk], 3)}")
