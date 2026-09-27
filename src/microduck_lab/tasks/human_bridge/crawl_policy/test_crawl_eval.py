import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.crawl.expert import evaluate as eval_expert
from microduck_lab.tasks.crawl.baby import evaluate as eval_baby

p_expert = np.load("local_storage/gaits/crawl_gait.npy")
p_baby = np.load("local_storage/gaits/crawl_baby_gait.npy")

print("--- Testing Expert Crawl Gait (5s) ---")
res_expert = eval_expert(p_expert, seconds=5.0)
print(f"Success: {res_expert['success']}, Travelled: {res_expert['travelled_mm']:.1f} mm, Speed: {res_expert['speed_mm_s']:.1f} mm/s")

print("--- Testing Baby Crawl Gait (5s) ---")
res_baby = eval_baby(p_baby, seconds=5.0)
print(f"Success: {res_baby['success']}, Travelled: {res_baby['travelled_mm']:.1f} mm, Speed: {res_baby['speed_mm_s']:.1f} mm/s")
