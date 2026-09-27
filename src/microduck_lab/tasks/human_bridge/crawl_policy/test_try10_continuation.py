"""Run the exact crawl_try10 configuration beyond t=10.76s with the fixed drop threshold."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.scene import Layout
from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.controller import SisterScriptedController
from microduck_lab.tasks.human_bridge.crawl_policy.story import STAND_POSE

# Layout with shelf_len=0.150 (the exact layout in crawl_try10)
layout = Layout(shelf_len=0.150, step_len=0.052)
story = Story(verbose=False)
story.world = w = World(layout.design())
story.he, story.she = w.ducks["he"], w.ducks["she"]
story.controller = SisterScriptedController(story.she)

story.he.stand_on(layout.gap_start - 0.005, 0.0, layout.step_z, STAND_POSE)

from scipy.spatial.transform import Rotation as R
rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
q_xyzw = rot.as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
story.she.place((-0.22, 0.0, layout.near_z + 0.05), q=story.controller.target)
story.she.data.qpos[story.she.adr + 3 : story.she.adr + 7] = quat_wxyz
w.start()

# Check with the fixed threshold
def check(self):
    she, he = self.she, self.he
    r_roll = abs(float(she.data.xmat[she.trunk, 7]))
    if r_roll > 1.20:
        raise RuntimeError(f"she rolled onto side at x={she.pos()[0]:.3f} m, roll={r_roll:.3f}")
    if abs(she.pos()[1]) > 0.18:
        raise RuntimeError(f"she drifted off bridge sideways: y={she.pos()[1]:.3f} m")
    if self.phase in ("cross", "across") and she.pos()[2] < layout.shelf_z - 0.05:
        raise RuntimeError(f"she dropped into gap at x={she.pos()[0]:.3f}, z={she.pos()[2]:.3f} m")
    if self.phase in ("cross", "across") and abs(he.up()) > 0.35:
        raise RuntimeError(f"brother was knocked loose: he.up() = {he.up():.3f}")

story.check = check.__get__(story)

m, d = w.model, w.data
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

print("Starting simulation with exact crawl_try10 controller...")
for i in range(1250): # 25 seconds
    try:
        story.tick()
    except RuntimeError as e:
        print(f"Halted at t={story.world.t:.2f}s: {e}")
        break
        
    if i % 50 == 0:
        pos = story.she.pos()
        rear = rearmost(story.she)
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        r_roll = float(story.she.data.xmat[story.she.trunk, 7])
        he_up = story.he.up()
        print(f"t={story.world.t:5.2f}s phase={story.phase:7s} she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} LF_x={lf[0]:+.3f} RF_x={rf[0]:+.3f} roll={r_roll:+.2f} he_up={he_up:.3f}")
        
    if story.phase == "across" and story.settle_time >= 3.0:
        print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
        break

rear_final = rearmost(story.she)
print(f"Result: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")
