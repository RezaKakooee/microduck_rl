"""Measure foot trajectories and contact events as Sister moves across Brother's back."""
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

layout = Layout(shelf_len=0.110, step_len=0.052)
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

def check(self):
    pass # disable early termination to observe full motion

story.check = check.__get__(story)

m, d = w.model, w.data
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

# Run until cross phase starts
while story.world.t < 5.02:
    story.tick()

print(f"Phase cross started at t={story.world.t:.2f}s. Monitoring foot trajectories...")
for step in range(500): # 10 seconds of crawl
    story.tick()
    t = story.world.t
    if step % 25 == 0:
        pos = story.she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        rear = rearmost(story.she)
        print(f"t={t:5.2f}s she_x={pos[0]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} LF=[{lf[0]:+.3f}, {lf[1]:+.3f}, {lf[2]:.3f}] RF=[{rf[0]:+.3f}, {rf[1]:+.3f}, {rf[2]:.3f}] he_up={story.he.up():+.3f}")

print(f"Finished at t={story.world.t:.2f}s. Final she_pos={story.she.pos()}, furthest_x={story.furthest_x:.3f}")
