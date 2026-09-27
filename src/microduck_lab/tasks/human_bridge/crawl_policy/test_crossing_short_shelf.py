"""Test Sister crossing with shortened shelf_len so she steps directly onto far_top."""
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

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
J = {name: i for i, name in enumerate(SERVOS)}

def test_shelf(shelf_len: float):
    print(f"\n================ Testing shelf_len = {shelf_len:.3f} ================")
    layout = Layout(shelf_len=shelf_len, step_len=0.052)
    
    # We can pass layout to Story by subclassing or initializing world with layout
    story = Story(verbose=False)
    # Re-initialize with custom layout
    story.world = w = World(layout.design())
    story.he, story.she = w.ducks["he"], w.ducks["she"]
    story.controller = SisterScriptedController(story.she)
    
    STAND_POSE = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,
    ], dtype=np.float64)
    story.he.stand_on(layout.gap_start - 0.005, 0.0, layout.step_z, STAND_POSE)
    
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    q_xyzw = rot.as_quat()
    quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    story.she.place((-0.22, 0.0, layout.near_z + 0.05), q=story.controller.target)
    story.she.data.qpos[story.she.adr + 3 : story.she.adr + 7] = quat_wxyz
    w.start()
    
    def check(self):
        she, he = self.she, self.he
        if abs(she.pos()[1]) > 0.18:
            raise RuntimeError(f"she drifted off bridge sideways: y={she.pos()[1]:.3f} m")
        if self.phase in ("cross", "across") and she.pos()[2] < layout.shelf_z - 0.05:
            raise RuntimeError(f"she dropped into gap at x={she.pos()[0]:.3f}, z={she.pos()[2]:.3f} m")
        if self.phase in ("cross", "across") and abs(he.up()) > 0.35:
            raise RuntimeError(f"brother was knocked loose: he.up() = {he.up():.3f}")

    story.check = check.__get__(story)

    def compute_targets(self, t: float):
        duck = self.duck
        u_phase = t - self.t_start_phase
        q = self.target.copy()
        if self.phase in ("wait", "he_lies"):
            return q
        elif self.phase == "across":
            q[J['neck_pitch']] = 0.50
            q[J['head_pitch']] = -0.40
            return q

        hp_bias = 0.30
        hr_val = 0.30

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * np.sin(2.0 * np.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                q[J['left_hip_roll']] = -hr_val
                q[J['right_hip_roll']] = hr_val
            elif name == 'hip_pitch':
                q[J['left_' + name]] = (off + hp_bias) + wave
                q[J['right_' + name]] = -((off + hp_bias) - wave)
            elif name == 'knee':
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)
            else:
                q[J['left_' + name]] = off + wave
                q[J['right_' + name]] = -(off - wave)

        q[J['left_hip_yaw']] = 0.0
        q[J['right_hip_yaw']] = 0.0
        q[J['neck_pitch']] = 0.50
        q[J['head_pitch']] = -0.40
        return np.clip(q, duck.lo, duck.hi)

    story.controller.compute_targets = compute_targets.__get__(story.controller)

    for i in range(1250): # 25 seconds
        try:
            story.tick()
        except RuntimeError as e:
            print(f"Halted at t={story.world.t:.2f}s: {e}")
            break
            
        if i % 50 == 0:
            pos = story.she.pos()
            rear = rearmost(story.she)
            r_roll = float(story.she.data.xmat[story.she.trunk, 7])
            he_up = story.he.up()
            print(f"t={story.world.t:5.2f}s phase={story.phase:7s} she_x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} rear={rear:+.3f} furthest_x={story.furthest_x:+.3f} roll={r_roll:+.2f} he_up={he_up:.3f}")
            
        if story.phase == "across" and story.settle_time >= 3.0:
            print(f">>> SUCCESS! Reached across and settled at t={story.world.t:.2f}s!")
            break

    rear_final = rearmost(story.she)
    print(f"Result shelf_len={shelf_len:.3f}: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")

for slen in [0.080, 0.090, 0.100, 0.110]:
    test_shelf(slen)
