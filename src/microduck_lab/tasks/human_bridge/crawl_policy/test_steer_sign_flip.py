"""Test correct negative feedback steering sign: steer_yaw = +kp * y + kd * vy."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

p = np.load('local_storage/gaits/crawl_gait.npy').copy()
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
f = p[-3]
J = {name: i for i, name in enumerate(SERVOS)}

def test_sign(sign: float, kp: float = 1.0, kd: float = 0.15):
    print(f"\n================ Testing sign={sign:+.1f}, kp={kp:.2f} ================")
    story = Story(verbose=False)
    m, d = story.world.model, story.world.data
    foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
    foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

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

        y_pos = float(duck.pos()[1])
        y_vel = float(duck.data.qvel[duck.dof + 1])
        steer_yaw = float(np.clip(sign * (kp * y_pos + kd * y_vel), -0.06, 0.06))

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

        q[J['left_hip_yaw']] = steer_yaw
        q[J['right_hip_yaw']] = steer_yaw
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
    print(f"Result sign={sign:+.1f}: furthest_x = {story.furthest_x:.3f} m, final_pos = {np.round(story.she.pos(), 3)}, rear = {rear_final:.3f} m, phase = {story.phase}")

test_sign(sign=-1.0, kp=1.0)
test_sign(sign=+1.0, kp=1.0)
test_sign(sign=+1.0, kp=1.5)
