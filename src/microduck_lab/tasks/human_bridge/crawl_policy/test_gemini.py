import os
import sys
from pathlib import Path
import unittest
import math
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.world import World, SERVOS, Brain
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.brother import PLANK, LieDown, HoldStraight


class TestSetup(unittest.TestCase):
    def test_world_init(self):
        w = World(L.design())
        self.assertIn("he", w.ducks)
        self.assertIn("she", w.ducks)
        he, she = w.ducks["he"], w.ducks["she"]
        self.assertEqual(len(he.acts), 14)
        self.assertEqual(len(she.acts), 14)
        print("World initialized successfully with 'he' and 'she'.")

    def test_brother_plank(self):
        w = World(L.design())
        he = w.ducks["he"]
        bh = Brain(he, stand_only=True)
        he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, bh.default_pose)
        w.start()
        for _ in range(50):
            bh.act()
            w.step()
        lie = LieDown(he, he.target, w.t)
        for _ in range(150):
            lie(w.t)
            w.step()
    def test_sister_stand(self):
        w = World(L.design())
        she = w.ducks["she"]
        from microduck_lab.tasks.human_bridge.crawl_policy.controller import STAND_POSE, TARGET_HEADING
        from microduck_lab.tasks.human_bridge.crawl_policy.story import START_X

        # Base pose: head more upright to center COM over feet
        pose0 = STAND_POSE.copy()
        pose0[5] = 0.05   # neck_pitch (was 0.3491)
        pose0[6] = 0.05   # head_pitch (was 0.3491)
        # Shift ankle slightly back to lean COM onto center of foot
        pose0[4] += 0.03  # left_ankle
        pose0[13] -= 0.03 # right_ankle

        she.place((START_X, 0.0, L.near_z + 0.13), yaw=TARGET_HEADING, q=pose0)
        soles = she.points(she.sole_geoms)
        p = she.pos()
        p[2] += L.near_z + 0.0005 - soles[:, 2].min()
        she.place(p, yaw=TARGET_HEADING, q=pose0)
        soles = she.points(she.sole_geoms)
        com = she.com()
        print(f"Soles X range: [{soles[:, 0].min():.4f}, {soles[:, 0].max():.4f}], CoM X: {com[0]:.4f}")
        print(f"Soles Y range: [{soles[:, 1].min():.4f}, {soles[:, 1].max():.4f}], CoM Y: {com[1]:.4f}")
        for s, lo, hi, q0 in zip(SERVOS, she.lo, she.hi, pose0):
            print(f"{s:18s} lo={lo:+.3f} hi={hi:+.3f} pose0={q0:+.3f}")
        w.start()

        for i in range(500):
            q = pose0.copy()
            R = she.R()
            # pitch_angle: positive means tilted backward (+y)
            pitch_angle = -float(R[2, 0])
            w_pitch = float(she.data.qvel[she.dof + 4])

            # Strong ankle PD stabilization:
            # Also add tiny integral or position term on y to keep y strictly at 0.000
            y_pos = she.pos()[1]
            target_pitch = np.clip(-1.5 * y_pos, -0.05, 0.05)
            err_pitch = pitch_angle - target_pitch

            u = 2.5 * err_pitch + 0.3 * w_pitch
            q[4] += u
            q[13] -= u

            she.hold(q)
            w.step()
            if i % 100 == 0:
                print(f"step {i:3d} t={w.t:.2f}s up={she.up():.3f} y={she.pos()[1]:.4f} pitch={pitch_angle:+.4f} q4={q[4]:.3f}")

        print(f"Final t={w.t:.2f}s up={she.up():.3f} pos={np.round(she.pos(), 4)}")
        self.assertGreater(she.up(), 0.95)

    def test_sister_step(self):
        w = World(L.design())
        she = w.ducks["she"]
        from microduck_lab.tasks.human_bridge.crawl_policy.controller import STAND_POSE, TARGET_HEADING
        from microduck_lab.tasks.human_bridge.crawl_policy.story import START_X

        pose0 = STAND_POSE.copy()
        pose0[5] = 0.05   # neck_pitch
        pose0[6] = 0.05   # head_pitch
        pose0[4] += 0.03  # left_ankle
        pose0[13] -= 0.03 # right_ankle

        she.place((START_X, 0.0, L.near_z + 0.13), yaw=TARGET_HEADING, q=pose0)
        soles = she.points(she.sole_geoms)
        p = she.pos()
        p[2] += L.near_z + 0.0005 - soles[:, 2].min()
        she.place(p, yaw=TARGET_HEADING, q=pose0)
        w.start()

        # Step parameters
        period = 1.10  # seconds per 2-step cycle
        sway_amp = 0.22 # rad (shifts CoM by ~40mm over stance foot)
        step_amp = 0.15
        lift_amp = 0.40

        for i in range(250):
            t = w.t
            q = pose0.copy()

            # Gait cycle phi in [0, 2*pi)
            phi = (2.0 * math.pi * (t / period)) % (2.0 * math.pi)

            # Terrain awareness: extra knee lift over brother's ankle servos
            x_pos = she.pos()[0]
            extra_lift = 0.08 if (-0.12 <= x_pos <= 0.06) else 0.0
            lift = lift_amp + extra_lift

            if phi < math.pi:
                # Half-cycle 1: Left leg swing, Right leg stance
                u = phi / math.pi
                sway = sway_amp * math.sin(u * math.pi)
                # Left swing: lift and reach +x (more negative)
                h_L = lift * math.sin(u * math.pi)
                q[3] += -h_L
                q[4] += 0.35 * h_L
                q[1] += -step_amp * u + sway
                # Right stance: push trunk forward (more positive)
                q[10] += step_amp * u + sway
            else:
                # Half-cycle 2: Right leg swing, Left leg stance
                u = (phi - math.pi) / math.pi
                sway = -sway_amp * math.sin(u * math.pi)
                # Right swing: lift and pull +x (more negative / less positive)
                h_R = lift * math.sin(u * math.pi)
                q[12] += h_R
                q[13] += -0.35 * h_R
                q[10] += -step_amp * (1.0 - u) + sway
                # Left stance: push trunk forward (more positive / less negative)
                q[1] += -step_amp * (1.0 - u) + sway

            # Pitch feedback
            R = she.R()
            pitch_angle = -float(R[2, 0])
            w_pitch = float(she.data.qvel[she.dof + 4])
            y_pos = she.pos()[1]
            target_pitch = np.clip(-1.5 * y_pos, -0.06, 0.06)
            err_pitch = pitch_angle - target_pitch
            u_pitch = 2.5 * err_pitch + 0.3 * w_pitch

            q[4] += u_pitch
            q[13] -= u_pitch

            # Yaw steering
            yaw_err = math.atan2(math.sin(she.yaw() - TARGET_HEADING), math.cos(she.yaw() - TARGET_HEADING))
            w_yaw = float(she.data.qvel[she.dof + 5])
            u_yaw = np.clip(-0.5 * yaw_err - 0.05 * w_yaw, -0.15, 0.15)
            q[0] += u_yaw
            q[9] += u_yaw

            she.hold(q)
            w.step()
            feet = she.feet()
            if i % 10 == 0:
                print(f"t={w.t:.2f}s x={she.pos()[0]:.4f} zL={feet[0][2]:.4f} zR={feet[1][2]:.4f} up={she.up():.3f}")

        print(f"Final t={w.t:.2f}s x={she.pos()[0]:.4f} y={she.pos()[1]:.4f} up={she.up():.3f}")
        self.assertGreater(she.pos()[0], START_X + 0.05)
        self.assertGreater(she.up(), 0.8)


if __name__ == "__main__":
    unittest.main()
