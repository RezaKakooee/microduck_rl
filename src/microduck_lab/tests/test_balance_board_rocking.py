"""Visible periodic motion, not merely surviving on the board."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microduck_lab.tasks.balance_board import rocking


class RockingMotionMetricTests(unittest.TestCase):
    motion = dict(amplitude_deg=8.0, frequency_hz=1.2, ramp_seconds=3.0)

    def trajectory(self, mode):
        t = np.arange(.02, 20.001, .02)
        wave = np.cos(2 * np.pi * self.motion["frequency_hz"] * t)
        if mode == "still":
            wave[:] = 0
        elif mode == "single_push":
            wave[t > 3] = 0
        elif mode == "stops_late":
            wave[t > 15] = 0
        trajectory = np.zeros((len(t), 8))
        trajectory[:, 0] = t
        trajectory[:, 1] = np.radians(6.7) * wave
        trajectory[:, 4] = .012 * wave
        return trajectory

    def test_rejects_the_users_original_failure_and_late_freezing(self):
        for mode in ("still", "single_push", "stops_late"):
            with self.subTest(mode=mode):
                result = rocking.score_motion(self.trajectory(mode), self.motion, 20)
                self.assertFalse(result["motion_success"])

    def test_accepts_continuous_visible_rocking(self):
        result = rocking.score_motion(self.trajectory("continuous"), self.motion, 20)
        self.assertTrue(result["motion_success"])
        self.assertEqual(result["completed_rocking_cycles"], 20)


class RockingPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = rocking.lqr.tb.Board("A")
        rocking.lqr.prepare(cls.board)
        cls.expert = rocking.design(cls.board)

    def assert_rocks(self, result):
        self.assertTrue(result["success"], result)
        self.assertTrue(result["balance_success"])
        self.assertEqual(result["completed_rocking_cycles"], result["expected_cycles"])
        self.assertEqual(result["plank_floor_contacts"], 0)
        self.assertEqual(result["bad_body_contacts"], 0)
        self.assertEqual(result["max_foot_airtime_s"], 0)
        self.assertGreater(result["min_plank_floor_clearance_mm"], 40)
        self.assertGreater(result["max_plank_tilt_deg"], 6)
        self.assertGreaterEqual(min(c["roller_travel_mm"] for c in result["cycles"]), 5)

    def test_rocks_throughout_a_long_rollout(self):
        result = rocking.evaluate(self.expert, seconds=30)
        self.assert_rocks(result)
        self.assertEqual(result["completed_rocking_cycles"], 32)
        self.assertGreater(result["cycles"][-1]["start_s"], 28)

    def test_saved_policy_starts_a_fresh_motion_each_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rocking.npz"
            self.expert.save(path)
            loaded = rocking.RockingExpert.load(path, self.board)
            first = rocking.evaluate(loaded, seconds=10)
            second = rocking.evaluate(loaded, seconds=10)
            self.assert_rocks(first)
            self.assert_rocks(second)
            self.assertEqual(first["cycles"], second["cycles"])

    def test_small_push_does_not_stop_the_rocking(self):
        result = rocking.evaluate(self.expert, seconds=20, push=.01, push_at=5.2)
        self.assertTrue(result["push_applied"])
        self.assert_rocks(result)


if __name__ == "__main__":
    unittest.main()
