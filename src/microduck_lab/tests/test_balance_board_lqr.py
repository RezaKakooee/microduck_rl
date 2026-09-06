"""CPU rollouts: a real free board, real contact failures and bounded servos.

Run without extra packages:
  OPENBLAS_NUM_THREADS=1 uv run --with pytest pytest src/microduck_lab/tests/test_balance_board_lqr.py -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microduck_lab.tasks.balance_board import lqr


class BalanceBoardExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = lqr.tb.Board("A")
        lqr.prepare(cls.board)
        cls.expert = lqr.design(cls.board)

    def assert_balanced(self, result, seconds=10):
        self.assertTrue(result["success"], result)
        self.assertEqual(result["held"], seconds)
        self.assertEqual(result["plank_floor_contacts"], 0)
        self.assertEqual(result["foot_floor_contacts"], 0)
        self.assertEqual(result["bad_body_contacts"], 0)
        self.assertEqual(result["roller_contact_loss_s"], 0)
        self.assertGreater(result["min_plank_floor_clearance_mm"], 40)
        self.assertLessEqual(result["max_foot_airtime_s"], lqr.tb.FOOT_GRACE_S)
        self.assertFalse(result["configuration"]["fixed_roller"])
        self.assertEqual(result["configuration"]["radius"], .03)
        self.assertAlmostEqual(result["configuration"]["plank_mass"], .07)

    def test_original_geometry_balances_ten_seconds(self):
        self.assert_balanced(lqr.evaluate(self.expert))

    def test_recovers_from_both_push_directions(self):
        for push in (-.05, .05):
            with self.subTest(push=push):
                result = lqr.evaluate(self.expert, push=push)
                self.assertTrue(result["push_applied"])
                self.assert_balanced(result)

    def test_unseen_random_start(self):
        self.assert_balanced(lqr.evaluate(self.expert, seed=123))

    def test_constant_targets_fail_on_the_same_board(self):
        expert = self.expert
        passive = lqr.LQRExpert(self.board.model, expert.q0, expert.u0,
                                np.zeros_like(expert.gain), expert.config)
        result = lqr.evaluate(passive)
        self.assertFalse(result["success"])
        self.assertLess(result["held"], 3.0)
        self.assertEqual(result["fail"], "plank touched floor")

    def test_saved_policy_replays_without_redesign(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "expert.npz"
            self.expert.save(path)
            loaded = lqr.LQRExpert.load(path, self.board)
            self.assert_balanced(lqr.evaluate(loaded, push=.02))

    def test_controller_only_writes_joint_targets(self):
        b = self.board
        qpos, qvel = b.data.qpos.copy(), b.data.qvel.copy()
        mass, limits = b.model.body_mass.copy(), b.model.actuator_forcerange.copy()
        command, _ = self.expert({"qpos": b.data.qpos, "qvel": b.data.qvel}, lqr.DT)
        self.assertEqual(command.shape, (14,))
        self.assertTrue(np.isfinite(command).all())
        np.testing.assert_array_equal(b.data.qpos, qpos)
        np.testing.assert_array_equal(b.data.qvel, qvel)
        np.testing.assert_array_equal(b.model.body_mass, mass)
        np.testing.assert_array_equal(b.model.actuator_forcerange, limits)


if __name__ == "__main__":
    unittest.main()
