"""The baby gait must make sustained progress with both physical legs together."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from microduck_lab.tasks.crawl.baby import PARAMS, REFERENCE, evaluate
from microduck_lab.tasks.crawl.crawl import Crawl


class BabyCrawlTests(unittest.TestCase):
    def test_selected_gait_moves_for_the_full_clip_with_synchronized_legs(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'trace.npz'
            r = evaluate(np.load(PARAMS), seconds=20., trace=trace)
            self.assertTrue(r['success'], r)
            self.assertGreater(r['travelled_mm'], 250.)
            self.assertGreater(r['second_half_speed_mm_s'], 10.)
            self.assertLess(r['trunk_height_mm']['max'], 200.)
            self.assertLess(r['trunk_height_mm']['mean'], 100.)
            self.assertLessEqual(r['maximum_servo_torque_nm'], r['servo_torque_limit_nm'])
            a = np.load(trace)
            names = a['joint_names'].tolist()
            joints = a['joint_positions'][50:]
            for joint in ('hip_pitch', 'knee'):
                left = joints[:, names.index('he_left_' + joint)]
                right = -joints[:, names.index('he_right_' + joint)]
                self.assertGreater(np.corrcoef(left, right)[0, 1], .9, joint)
                self.assertGreater(np.ptp(left), .3, joint)

    def test_height_and_standing_are_rejected_even_before_the_final_state(self):
        for height, quat, message in [(.21, [2**-.5, 0, 2**-.5, 0], '200 mm'),
                                      (.10, [1, 0, 0, 0], 'upright')]:
            c = Crawl()
            # Deliberately inject invalid states to exercise the evaluator gate.
            c.data.qpos[c.root + 2] = height
            c.data.qpos[c.root + 3:c.root + 7] = quat
            mujoco.mj_forward(c.model, c.data)
            with patch('microduck_lab.tasks.crawl.baby.Crawl', return_value=c):
                r = evaluate(REFERENCE, seconds=.02)
            self.assertFalse(r['success'])
            self.assertIn(message, r['failure'])


if __name__ == '__main__':
    unittest.main()
