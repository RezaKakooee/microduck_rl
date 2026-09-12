"""Unit tests for the suspension bridge walking task."""
import unittest
import os
import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.tasks.bridge.bridge import bridge_xml, write_scene, run, evaluate


class BridgeWalkingTests(unittest.TestCase):
    def test_bridge_xml_compilation_and_geoms(self):
        xml_str = bridge_xml(start_x=-0.25, end_x=1.75, sag=0.04)
        scene_file = write_scene(xml_str)
        try:
            model, data = duck_sim.load_scene(scene_file)
            self.assertGreater(model.ngeom, 1000)
            self.assertEqual(model.nmocap, 0)
            self.assertEqual(model.neq, 0)

            # Check that bridge anchors and deck bodies exist
            anchor_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bridge_anchors")
            deck_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bridge_deck")
            self.assertGreaterEqual(anchor_b, 0)
            self.assertGreaterEqual(deck_b, 0)

            # Check collisions enabled
            self.assertTrue(np.any(model.geom_contype > 0))
            self.assertTrue(np.any(model.geom_conaffinity > 0))
        finally:
            if os.path.exists(scene_file):
                os.remove(scene_file)

    def test_bridge_crossing_standard_sag(self):
        res = evaluate(sag=0.04, sway=False)
        self.assertTrue(res["success"], f"Crossing failed: {res.get('fail_reason')}")
        self.assertGreater(res["distance_m"], 1.60)
        self.assertIsNotNone(res["cross_time_s"])
        self.assertLess(res["cross_time_s"], 20.0)
        self.assertLess(res["max_lat_error_mm"], 50.0)
        self.assertIsNone(res["fail_reason"])

    def test_bridge_crossing_compliant_sway(self):
        res = evaluate(sag=0.03, sway=True)
        self.assertTrue(res["success"], f"Sway crossing failed: {res.get('fail_reason')}")
        self.assertGreater(res["distance_m"], 1.60)
        self.assertIsNotNone(res["cross_time_s"])


if __name__ == "__main__":
    unittest.main()
