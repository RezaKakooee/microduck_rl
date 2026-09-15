"""Unit tests for the soft dynamic suspension bridge walking task."""
import unittest
import os
import numpy as np
import mujoco

from microduck_lab import paths
from microduck_lab.sim import duck_sim
from microduck_lab.tasks.bridge.bridge import bridge_xml, write_scene, run, evaluate


class BridgeWalkingTests(unittest.TestCase):
    def test_bridge_xml_compilation_and_geoms(self):
        xml_str = bridge_xml(start_x=0.0, length=2.0, sag=0.035, num_rungs=24, center_kz=160.0)
        scene_file = write_scene(xml_str)
        try:
            model, data = duck_sim.load_scene(scene_file)
            self.assertGreater(model.ngeom, 200)
            self.assertEqual(model.nmocap, 0)
            self.assertEqual(model.neq, 0)

            # Check that fixed anchors body exists
            anchor_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bridge_anchors")
            self.assertGreaterEqual(anchor_b, 0)

            # Check that articulated compliant rungs exist
            for r in [0, 11, 23]:
                rung_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"soft_rung_{r}")
                self.assertGreaterEqual(rung_b, 0, f"Rung {r} body not found")

                # Check 3-axis compliance joints
                j_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_z_{r}")
                j_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_pitch_{r}")
                j_roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_roll_{r}")
                self.assertGreaterEqual(j_z, 0)
                self.assertGreaterEqual(j_pitch, 0)
                self.assertGreaterEqual(j_roll, 0)

            # Articulated rungs must contribute active DOFs to the simulation model
            self.assertGreater(model.nv, 70)

            # Check collisions enabled
            self.assertTrue(np.any(model.geom_contype > 0))
            self.assertTrue(np.any(model.geom_conaffinity > 0))
        finally:
            if os.path.exists(scene_file):
                os.remove(scene_file)

    def test_bridge_crossing_standard(self):
        res = evaluate(center_kz=180.0, sag=0.035, speed=0.42)
        self.assertTrue(res["success"], f"Crossing failed: {res.get('fail_reason')}")
        self.assertGreater(res["distance_m"], 1.80)
        self.assertIsNotNone(res["cross_time_s"])
        self.assertLess(res["cross_time_s"], 20.0)
        self.assertIsNone(res["fail_reason"])
        # Verify dynamic physical deflection under duck's weight
        self.assertGreater(res["max_sink_mm"], 15.0, "Bridge did not deflect under robot weight")

    def test_bridge_crossing_soft_net(self):
        res = evaluate(center_kz=160.0, sag=0.035, speed=0.42)
        self.assertTrue(res["success"], f"Soft net crossing failed: {res.get('fail_reason')}")
        self.assertGreater(res["distance_m"], 1.80)
        self.assertIsNotNone(res["cross_time_s"])
        # Verifies dynamic deflection on soft rope net ("طناب ساده")
        self.assertGreater(res["max_sink_mm"], 15.0, "Soft bridge should sink > 15 mm under robot weight")


if __name__ == "__main__":
    unittest.main()
