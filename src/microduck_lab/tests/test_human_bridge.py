"""Human bridge: the set, the brother's bridge, and the physics-only rule.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_human_bridge -v
"""
import unittest

import numpy as np

from microduck_lab.tasks.human_bridge.rl_policy.brother import PLANK, SETTLE_S, LieDown
from microduck_lab.tasks.human_bridge.rl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.rl_policy.world import World, Brain, Box, SetDesign


class Plank(unittest.TestCase):
    """PLANK is what makes his back walkable: flat soles, no neck dip."""

    @classmethod
    def setUpClass(cls):
        cls.w = World(SetDesign(), cast=("he",))
        cls.he = cls.w.ducks["he"]
        cls.he.place((0, 0, 0.2), pitch=np.pi / 2, q=PLANK)

    def test_soles_flat_and_facing_down(self):
        for g in self.he.sole_geoms:
            P = self.he.points([g])
            self.assertLess(np.ptp(P[:, 2]), 0.015)     # 54 mm sole lies flat
            self.assertGreater(np.ptp(P[:, 0]), 0.045)

    def test_back_has_no_deep_dip(self):
        P = self.he.points(self.he.solid)
        top = []
        for x in np.arange(-0.04, 0.12, 0.01):          # trunk, neck, head
            s = P[(P[:, 0] >= x) & (P[:, 0] < x + 0.01) & (np.abs(P[:, 1]) < 0.035)]
            top.append(s[:, 2].max() - 0.2)
        self.assertGreater(min(top), 0.025)             # head straight: -16 mm over the neck


class Bridge(unittest.TestCase):
    """He stands on the near step, lies down, and holds as a bridge."""

    @classmethod
    def setUpClass(cls):
        w = World(L.design(), cast=("he",))
        he = w.ducks["he"]
        brain = Brain(he, stand_only=True)
        he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, brain.default_pose)
        w.start()
        for _ in range(50):
            brain.act()
            w.step()
        lie = LieDown(he, he.target, w.t)
        for _ in range(200):
            lie(w.t)
            w.step()
        cls.w, cls.he = w, he

    def test_he_lies_level(self):
        axis = self.he.R()[:, 2]                         # head direction
        self.assertGreater(axis[0], 0.98)
        self.assertLess(abs(axis[2]), 0.1)

    def test_soles_on_step_and_head_on_shelf(self):
        m = self.w.model
        world_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        touches = self.w.contacts(set(self.he.solid), world_geoms)
        near = [c for c in touches if c[0][0] < L.gap_start]
        far = [c for c in touches if c[0][0] > L.gap_end]
        self.assertTrue(near and far)
        self.assertTrue(all(abs(c[0][2] - L.step_z) < 0.003 for c in near))
        self.assertTrue(all(abs(c[0][2] - L.shelf_z) < 0.003 for c in far))
        self.assertFalse([c for c in touches if L.gap_start < c[0][0] < L.gap_end])


class PhysicsOnly(unittest.TestCase):
    def test_external_force_is_refused(self):
        w = World(SetDesign(), cast=("she",))
        w.start()
        w.data.xfrc_applied[w.ducks["she"].trunk, 2] = 1.0
        with self.assertRaises(RuntimeError):
            w.step()

    def test_place_is_refused_once_running(self):
        w = World(SetDesign(), cast=("she",))
        w.start()
        with self.assertRaises(RuntimeError):
            w.ducks["she"].place((0, 0, 0.2))

    def test_collision_bits_are_guarded(self):
        w = World(SetDesign(), cast=("she",))
        w.start()
        w.model.body_contype[w.ducks["she"].trunk] = 0
        with self.assertRaises(RuntimeError):
            w.step()


class Armed(unittest.TestCase):
    """Every part meant to be solid really collides. Until 2026-09-26 the
    collision bits were set after compile, MuJoCo's per-body filter stayed
    0/0 on the thighs and neck of both ducks, and his neck rested 16.8 mm
    inside the far shelf."""

    def test_every_solid_geom_is_on_a_body_that_collides(self):
        w = World(L.design())
        m = w.model
        for duck in w.ducks.values():
            for g in duck.solid:
                b = m.geom_bodyid[g]
                self.assertTrue(m.body_contype[b] & m.geom_contype[g])
                self.assertTrue(m.body_conaffinity[b] & m.geom_conaffinity[g])

    def test_servos_plates_and_brackets_are_solid(self):
        import mujoco
        w = World(SetDesign(), cast=("he",))
        m, he = w.model, w.ducks["he"]
        mesh = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]).partition("_")[2]
        names = [mesh(g) for g in he.solid if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.assertEqual(names.count("xl330"), 15)
        self.assertEqual(names.count("upper_leg_rigidity_plate"), 2)
        self.assertIn("ankle_left", names)
        self.assertIn("ankle_right", names)

    def test_neck_inside_a_box_makes_contact(self):
        import mujoco
        pose = ((0.0, 0.0, 0.3), np.pi / 2)
        w = World(SetDesign(), cast=("he",))
        he = w.ducks["he"]
        he.place(pose[0], pitch=pose[1], q=PLANK)
        neck = [g for g in he.solid if "neck" in mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_BODY,
                                                                    w.model.geom_bodyid[g])]
        self.assertTrue(neck)
        p = w.data.geom_xpos[neck[0]]
        w = World(SetDesign([Box("probe", tuple(p - 0.004), tuple(p + 0.004))]), cast=("he",))
        he = w.ducks["he"]
        he.place(pose[0], pitch=pose[1], q=PLANK)
        m, d = w.model, w.data
        probe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe")
        bodies = {m.geom_bodyid[g] for g in neck}
        hits = [i for i in range(d.ncon) if probe in (d.contact[i].geom1, d.contact[i].geom2)
                and {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]} & bodies]
        self.assertTrue(hits)


class Rules(unittest.TestCase):
    """story.Judge: only his back and the two ledges count, and she must step."""

    def judge_on(self, box_name):
        from microduck_lab.tasks.human_bridge.rl_policy.story import Judge
        top = 0.33
        w = World(SetDesign([Box(box_name, (-0.5, -0.5, 0), (0.5, 0.5, top))]))
        he, she = w.ducks["he"], w.ducks["she"]
        he.place((1.5, 0.0, 0.05), pitch=np.pi / 2, q=PLANK)       # lying, out of the way
        brain = Brain(she, stand_only=True)
        she.stand_on(0.05, 0.0, top, brain.default_pose, yaw=-np.pi / 2)
        w.start()
        for _ in range(10):
            w.step()
        return Judge(w, she, he)

    def test_a_ledge_counts(self):
        self.assertIsNone(self.judge_on("near_top").tick())

    def test_the_notch_floor_does_not_count(self):
        failure = self.judge_on("near_step").tick()
        self.assertIsNotNone(failure)
        self.assertIn("near_step", failure)

    def test_sliding_across_is_not_walking(self):
        judge = self.judge_on("near_top")
        judge.tick()
        self.assertIn("did not step", judge.finish())

    def test_foot_shells_count_as_feet(self):
        judge = self.judge_on("near_top")
        self.assertEqual(len(judge.shells), 2)
        self.assertFalse(judge.shells & judge.rest)



class Baked(unittest.TestCase):
    """The training scene rebuilds him from bridge_pose.json. The first bake
    saved compiled mesh frames, MuJoCo re-centred each mesh a second time, and
    every bridge policy up to v8 trained on a scrambled brother (thighs 56 mm
    off, soles 50 mm high). The rebuilt body must sit on the real one."""

    def test_baked_brother_matches_the_simulated_one(self):
        import mujoco
        import mjlab_microduck.tasks  # noqa: F401
        from scipy.spatial import cKDTree
        from microduck_lab.rl.microduck_bridge_sideways_env_cfg import add_bridge
        from microduck_lab.tasks.human_bridge.rl_policy.brother import HoldStraight
        w = World(L.design(), cast=("he",))
        he = w.ducks["he"]
        brain = Brain(he, stand_only=True)
        he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, brain.default_pose)
        w.start()
        for _ in range(75):
            brain.act()
            w.step()
        lie = LieDown(he, he.target, w.t)
        while not lie(w.t):
            w.step()
        hold = HoldStraight(he)
        for _ in range(int(round(SETTLE_S / 0.02))):
            hold()
            w.step()
        real = he.points([g for g in he.solid if w.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
        s = mujoco.MjSpec()
        s.worldbody.add_body(name="terrain")
        add_bridge(s)
        m = s.compile()
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        pts = []
        for g in range(m.ngeom):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("brother_"):
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        dist, _ = cKDTree(real).query(np.vstack(pts))
        self.assertLess(float(dist.max()), 0.002)


if __name__ == "__main__":
    unittest.main()
