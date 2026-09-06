"""Physical release, collision and runtime-state regression checks (CPU)."""
import unittest

import mujoco
import numpy as np

from microduck_lab.tasks.love_story.dispenser import Story


class PhysicalStoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.story=Story(verbose=False)

    def test_model_has_real_props_and_no_mocap_or_welds(self):
        s=self.story;m=s.model
        self.assertEqual(m.nmocap,0)
        self.assertEqual(m.neq,0)
        for name in ('ring_segment_0','egg0_bottom_panel_1_0','egg1_top_panel_1_0','nest_mat','gate_plate_0'):
            g=s.id('GEOM',name)
            self.assertNotEqual(m.geom_contype[g],0)
            self.assertEqual(m.geom_rgba[g,3],1)
        for i in (0,1):
            j=s.id('JOINT',f'egg_lid_{i}')
            self.assertAlmostEqual(m.jnt_range[j,0],-2.2,places=5)
            self.assertEqual(m.jnt_type[s.id('JOINT',f'egg{i}_free')],mujoco.mjtJoint.mjJNT_FREE)
        # Every chick collision hull contacts the world/props from time zero.
        for g in range(m.ngeom):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
            if name.startswith(('kidb_','kidp_')) and m.geom_group[g]==3:
                self.assertTrue(m.geom_conaffinity[g]&16)

    def test_release_and_hatch_with_continuous_collisions(self):
        s=self.story;m,d=s.model,s.data
        state=mujoco.MjData(m)
        mujoco.mj_copyData(state,m,d)
        masks=m.geom_contype.copy(),m.geom_conaffinity.copy(),m.geom_rgba.copy()
        initial=d.xpos[s.egg_body,2].copy()
        min_up=np.ones(2)
        try:
            for tick in range(1000):
                t=tick*.02
                q,v=d.qpos.copy(),d.qvel.copy()
                for a in s.actors:a.tick()
                for i in (0,1):
                    d.ctrl[s.gate_act[i]]=.085*np.clip((t-2-i*2)/1.5,0,1)
                    d.ctrl[s.lid_act[i]]=-2*np.clip((t-10-i*2)/2.5,0,1)
                np.testing.assert_array_equal(q,d.qpos)
                np.testing.assert_array_equal(v,d.qvel)
                for _ in range(4):mujoco.mj_step(m,d)
                mujoco.mj_forward(m,d)
                min_up=np.minimum(min_up,[k.up() for k in s.kids])
                if t<2:np.testing.assert_allclose(d.xpos[s.egg_body,2],initial,atol=.001)
            self.assertTrue(np.all(initial-d.xpos[s.egg_body,2]>.010))
            self.assertTrue(np.all(d.qpos[s.gate_q]>.080))
            self.assertTrue(np.all(d.qpos[s.lid_q]<-1.7))
            self.assertTrue(np.all(min_up>.85),min_up)
            for i,k in enumerate(s.kids):
                self.assertLess(np.linalg.norm(k.xy()-d.xpos[s.egg_body[i],:2]),.015)
            for before,after in zip(masks,(m.geom_contype,m.geom_conaffinity,m.geom_rgba)):
                np.testing.assert_array_equal(before,after)
        finally:mujoco.mj_copyData(d,m,state)

    def test_guard_rejects_position_edits(self):
        s=self.story;old=s.control;before=s.data.qpos.copy()
        try:
            def forbidden():s.data.qpos[0]+=.001
            s.control=forbidden
            with self.assertRaisesRegex(RuntimeError,'modified position'):s.step()
        finally:s.control=old;s.data.qpos[:]=before

    def test_timeout_is_failure_not_completion(self):
        s=self.story;old=s.phases
        try:
            s.phases=[('unreached',-.1,lambda elapsed:False)]
            with self.assertRaisesRegex(RuntimeError,'Timed out'):s.control()
            self.assertEqual(s.phase_i,0)
        finally:s.phases=old


if __name__=='__main__':unittest.main()
