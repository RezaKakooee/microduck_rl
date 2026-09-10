"""The swing must be passive; robot motor work must create the oscillation."""
import unittest
import mujoco
import numpy as np
from microduck_lab.tasks.swing.swing import Swing
from microduck_lab.tasks.swing.expert import PumpingExpert,evaluate


class SwingTests(unittest.TestCase):
    def test_passive_suspension_and_free_robot(self):
        s=Swing();m,d=s.model,s.data
        joint=s.id('JOINT','passive_swing')
        self.assertFalse(np.any(m.actuator_trnid[:,0]==joint))
        self.assertEqual(m.nmocap,0);self.assertEqual(m.neq,0)
        root=s.id('JOINT','he_trunk_base_freejoint')
        self.assertEqual(m.jnt_type[root],mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_array_equal(d.qvel,0)
        self.assertEqual(d.qpos[s.swing_q],0)
        self.assertGreater(min((c.dist for c in d.contact),default=0),-.0005)

    def test_controller_responds_to_swing_direction(self):
        s=Swing();p=PumpingExpert(s)
        before=s.data.qpos.copy()
        forward=p(.025,.5,10);backward=p(.025,-.5,10)
        self.assertGreater(np.linalg.norm(forward-backward),1.)
        np.testing.assert_array_equal(s.data.qpos,before)

    def test_external_push_is_rejected(self):
        s=Swing();s.data.xfrc_applied[s.seat,0]=.1
        with self.assertRaisesRegex(RuntimeError,'External force'):s.step(s.base)

    def test_collision_bypass_is_rejected(self):
        s=Swing();g=s.id('GEOM','seat');s.model.geom_contype[g]=0
        with self.assertRaisesRegex(RuntimeError,'Changed geom_contype'):s.step(s.base)

    def test_pumping_beats_same_initial_state_without_pumping(self):
        active=evaluate();passive=evaluate(active=False)
        self.assertTrue(active['success'],active)
        self.assertTrue(passive['success'],passive)
        self.assertGreater(active['last_20s_amplitude_deg'],20)
        self.assertLess(passive['last_20s_amplitude_deg'],1)
        self.assertGreater(active['net_servo_work_j'],5)
        self.assertLess(active['maximum_servo_torque_nm'],active['servo_torque_limit_nm'])
        self.assertGreater(active['minimum_seated_upright'],.9)


    def test_legs_actually_move(self):
        """The first version pumped with the head alone: the knees moved 0.0 deg
        and the hips 6.8 deg, while the neck moved 70 deg. Lock in real leg use."""
        r=evaluate(seconds=60.)
        self.assertTrue(r['success'],r)
        ranges=r['joint_range_last_20s_deg']
        self.assertGreater(ranges.get('left_knee',0),25,'knees barely move')
        self.assertGreater(ranges.get('right_knee',0),25,'knees barely move')
        self.assertGreater(ranges.get('left_hip_pitch',0),20,'hips barely move')
        legs=ranges.get('left_knee',0)+ranges.get('left_hip_pitch',0)
        self.assertGreater(legs,ranges.get('neck_pitch',0),'head still dominates')

    def test_legs_alone_can_pump(self):
        """With the head held still the legs must still drive the swing well
        above the fixed-pose baseline of 0.3 deg."""
        legs=evaluate(seconds=60.,expert_kw=dict(legs_only=True))
        fixed=evaluate(seconds=60.,active=False)
        self.assertGreater(legs['last_20s_amplitude_deg'],
                           10*fixed['last_20s_amplitude_deg'])
        self.assertGreater(legs['net_servo_work_j'],3)

    def test_knee_second_harmonic_does_not_pump(self):
        """Parametric pumping at twice the swing rate fails on this robot.
        Measured legs-only: 0.1 deg at 2x against 8.8 deg at 1x."""
        once=evaluate(seconds=50.,expert_kw=dict(legs_only=True,hip=-.9,knee=.6,knee_harmonic=1))
        twice=evaluate(seconds=50.,expert_kw=dict(legs_only=True,hip=-.9,knee=.6,knee_harmonic=2))
        self.assertGreater(once['last_20s_amplitude_deg'],
                           5*twice['last_20s_amplitude_deg'])

    def test_bottom_rests_on_the_seat_plate(self):
        """The duck must sit on the plate, not hang off the frame.

        An earlier seat only reached the shins, so the duck's underside stayed
        36 mm above the plate and nothing was under it."""
        s=Swing()
        for _ in range(250):s.step(s.base)
        pairs=set()
        for c in s.data.contact[:s.data.ncon]:
            names=[mujoco.mj_id2name(s.model,mujoco.mjtObj.mjOBJ_BODY,s.model.geom_bodyid[g])
                   for g in (c.geom1,c.geom2)]
            geoms=[mujoco.mj_id2name(s.model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c.geom1,c.geom2)]
            for i in (0,1):
                if (names[i] or '').startswith('he_'):pairs.add((names[i],geoms[1-i]))
        self.assertIn(('he_trunk_base','seat'),pairs,pairs)
        self.assertLess(s.max_penetration,.0015,'seat digs into the duck')
        self.assertGreater(s.min_up,.97,'duck does not sit straight')

    def test_parts_that_meet_the_seat_are_solid(self):
        """Contact must match what you can see.

        Group 3 leaves the thigh and the front of the belly out. The visible
        thigh hung 41 mm below any collision shape, so the seat touched a shape
        nobody can see and the duck looked sunk into the plate."""
        s=Swing();m,d=s.model,s.data
        def lowest(body,solid):
            out=[]
            for g in range(m.ngeom):
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
                if name!=body or bool(m.geom_contype[g])!=solid:continue
                if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                a=m.mesh_vertadr[m.geom_dataid[g]];n=m.mesh_vertnum[m.geom_dataid[g]]
                v=m.mesh_vert[a:a+n].astype(float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                out.append(v[:,2].min())
            return min(out) if out else None
        for body in ('he_trunk_base','he_upper_leg_left','he_upper_leg_right','he_leg','he_ankle_left'):
            visual=lowest(body,False)
            if visual is None:continue
            solid=lowest(body,True)
            self.assertIsNotNone(solid,body+' has no collision shape at all')
            self.assertLess(solid-visual,.002,body+' is see-through underneath')

    def test_no_part_of_the_duck_passes_through_the_seat(self):
        """Nothing may look see-through, not just the parts that carry load.

        The run is replayed on a second model where all 81 duck geoms are solid.
        Only that model is checked, and it is never stepped."""
        s=Swing();e=PumpingExpert(s);poses=[]
        for i in range(500):
            angle,velocity=s.observation();s.step(e(angle,velocity,s.t))
            if i%3==0:poses.append(s.data.qpos.copy())
        look=Swing();m=look.model
        for g in range(m.ngeom):
            body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
            if body.startswith('he_'):m.geom_contype[g]=1;m.geom_conaffinity[g]=2
        d=mujoco.MjData(m);worst=0.;where=None
        for q in poses:
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            for c in d.contact[:d.ncon]:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])or''
                       for g in (c.geom1,c.geom2)]
                geoms=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c.geom1,c.geom2)]
                if names[0].startswith('he_')==names[1].startswith('he_'):continue
                if 'floor' in geoms:continue
                if -c.dist>worst:worst=-c.dist;where=geoms
        self.assertLess(worst,.0015,'duck sinks %.2f mm into %s'%(1000*worst,where))


if __name__=='__main__':unittest.main()
