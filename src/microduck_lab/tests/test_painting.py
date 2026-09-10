"""Painting must originate at a loaded brush near the blank canvas."""
import unittest
import numpy as np
from microduck_lab.tasks.painting.painting import Painting, CANVAS_X, CANVAS_Z, WELLS
from microduck_lab.tasks.painting.expert import Canvas


class PaintingTests(unittest.TestCase):
    def test_no_paint_without_loading_or_canvas_reach(self):
        c=Canvas();blank=np.asarray(c.image).copy()
        draw=dict(mode='draw',color='pink')
        at=np.array([CANVAS_X,0,CANVAS_Z])
        self.assertFalse(c.update(at,draw))
        c.update(WELLS['pink']+[0,0,.003],dict(mode='dip',color='pink'))
        self.assertFalse(c.update(at+[-.02,0,0],draw))
        np.testing.assert_array_equal(c.image,blank)
        self.assertTrue(c.update(at,draw))
        self.assertEqual(c.contacts['pink'],1)
        self.assertTrue(np.any(np.asarray(c.image)!=blank))

    def test_lift_does_not_connect_separate_strokes(self):
        c=Canvas();c.color='green';draw=dict(mode='draw')
        c.update(np.array([CANVAS_X,-.03,CANVAS_Z]),draw)
        c.update(np.array([CANVAS_X-.02,0,CANVAS_Z]),draw)
        c.update(np.array([CANVAS_X,.03,CANVAS_Z]),draw)
        np.testing.assert_array_equal(np.asarray(c.image)[384,384],[250,247,235])

    def test_ik_does_not_edit_live_robot_state(self):
        s=Painting();q=s.data.qpos.copy();v=s.data.qvel.copy()
        cmd=s.command(np.array([.16,0,.32]))
        self.assertTrue(np.isfinite(cmd).all())
        np.testing.assert_array_equal(s.data.qpos,q)
        np.testing.assert_array_equal(s.data.qvel,v)
        self.assertEqual(s.model.nmocap,0)
        self.assertEqual(s.model.neq,0)
        self.assertGreater(s.data.xmat[s.trunk,8],.99)


if __name__=='__main__':unittest.main()
