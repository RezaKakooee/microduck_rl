"""Twenty real-time seconds: Fast exits, Little remains under the net."""
import unittest
import numpy as np
from microduck_lab.tasks.crawl import crawl
from microduck_lab.tasks.crawl.net import NetCrawl,evaluate


class CrawlNetTests(unittest.TestCase):
    def test_obstacle_is_solid_and_scene_override_is_scoped(self):
        factory=crawl.world_xml
        c=NetCrawl()
        self.assertIs(crawl.world_xml,factory)
        ids=list(c.net_geoms)
        self.assertGreater(len(ids),1000)
        self.assertTrue(np.all(c.model.geom_contype[ids]))
        self.assertTrue(np.all(c.model.geom_conaffinity[ids]))
        self.assertEqual(c.model.nmocap,0)
        self.assertEqual(c.model.neq,0)
        self.assertFalse(np.any(c.model.body_gravcomp))

    def test_real_time_scenarios_finish_at_the_requested_positions(self):
        for kind in ('fast','little'):
            with self.subTest(gait=kind):
                r=evaluate(kind)
                self.assertTrue(r['success'])
                self.assertEqual(r['seconds'],20.)
                self.assertEqual(r['video_seconds'],20.)
                self.assertEqual(r['playback_speed'],1.)
                self.assertGreater(r['entered_at_seconds'],.5)
                if kind=='fast':
                    self.assertTrue(r['final_whole_body_clear'])
                    self.assertGreater(r['distance_mm'],3000)
                    self.assertLess(r['whole_body_cleared_at_seconds'],17.)
                else:
                    self.assertTrue(r['final_whole_body_under_net'])
                    self.assertFalse(r['final_whole_body_clear'])
                    self.assertIsNone(r['whole_body_cleared_at_seconds'])
                    self.assertGreater(r['distance_mm'],450)
                self.assertGreater(r['minimum_conservative_rope_clearance_mm'],10)
                self.assertLess(r['trunk_height_mm']['max'],100)
                self.assertEqual(r['observed_net_contacts'],0)


if __name__=='__main__':unittest.main()
