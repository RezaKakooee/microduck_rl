"""Two coplanar visible faces make the renderer flicker. Catch them here.

The nest rim in `scene_v2` had its top face at exactly z = 0, the same as the
ground slabs it sits inside. Nothing about the physics was wrong and no test
failed; the depth buffer simply could not order the two surfaces, so on screen
the whole rectangle flashed brown-green-brown as the camera moved. It was found
by watching the film.

A pair only flickers if both faces point the SAME way -- two up-faces or two
down-faces at one height, with overlapping footprints. A plate resting on a
column also has two coplanar faces, but they point at each other and are
enclosed, so they are never drawn.

Only axis-aligned boxes are checked: a rotated box cannot be exactly coplanar
with anything by accident, and inexact is enough for the depth buffer.
"""

import itertools
import unittest

import mujoco
import numpy as np

from microduck_lab.film import physical_stage, scene_v2

FLUSH = 1e-9        # closer than this in z and the depth buffer cannot choose
TOUCH = 1e-6        # footprints must really overlap, not just share an edge


def flickering_pairs(model, data):
    mujoco.mj_forward(model, data)

    boxes = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        if np.abs(np.abs(rot) - np.eye(3)).max() > FLUSH:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        boxes.append((name, data.geom_xpos[g], model.geom_size[g]))

    bad = []
    for (n1, p1, s1), (n2, p2, s2) in itertools.combinations(boxes, 2):
        for face in (+1, -1):                       # both up-faces, or both down-faces
            if abs((p1[2] + face * s1[2]) - (p2[2] + face * s2[2])) > FLUSH:
                continue
            ox = min(p1[0] + s1[0], p2[0] + s2[0]) - max(p1[0] - s1[0], p2[0] - s2[0])
            oy = min(p1[1] + s1[1], p2[1] + s2[1]) - max(p1[1] - s1[1], p2[1] - s2[1])
            if ox > TOUCH and oy > TOUCH:
                bad.append(f"{n1} and {n2} share a face at z={p1[2] + face * s1[2]:+.5f} "
                           f"over {ox * 1000:.0f}x{oy * 1000:.0f} mm")
                break
    return bad


SHADOW_TEXEL_LIMIT_MM = 1.0     # a duck's head is 50 mm; blocks this big crawl on it


def shadow_texel_mm(model):
    """Millimetres of world covered by one texel of the shadow map."""
    return model.vis.map.shadowclip * model.stat.extent / model.vis.quality.shadowsize * 1000


class RenderFlickerTests(unittest.TestCase):
    def test_love_story_v2_set_has_no_coplanar_visible_faces(self):
        bad = flickering_pairs(*scene_v2.build_model())
        self.assertEqual(bad, [], "coplanar faces flicker on screen:\n  " + "\n  ".join(bad))

    def test_dispenser_set_has_no_coplanar_visible_faces(self):
        bad = flickering_pairs(*physical_stage.build_model())
        self.assertEqual(bad, [], "coplanar faces flicker on screen:\n  " + "\n  ".join(bad))

    def test_the_shadow_map_is_fine_enough_for_a_duck(self):
        """A coarse shadow map is the other way this scene flickers.

        The map is spread over `shadowclip * extent`, and the extent here is
        set by 5 m ground slabs, not by the 50 mm robot anyone is looking at.
        At the MuJoCo default that was 3 mm per texel, so self-shadow landed in
        blocks a sixth of a head wide and they crawled over her face as she
        moved. Anything under a millimetre per texel is fine.
        """
        for name, build in (("scene_v2", scene_v2.build_model),
                            ("physical_stage", physical_stage.build_model)):
            mm = shadow_texel_mm(build()[0])
            self.assertLess(mm, SHADOW_TEXEL_LIMIT_MM,
                            f"{name}: shadow map is {mm:.2f} mm per texel; "
                            f"self-shadow will flicker on a 50 mm head")

    def test_the_nest_rim_stands_clear_of_the_grass_but_stays_walkable(self):
        """The rim must be off z=0, and low enough that nobody trips on it."""
        top = scene_v2.RIM_Z + scene_v2.RIM_H
        self.assertGreater(top, 1e-5, "rim is flush with the ground again; it will flicker")
        self.assertLess(top, 0.0005, "rim stands too proud; 1.5 mm has tripped a duck here")


if __name__ == "__main__":
    unittest.main()
