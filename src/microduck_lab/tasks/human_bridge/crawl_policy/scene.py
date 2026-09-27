"""Where everything is. All numbers in metres, world x points across the gap.

    near top      near step   gap (a real drop)   far shelf    far top
    her start     his spot                        his head     her goal
  ___________
             |___________                      ___________ ___________
                         |                    |           |
                         |                    |___________|

Seen from above, the near step and the far shelf are notches cut into the two
ledges, only as wide as he is. Her ledge runs on both sides of him. When the
step ran the full width, her head dropped off his legs onto it in the first
3 s of every crawl.

He stands on the near step, 35 mm below her ledge, and falls forward. His
soles stay on the step, his head lands on the far shelf, and his back ends up
roughly level with her ledge. The steps exist because this robot's walking
policy climbs 15 mm and its feet lift 15 mm: a duck's back 80+ mm above her
ledge could never be reached.

The heights were measured off his plank pose (`brother.PLANK`): soles 7 mm
above his trunk axis, back 42-46 mm above it. The lowest part of his head end
is his neck servo, 36 mm below the axis; he rests on it on the far shelf.

2026-09-26: until the collision fix in `world.arm`, his neck did not collide
and sank 17 mm into the far shelf. The old shelf (head underside 31 mm below
the axis, +20 mm) then tipped him 11 deg head-up. The shelf is now 26 mm
lower (shelf_z 0.256). After the lie-down and SETTLE_S of HoldStraight he lies
level (pitch -0.6 deg), lands well from 15 of 15 starts, and holds a 757 g
block (her mass) at 5 points along his back with at most 2.7 mm of sag.
"""
from __future__ import annotations

from dataclasses import dataclass

from microduck_lab.tasks.human_bridge.crawl_policy.world import Box, SetDesign

STONE_TOP = (0.80, 0.77, 0.71, 1)
STEP = (0.66, 0.62, 0.56, 1)


@dataclass(frozen=True)
class Layout:
    step_z: float = 0.300        # top of the near step, where he stands
    gap_start: float = 0.000     # the near step ends here; the drop starts
    gap_end: float = 0.100       # the far shelf starts here
    shelf_len: float = 0.100     # far shelf length, under his head (reduced so Sister's legs don't stick)
    step_len: float = 0.052      # near step length, under his feet (reduced so Sister's legs don't stick in the gap)
    half_width: float = 0.35     # how far the set runs to each side (y)
    shelf_rise: float = -0.001   # far shelf height trim; -0.001 gives shelf_z 0.256
    notch_near: float = 0.085    # half-width of the near step (he is 71 mm)
    notch_far: float = 0.100     # half-width of the far shelf (his head is 46 mm); side ledges well clear of him
    drop: float = 0.0            # far side this much lower: he lies as a slope

    # Offsets measured on the plank pose, relative to his trunk axis.
    SOLE_BELOW_AXIS = -0.007     # soles are 7 mm ABOVE the axis
    HEAD_BELOW_AXIS = 0.036      # his neck servo, the lowest part of his head end
    # Her ledge is level with his shin servos (335 mm), so her first step onto
    # him is flat. A 30 mm hole stays between her ledge edge and his ankles:
    # his standing heel needs that room before he falls. (2026-09-26, after the
    # collision fix; before, her ledge was 6 mm lower with a 35 mm pit.)
    BACK_ABOVE_AXIS = 0.042
    # Far ledge level with the lowest point her sole reaches on his head, which
    # ends 13 mm short of it: a 0.8 mm step instead of a 23 mm drop.
    HEAD_TOP_ABOVE_AXIS = 0.0295

    @property
    def axis_z(self):
        """His trunk axis when he lies as the bridge."""
        return self.step_z + self.SOLE_BELOW_AXIS

    @property
    def near_z(self):
        return self.axis_z + self.BACK_ABOVE_AXIS

    @property
    def shelf_z(self):
        return self.axis_z - self.HEAD_BELOW_AXIS + self.shelf_rise - self.drop

    @property
    def far_z(self):
        return self.axis_z + self.HEAD_TOP_ABOVE_AXIS - self.drop

    @property
    def near_edge(self):
        """Her ledge ends here."""
        return self.gap_start - self.step_len

    @property
    def far_edge(self):
        """Her goal ledge starts here."""
        return self.gap_end + self.shelf_len

    def design(self, extra=""):
        y, nn, nf = self.half_width, self.notch_near, self.notch_far
        a, b, c, d = self.near_edge, self.gap_start, self.gap_end, self.far_edge
        boxes = [
            Box("near_top", (-0.90, -y, 0), (a, y, self.near_z), STONE_TOP),
            Box("near_step", (a, -nn, 0), (b, nn, self.step_z), STEP),
            Box("far_shelf", (c, -nf, 0), (d, nf, self.shelf_z), STEP),
            Box("far_top", (d, -y, 0), (1.10, y, self.far_z), STONE_TOP),
        ]
        for side, name in ((1, "left"), (-1, "right")):
            lo, hi = sorted((side * nn, side * y))
            boxes.append(Box(f"near_top_{name}", (a, lo, 0), (b, hi, self.near_z), STONE_TOP))
            lo, hi = sorted((side * nf, side * y))
            boxes.append(Box(f"far_top_{name}", (c, lo, 0), (d, hi, self.far_z), STONE_TOP))
        return SetDesign(boxes, extra=extra)


LAYOUT = Layout()
