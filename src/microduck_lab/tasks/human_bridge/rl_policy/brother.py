"""The brother: he stands on the near step, then lies down across the gap.

Lying down is one smooth joint-target blend from his standing pose to
PLANK. Nothing else is needed: as the ankles tip him forward his centre of
mass passes the front of his feet, he falls, and his head lands on the far
shelf. His soles stay flat on the step the whole time, because PLANK turns
the ankles by 90 degrees (the servo range ends at 90.0 degrees).

PLANK was found by mapping his back with the mesh vertices:

* legs pressed together (hip roll -/+0.38): no gap between his legs for her
  feet to drop into;
* legs straight, ankles at -/+90 deg: soles flat, facing down, 7 mm above the
  trunk axis;
* head_pitch -1.2 rad: the head folds back like a person in a plank looking
  ahead. With the head straight there is a 60 mm deep dip over the neck; with
  -1.2 his back runs smoothly from the trunk (46 mm) down the head (20 mm).
"""
from __future__ import annotations

import numpy as np

from microduck_lab.tasks.human_bridge.rl_policy.world import SERVOS

J = {name: i for i, name in enumerate(SERVOS)}

PLANK = np.zeros(len(SERVOS))
PLANK[J["left_ankle"]] = -1.57
PLANK[J["right_ankle"]] = 1.57
PLANK[J["head_pitch"]] = -1.2
# Legs pressed together (hip roll at its limit, 0.384 rad). Face down his legs
# lie at |y| 32-71 mm with a 64 mm gap between them on his centre line, and
# her sideways sole (54 mm across his back) dropped into it. Adducted, the
# centre line has solid body along his whole length.
PLANK[J["left_hip_roll"]] = -0.38
PLANK[J["right_hip_roll"]] = 0.38

LIE_DOWN_S = 2.0        # the blend; the fall itself takes about 0.5 s of it
# HoldStraight needs about 5 s to settle: his trunk rises 15 mm while the
# integral winds up, then drifts under 3 mm. The story lets him hold this
# long before she starts, and bake.py freezes him at the same moment, so
# she trains on the body she meets.
SETTLE_S = 8.0


def smooth(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


class LieDown:
    """Blend from `start` to PLANK over LIE_DOWN_S, then hold PLANK."""

    def __init__(self, duck, start, t0):
        self.duck, self.start, self.t0 = duck, np.asarray(start, float), t0

    def __call__(self, t):
        s = smooth((t - self.t0) / LIE_DOWN_S)
        self.duck.hold((1 - s) * self.start + s * PLANK)
        return t - self.t0 >= LIE_DOWN_S


class HoldStraight:
    """Hold PLANK against her weight.

    The BAM servo is a P loop of only 0.55 Nm/rad, so with her on his back
    his joints sag and he sank 19 mm in 54 s. Like the boy tensing his body,
    he raises each joint target by the integral of its error until the joint
    is back at PLANK. Only the targets change.
    """

    GAIN = 1.5          # 1/s
    LIMIT = 0.6         # rad; 0.6 * 0.55 Nm/rad = 0.33 Nm of extra hold

    def __init__(self, duck, dt=0.02):
        self.duck, self.dt = duck, dt
        self.extra = np.zeros(len(SERVOS))

    def __call__(self):
        err = PLANK - self.duck.q
        self.extra = np.clip(self.extra + self.GAIN * err * self.dt, -self.LIMIT, self.LIMIT)
        self.duck.hold(PLANK + self.extra)
