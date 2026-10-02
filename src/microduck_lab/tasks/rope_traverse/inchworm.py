"""Open-loop inchworm along the rope with the two leg hooks.

One cycle has three moves. Both legs stay on the rope the whole time:

1. The trailing leg squeezes its hook shut (a clamp). The leading leg opens
   its hook a little and its hip roll pushes it out along the rope.
2. The leading leg clamps. The trailing leg opens and is pulled in.
3. Both hooks relax. Both hip rolls turn back together, which carries the
   body forward between the two hooks.

Only joint targets are sent (50 Hz, smooth steps). No rope sensing is used.
"""
from dataclasses import dataclass, fields

import numpy as np

from microduck_lab.tasks.rope_traverse.arch import hang_pose, leg_targets
from microduck_lab.tasks.rope_traverse.fit import I


@dataclass(frozen=True)
class Gait:
    # Hip roll of the leading leg at the inner and outer end of its slide,
    # and of the trailing leg at its outer and inner end (rad, in the travel
    # direction: positive = toward where the robot goes).
    lead_in: float = -.38
    lead_out: float = .38
    trail_out: float = -.38
    trail_in: float = .38
    # Knee / ankle (left-leg sign) of a clamped hook, a sliding hook, and of
    # both hooks while the body is carried forward.
    clamp_knee: float = 1.4
    clamp_ankle: float = -.5
    free_knee: float = .36
    free_ankle: float = .06
    carry_knee: float = .36
    carry_ankle: float = .06
    # Durations of the six steps of one cycle (s).
    grip_trail: float = 1.2
    slide_lead: float = 1.2
    grip_lead: float = 1.2
    slide_trail: float = 1.2
    relax: float = 1.2
    carry: float = 1.2
    # Time to close both hooks before the first cycle (s).
    start: float = 1.5

    @property
    def cycle_s(self):
        return self.grip_trail + self.slide_lead + self.grip_lead + self.slide_trail + self.relax + self.carry

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3 - 2 * u)


class Inchworm:
    """Joint targets as a function of time. `direction` +1 moves toward the
    robot's left (world +x in `arch.place`), -1 toward its right."""

    def __init__(self, gait=Gait(), cycles=1, direction=1):
        if direction not in (1, -1):
            raise ValueError('direction is +1 or -1')
        self.gait, self.cycles, self.direction = gait, cycles, direction
        lead, trail = ('left', 'right') if direction == 1 else ('right', 'left')
        g = gait
        clamp, free, carry = (g.clamp_knee, g.clamp_ankle), (g.free_knee, g.free_ankle), (g.carry_knee, g.carry_ankle)
        # Hip roll: both legs' +roll moves the hook toward the robot's left (+y).
        roll = lambda value: direction * value

        def pose(lead_shape, lead_roll, trail_shape, trail_roll):
            q = hang_pose()
            for side, (knee, ankle), r in ((lead, lead_shape, lead_roll), (trail, trail_shape, trail_roll)):
                q = leg_targets(q, side, knee=knee, ankle=ankle)
                q[I[f'{side}_hip_roll']] = roll(r)
            return q

        steps = [('close both hooks', g.start, pose(clamp, g.lead_in, clamp, g.trail_out))]
        for _ in range(cycles):
            steps += [('grip trailing, free leading', g.grip_trail, pose(free, g.lead_in, clamp, g.trail_out)),
                      ('slide leading leg out', g.slide_lead, pose(free, g.lead_out, clamp, g.trail_out)),
                      ('grip leading, free trailing', g.grip_lead, pose(clamp, g.lead_out, free, g.trail_out)),
                      ('slide trailing leg in', g.slide_trail, pose(clamp, g.lead_out, free, g.trail_in)),
                      ('relax both hooks', g.relax, pose(carry, g.lead_out, carry, g.trail_in)),
                      ('carry body forward', g.carry, pose(carry, g.lead_in, carry, g.trail_out))]
        self.steps = steps
        self.q0 = hang_pose()
        self.t_end = np.cumsum([s[1] for s in steps])
        self.duration = float(self.t_end[-1])

    def step(self, t):
        i = int(np.searchsorted(self.t_end, t, side='right'))
        return min(i, len(self.steps) - 1)

    def phase(self, t):
        return self.steps[self.step(t)][0]

    def __call__(self, t):
        """Joint targets at time t (s since the start of the gait)."""
        i = self.step(t)
        name, dur, q1 = self.steps[i]
        q0 = self.q0 if i == 0 else self.steps[i - 1][2]
        u = smooth((t - (self.t_end[i] - dur)) / dur)
        return q0 + (q1 - q0) * u

    def cycle_starts(self):
        """Times at which each cycle starts, plus the end time."""
        per = 6
        return [float(self.t_end[i * per]) for i in range(self.cycles)] + [self.duration]
