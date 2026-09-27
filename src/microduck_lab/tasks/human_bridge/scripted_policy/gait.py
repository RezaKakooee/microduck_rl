"""A timed sideways step with a moving body reference and leg IK."""
import math

import mujoco
import numpy as np

from microduck_lab.tasks.human_bridge.scripted_policy.surface import Surface


class Gait:
    """Read full state. Send joint targets only. No live state writes."""
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.duck, self.cfg = supervisor.duck, supervisor.cfg
        d = self.duck
        self.t0 = d.world.t
        self.start = d.pos().copy()
        self.feet = np.array(d.feet())
        self.rotations = [d.data.site_xmat[s].reshape(3, 3).copy() for s in d.foot_sites]
        self.surface = Surface(d.world.ducks['he'])
        self.step = -1
        self.extra = np.zeros(14)
        self.last = d.target.copy()
        self.sway_width = abs(self.feet[0, 0] - self.feet[1, 0]) / 2

    def height(self, x):
        return max(self.surface.height(x + a, b) for a in (-.012, 0., .012) for b in (-.02, 0., .02))

    def act(self):
        d, c = self.duck, self.cfg
        t = d.world.t - self.t0
        n = int(t / c.step_time)
        u = (t / c.step_time) - n
        side = n % 2
        if n != self.step:
            if self.step >= 0:
                self.feet[1-side] = self.end.copy()
            self.step = n
            self.begin = self.feet[side].copy()
            self.end = self.begin.copy()
            self.end[0] += c.stride
            self.end[1] = c.y_target
            self.end[2] = self.height(self.end[0]) + .001
        blend = u*u*(3-2*u)
        feet = self.feet.copy()
        feet[side] = (1-blend)*self.begin + blend*self.end
        feet[side, 2] += c.step_height * math.sin(math.pi*u)
        progress = c.stride / (2*c.step_time) * t
        root = self.start.copy()
        # Alternate the support side. Start and end each step at the midpoint.
        root[0] += progress + (-1 if side == 0 else 1) * c.sway * math.sin(math.pi*u)
        root[1] = c.y_target + c.gait_y_bias
        root[2] = (self.height(feet[0, 0]) + self.height(feet[1, 0]))/2 + c.body_height
        root[1] += c.gait_feedback * (c.y_target - d.pos()[1]) - c.gait_damping*d.data.qvel[d.dof+1]
        k = self.supervisor.kinematics
        k.qpos[:] = d.data.qpos
        k.qpos[d.qidx] = self.last
        k.qpos[d.adr:d.adr+3] = root
        yaw = -math.pi/2
        k.qpos[d.adr+3:d.adr+7] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
        mujoco.mj_forward(d.model, k)
        for i, site in enumerate(d.foot_sites):
            ids = np.arange(0, 5) if i == 0 else np.arange(9, 14)
            self.supervisor.solve_foot(k, site, ids, feet[i], self.rotations[i])
        target = k.qpos[d.qidx].copy()
        self.extra = np.clip(self.extra + c.joint_integral*.02*(target-d.q), -.4, .4)
        self.last = target
        d.hold(target + self.extra)
