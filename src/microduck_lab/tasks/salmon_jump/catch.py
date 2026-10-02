"""Closed-loop landing ("catch") after the salmon-jump takeoff.

Why: in every earlier trial the robot touched down with its centre of mass
(COM) 25-40 mm behind the heel. The sole is 54 mm long, so no standing
controller can hold that. The catch puts a flat foot under the COM instead.

Inputs are what the robot can sense: joint angles, IMU trunk pitch and gyro.
The COM and sole positions come from forward kinematics of those readings
(on a scratch MjData for planning, so the running simulation is untouched).
Outputs are joint targets only.

Phases, starting at the airborne trigger:
  air    -- each tick, solve legs that put a flat sole under the COM for the
            trunk pitch the IMU measures (optionally a little ahead in time).
  crouch -- after touchdown: hold that solution for the measured pitch, with
            ankle feedback on the COM offset and hip feedback on pitch.
  rise   -- move the pitch and COM height references to the stand pose.
  stand  -- the standing policy.

Measured (2026-09-30): the scripted rise failed in 54 of 54 tests. Under
load the servos lag their targets by up to 0.8 rad, so the fixed rise path
tips the robot over. Use rise_s=0 and a large pitch_rate_limit: after the crouch,
hand over to the standing policy at once. That worked for crouch times of
0.1-0.8 s. Defaults are unchanged so saved runs replay exactly.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from microduck_lab.tasks.common.world import SERVOS

INDEX = {name: i for i, name in enumerate(SERVOS)}
LEG = ('hip_pitch', 'knee', 'ankle')
SOLE_FLAT = np.pi/2  # sole geom frame pitch when the sole lies flat


@dataclass
class Catch:
    com_offset: float = .005     # desired COM x minus sole centre x (m)
    neck: float = -1.0           # neck and head targets in air and crouch;
    head: float = .349           # neck > 0 moves the head (and COM) back
    air_neck: float = .349       # neck in flight: moving the head back turns
                                 # the trunk forward, moving it forward turns it back
    pitch_gain: float = .3       # hip += gain*(pitch - reference) + damping*rate
    pitch_damping: float = .03
    com_gain: float = 8.         # ankle += gain*(offset error) + damping*rate
    com_damping: float = .3
    crouch_s: float = .4         # hold time after touchdown
    rise_s: float = 1.2          # time to move the references to the stand pose
    pitch_rate_limit: float = .5  # rad/s: how fast the pitch reference moves to 0
    touchdown_n: float = 3.      # foot force that ends the air phase
    lookahead_s: float = 0.      # in flight, plan for pitch + rate*lookahead

    def __post_init__(self):
        values = np.array([getattr(self, k) for k in self.__dataclass_fields__], float)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite catch parameter')
        if min(self.crouch_s, self.rise_s, self.pitch_rate_limit) < 0:
            raise ValueError('Negative catch duration or rate')


class Planner:
    """Flat-foot inverse kinematics on a scratch copy of the model."""

    def __init__(self, duck):
        self.duck = duck
        self.model = duck.model
        self.data = mujoco.MjData(duck.model)
        self.bodies = np.array(sorted(duck.bodies))
        self.mass = self.model.body_mass[self.bodies]
        self.sole = duck.sole_geoms[0]

    def measure(self, q, pitch):
        """(sole pitch error, COM x minus sole centre x, COM height above the sole)."""
        m, d, duck = self.model, self.data, self.duck
        d.qpos[duck.adr:duck.adr+3] = 0.
        d.qpos[duck.adr+3:duck.adr+7] = (np.cos(pitch/2), 0., np.sin(pitch/2), 0.)
        d.qpos[duck.qidx] = q
        mujoco.mj_kinematics(m, d)
        com = self.mass @ d.xipos[self.bodies] / self.mass.sum()
        R = d.geom_xmat[self.sole].reshape(3, 3)
        points = []
        for g in duck.sole_geoms:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            points.append(m.mesh_vert[a:a+n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        points = np.vstack(points)
        return np.array([np.arctan2(R[2, 0], R[2, 2]) - SOLE_FLAT,
                         com[0] - points[:, 0].mean(), com[2] - points[:, 2].min()])

    def solve(self, base, pitch, offset, guess, height=None):
        """Left hip, knee, ankle (mirrored right): flat sole, COM `offset` ahead
        of the sole centre and, if given, COM `height` above the sole. Without
        a height the answer is the closest such pose to `guess` (min-norm)."""
        def pose(x):
            q = base.copy()
            for name, v in zip(LEG, x):
                q[INDEX['left_'+name]] = v
                q[INDEX['right_'+name]] = -v
            return q
        goal = np.array([0., offset, 0. if height is None else height])
        rows = slice(0, 2 if height is None else 3)
        x = np.array(guess, float)
        for _ in range(12):
            f = (self.measure(pose(x), pitch) - goal)[rows]
            if np.abs(f).max() < 1e-5:
                break
            J = np.column_stack([(self.measure(pose(x+e), pitch)-goal)[rows]-f for e in np.eye(3)*1e-5])/1e-5
            step = np.linalg.lstsq(J, -f, rcond=None)[0]
            x = np.clip(x+np.clip(step, -.3, .3), -1.55, 1.55)
        return pose(x), x


class CatchController:
    """Runs the catch phases. Call `act(t, feet_force)` once per 50 Hz tick."""

    def __init__(self, duck, brain, home, params: Catch):
        self.duck, self.brain, self.home, self.p = duck, brain, home, params
        self.planner = Planner(duck)
        self.phase = 'air'
        self.touchdown = self.pitch_ref = self.guess = self.last_offset = None
        self.base = home.copy()
        self.base[INDEX['neck_pitch']], self.base[INDEX['head_pitch']] = params.neck, params.head
        self.air_base = self.base.copy()
        self.air_base[INDEX['neck_pitch']] = params.air_neck
        _, self.stand_offset, self.stand_height = self.planner.measure(home, 0.)
        self.rise_height = None

    def pitch(self):
        R = self.duck.R()
        return float(np.arctan2(-R[2, 0], R[2, 2]))

    def offset(self):
        """Measured COM x minus sole centre x (forward kinematics of the readings)."""
        sole = self.duck.points(self.duck.sole_geoms)
        return float(self.duck.com()[0] - sole[:, 0].mean())

    def act(self, t, feet_force):
        p, dt = self.p, .02
        pitch, rate = self.pitch(), float(self.duck.data.qvel[self.duck.dof+4])
        if self.guess is None:
            self.guess = np.array([self.duck.q[INDEX['left_'+n]] for n in LEG])
        if self.phase == 'air' and feet_force > p.touchdown_n:
            self.phase, self.touchdown, self.pitch_ref = 'crouch', t, pitch
        if self.phase == 'stand':
            self.brain.act()
            return
        if self.phase == 'air':
            planned = pitch + rate*p.lookahead_s
            target, self.guess = self.planner.solve(self.air_base, planned, p.com_offset, self.guess)
            self.duck.hold(target)
            return
        since = t - self.touchdown
        if self.phase == 'crouch' and since >= p.crouch_s:
            self.phase = 'rise'
            self.rise_height = self.planner.measure(self.duck.target, self.pitch_ref)[2]
        if self.phase == 'crouch':
            blend, height = 0., None
            offset_ref = p.com_offset
            base = self.base
        else:
            u = np.clip((since-p.crouch_s)/max(p.rise_s, 1e-6), 0., 1.)
            blend = u*u*(3-2*u)
            step = p.pitch_rate_limit*dt
            self.pitch_ref = float(np.clip(0., self.pitch_ref-step, self.pitch_ref+step))
            height = (1-blend)*self.rise_height + blend*self.stand_height
            offset_ref = (1-blend)*p.com_offset + blend*self.stand_offset
            base = self.base + (self.home-self.base)*blend
        target, self.guess = self.planner.solve(base, self.pitch_ref, offset_ref, self.guess, height)
        offset = self.offset()
        d_offset = 0. if self.last_offset is None else (offset-self.last_offset)/dt
        self.last_offset = offset
        hip = p.pitch_gain*(pitch-self.pitch_ref) + p.pitch_damping*rate
        ankle = p.com_gain*(offset-offset_ref) + p.com_damping*d_offset
        for name, value in (('hip_pitch', hip), ('ankle', ankle)):
            target[INDEX['left_'+name]] += value
            target[INDEX['right_'+name]] -= value
        self.duck.hold(target)
        if blend >= 1. and self.pitch_ref == 0.:
            self.phase = 'stand'
