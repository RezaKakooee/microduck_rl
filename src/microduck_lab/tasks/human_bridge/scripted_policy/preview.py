"""Choose a gait command by testing short predictions in a separate world.

The live world receives motor targets only. Predictions start from copies
of its state. They cannot move, support or change either live robot.
"""
import copy
import math

import numpy as np
import mujoco

from microduck_lab.tasks.human_bridge.scripted_policy.world import World, Brain
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.scripted_policy.brother import HoldStraight, PLANK
from microduck_lab.tasks.human_bridge.scripted_policy.story import DROP_Z
from microduck_lab.tasks.human_bridge.scripted_policy.runtime import brain as single_thread_brain


class Preview:
    def __init__(self, supervisor):
        self.live = supervisor
        self.cfg = supervisor.cfg
        self.world = World(L.design())
        self.brain = single_thread_brain(self.world.ducks['she'])
        self.world.start()
        self.hold = HoldStraight(self.world.ducks['he'])
        self.rng = np.random.default_rng(0)
        self.next_plan = 0.
        self.best = np.array([0., self.cfg.speed, 0.])
        if self.cfg.preview_feedback:
            self.best = np.array([self.cfg.speed, 0., 0.])
        if self.cfg.preview_residual:
            self.best = np.r_[self.best, np.zeros(5 if self.cfg.preview_independent_lift else 4 if self.cfg.preview_lift else 3)]
        self.scores = []
        self.predictions = []
        self.expected = None
        self.pool = None
        if self.cfg.preview_workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            from microduck_lab.tasks.human_bridge.scripted_policy.predict_workers import initialize
            self.pool = ProcessPoolExecutor(max_workers=self.cfg.preview_workers,
                                           mp_context=multiprocessing.get_context('spawn'),
                                           initializer=initialize, initargs=(self.cfg,))

    def policy_state(self):
        src = self.live.brain.pol
        return {name: copy.deepcopy(getattr(src, name))
                for name in ('last_action', 'vel_cmd', 'command', 'body_cmd', 'head_offset',
                             'current_policy', 'action_buffer', 'buffer_index', 'current_lag')
                if hasattr(src, name)}

    def copy_policy(self):
        src, dst = self.live.brain.pol, self.brain.pol
        for name in ('walking_session', 'standing_session', 'ort_session'):
            setattr(dst, name, getattr(src, name))
        for name in ('last_action', 'vel_cmd', 'command', 'body_cmd', 'head_offset',
                     'current_policy', 'action_buffer', 'buffer_index', 'current_lag'):
            if hasattr(src, name):
                setattr(dst, name, copy.deepcopy(getattr(src, name)))

    def score(self, snapshot, command):
        w = self.world
        if 'bam_friction' in snapshot:
            # BAM reads the previous constraint forces when computing friction.
            # Copy these into the prediction only, never into the live world.
            w.model.dof_frictionloss[:] = snapshot['bam_friction']
            w.model.dof_damping[:] = snapshot['bam_damping']
        w.restore(snapshot)
        if 'bam_forces' in snapshot:
            for name, values in snapshot['bam_forces'].items():
                getattr(w.data, name)[:] = values
            w.data.qacc_warmstart[:] = snapshot['warm']
        self.copy_policy()
        he, she = w.ducks['he'], w.ducks['she']
        self.hold.extra[:] = self.live.brother_hold.extra
        x0 = she.pos()[0]
        cost = 0.
        count = round(self.cfg.preview_horizon / .02)
        for i in range(count):
            parameters = self.parameters_at(command, i*.02, self.cfg.preview_horizon)
            # The live trial has already set his targets for this tick.
            if i:
                self.hold()
            if self.live.table is not None:
                from microduck_lab.tasks.human_bridge.scripted_policy.story import bridge_state
                self.brain.pol.body_cmd = bridge_state(she, self.live.table)
            with self.brain.quiet():
                self.brain.pol.set_vel_cmd(*self.body_command(she, parameters))
            self.brain.act()
            if self.cfg.preview_residual:
                self.apply_residual(she, parameters, self.cfg)
            w.step()
            mujoco.mj_forward(w.model, w.data)
            if i == 0:
                self.predictions.append(w.data.qpos.copy())
            x, y, z = she.pos()
            up_limit = .5 if self.cfg.preview_up_cost else self.cfg.preview_min_up
            if she.up() <= up_limit or z < DROP_Z or abs(he.up()) >= .35:
                return -1.e6 - (count-i)/count + 5*(x-x0) - cost/count
            yaw_error = math.atan2(math.sin(she.yaw()+math.pi/2), math.cos(she.yaw()+math.pi/2))
            cost += (self.cfg.preview_center_cost*((y-self.cfg.y_target)/.025)**2
                     + self.cfg.preview_heading_cost*(yaw_error/.3)**2)
            cost += .15 * (1-she.up()) + .15 * max(0., (abs(y)-.035)/.02)**2
            cost += self.cfg.preview_up_cost * max(0., (self.cfg.preview_min_up-she.up())/.2)**2
            cost += self.cfg.preview_height_cost * max(0., (self.cfg.preview_height_target-z)/.05)**2
            if self.cfg.preview_foot_cost:
                for foot in she.feet():
                    if L.near_edge < foot[0] < L.far_edge:
                        cost += self.cfg.preview_foot_cost * (max(0., abs(foot[1])-.005)/.025)**2
            if self.cfg.preview_support_cost:
                from microduck_lab.tasks.human_bridge.scripted_policy.metrics import nonfoot_support
                force = nonfoot_support(she)
                cost += self.cfg.preview_support_cost * min(1., force/7.4)
        capture = she.pos()[1] + .14*she.data.qvel[she.dof+1] - self.cfg.y_target
        terminal_cost = self.cfg.preview_capture_cost * (capture/.025)**2
        return (self.terminal_value(x0, she.pos()[0], she.up(),
                                    she.data.qvel[she.dof:she.dof+3])
                - cost/count - terminal_cost)

    def terminal_value(self, start_x, end_x, up, velocity):
        if self.cfg.preview_finish:
            # Penalize overshoot even when a prediction starts on the bridge.
            progress = abs(start_x-.55) - abs(end_x-.55)
            return (self.cfg.preview_progress*progress - .5*(1-up)
                    - self.cfg.preview_finish_speed_cost*np.linalg.norm(velocity))
        return self.cfg.preview_progress*(end_x-start_x)

    @staticmethod
    def parameters_at(sequence, time, horizon):
        if sequence.ndim == 1:
            return sequence
        u = np.clip(time/horizon, 0., 1.)*(len(sequence)-1)
        i = min(int(u), len(sequence)-2)
        f = u-i
        return (1-f)*sequence[i] + f*sequence[i+1]

    def current_parameters(self):
        return self.best

    def snapshot(self):
        live = self.live.duck.world
        snapshot = live.snapshot()
        if vars(self.cfg).get('preview_complete_snapshot', False):
            snapshot['bam_friction'] = live.model.dof_frictionloss.copy()
            snapshot['bam_damping'] = live.model.dof_damping.copy()
            snapshot['bam_forces'] = {name: getattr(live.data, name).copy()
                                      for name in ('qfrc_bias', 'qfrc_constraint', 'qfrc_actuator', 'efc_force')}
        return snapshot

    def evaluate(self, snapshot, samples):
        self.predictions = []
        if self.pool is None:
            return [self.score(snapshot, cmd) for cmd in samples]
        from microduck_lab.tasks.human_bridge.scripted_policy.predict_workers import evaluate
        state = self.policy_state()
        payloads = [(snapshot, state, self.live.brother_hold.extra, self.live.table, cmd)
                    for cmd in samples]
        results = list(self.pool.map(evaluate, payloads))
        self.predictions = [r[1] for r in results]
        return [r[0] for r in results]

    @staticmethod
    def apply_residual(duck, command, cfg=None):
        """Small pitch, roll and neck offsets. The motors do all the work."""
        q = duck.target.copy()
        pitch_offset = command[3]
        if cfg is not None and cfg.preview_balance:
            rotation = duck.R()
            pitch = math.atan2(-rotation[2, 0], rotation[2, 2])
            forward = rotation[:2, 0]
            error = forward[1]*(cfg.y_target-duck.pos()[1])
            speed = forward @ duck.data.qvel[duck.dof:duck.dof+2]
            target_pitch = np.clip(3.*error - .6*speed, -.25, .25)
            sensor = mujoco.mj_name2id(duck.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                      f'{duck.prefix}_imu_ang_vel')
            rate = duck.data.sensordata[duck.model.sensor_adr[sensor]+1]
            pitch_offset += cfg.preview_balance*(pitch-target_pitch) + .15*rate
            pitch_offset = np.clip(pitch_offset, -.75, .75)
        q[4] += pitch_offset
        q[13] -= pitch_offset
        q[1] += command[4]
        q[10] += command[4]
        q[5] += command[5]
        if len(command) > 7:
            left, right = command[6:8]
            q[2:5] += left * np.array([.5, -1., .5])
            q[11:14] += right * np.array([-.5, 1., -.5])
        elif len(command) > 6:
            # Keep the gait's phase. Flex its more bent knee further while
            # keeping the sum of hip, knee and ankle pitch unchanged.
            lift = command[6]
            if not hasattr(duck, '_lift_kinematics'):
                duck._lift_kinematics = mujoco.MjData(duck.model)
            k = duck._lift_kinematics
            k.qpos[:] = duck.data.qpos
            k.qpos[duck.qidx] = np.clip(duck.target, duck.lo, duck.hi)
            mujoco.mj_kinematics(duck.model, k)
            heights = k.site_xpos[list(duck.foot_sites)] @ duck.R()[:, 2]
            if heights[0] > heights[1]:
                q[2] += lift/2
                q[3] -= lift
                q[4] += lift/2
            else:
                q[11] -= lift/2
                q[12] += lift
                q[13] -= lift/2
        duck.hold(q)

    def body_command(self, duck, parameters):
        if not self.cfg.preview_feedback:
            return parameters[:3]
        yaw = duck.yaw()
        error = math.atan2(math.sin(yaw+math.pi/2), math.cos(yaw+math.pi/2))
        vx = parameters[0]
        vy = (parameters[1] - self.cfg.lateral_gain*(duck.pos()[1]-self.cfg.y_target)
              - self.cfg.lateral_damping*duck.data.qvel[duck.dof+1])
        wz = parameters[2] - self.cfg.heading_gain*error
        return np.array([math.cos(yaw)*vx+math.sin(yaw)*vy,
                         -math.sin(yaw)*vx+math.cos(yaw)*vy, np.clip(wz, -1.2, 1.2)])

    def command(self, nominal):
        tick = (round(self.live.duck.world.t / .02) if self.cfg.preview_tick_timing
                else self.live.duck.world.t)
        if tick < self.next_plan:
            return self.body_command(self.live.duck, self.best)
        interval = (max(1, round(self.cfg.preview_interval / .02)) if self.cfg.preview_tick_timing
                    else self.cfg.preview_interval)
        self.next_plan = tick + interval
        snapshot = self.snapshot()
        samples = [nominal.copy(), np.zeros(3)]
        samples += [np.array([vx, vy, wz]) for vx in (-.16, 0., .16)
                    for vy in (.06, .18) for wz in (-.6, .6)]
        if self.cfg.preview_feedback:
            samples[0] = np.array([self.cfg.speed, 0., 0.])
            samples[2:] = [s[[1, 0, 2]] for s in samples[2:]]
        if self.cfg.preview_residual:
            count = 4 if self.cfg.preview_lift else 3
            samples = [np.r_[s, np.zeros(count)] for s in samples]
            std = [.06, .05, .3, .15, .10, .15] + ([.2] if self.cfg.preview_lift else [])
            samples += [self.best+self.rng.normal(0., std)
                        for _ in range(self.cfg.preview_samples)]
            low, high = [-.3, -.12, -1.2, -.5, -.3, -.5], [.3, .3, 1.2, .5, .3, .5]
            if self.cfg.preview_lift:
                low.append(0.)
                high.append(self.cfg.preview_lift_cap)
        else:
            samples += [self.best+self.rng.normal(0., [.08, .06, .4]) for _ in range(self.cfg.preview_samples)]
            low, high = [-.3, -.12, -1.2], [.3, .3, 1.2]
        samples.append(self.best.copy())
        if self.cfg.preview_feedback:
            low[:2], high[:2] = [-.12, -.3], [.3, .3]
        samples = np.clip(samples, low, high)
        scores = self.evaluate(snapshot, samples)
        best = int(np.argmax(scores))
        self.best = samples[best].copy()
        self.expected = (self.live.duck.world.t + .02, self.predictions[best])
        self.scores.append([self.live.duck.world.t, float(scores[best]), *self.best])
        return self.body_command(self.live.duck, self.best)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
