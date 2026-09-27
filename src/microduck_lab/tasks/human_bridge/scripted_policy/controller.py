"""A sideways gait with world-position and heading feedback."""
from dataclasses import dataclass
import math

import numpy as np
import mujoco

HEADING = -math.pi / 2


@dataclass
class Settings:
    policy: str = "local_storage/hb_dev/scripted_policy/policies/sideways_v2_iter4750.onnx"
    speed: float = 0.12
    lateral_gain: float = 2.0
    heading_gain: float = 3.0
    lateral_damping: float = 0.0
    yaw_damping: float = 0.0
    y_target: float = 0.0
    lift: float = 0.0
    lift_start: float = -0.18
    lift_cap: float = 0.045
    terrain: float = 0.0
    foot_center: float = 0.0
    body_center: float = 0.0
    balance: float = 0.0
    balance_damping: float = 0.1
    integral_gain: float = 0.0
    lateral_limit: float = 0.12
    bridge_observations: bool = False
    mode: str = "policy"
    step_time: float = .22
    stride: float = .035
    step_height: float = .025
    body_height: float = .116
    sway: float = .008
    gait_y_bias: float = 0.
    gait_feedback: float = 0.
    gait_damping: float = 0.
    joint_integral: float = 0.
    preview_horizon: float = .6
    preview_interval: float = .12
    preview_samples: int = 10
    preview_progress: float = 10.
    preview_residual: bool = False
    landing_margin: float = .05
    landing_up: float = .8
    landing_support_limit: float = -1.
    landing_entry_speed: float = -1.
    landing_entry_height: float = -1.
    preview_finish: bool = False
    preview_finish_speed_cost: float = .5
    preview_feedback: bool = False
    preview_workers: int = 1
    settle_with_preview: bool = False
    preview_tick_timing: bool = True
    preview_min_up: float = .5
    preview_height_cost: float = 0.
    preview_height_target: float = .42
    preview_support_cost: float = 0.
    preview_complete_snapshot: bool = True
    preview_lift: bool = False
    preview_independent_lift: bool = False
    preview_balance: float = 0.
    preview_center_cost: float = .10
    preview_heading_cost: float = .04
    preview_sequence: bool = False
    preview_knots: int = 3
    preview_iterations: int = 2
    preview_capture_cost: float = 0.
    preview_up_cost: float = 0.
    preview_lift_cap: float = .8
    preview_foot_cost: float = 0.

    def __post_init__(self):
        if self.preview_independent_lift and not (self.preview_lift and self.preview_residual and self.preview_sequence):
            raise ValueError("Independent lifts require sequence planning with lift residuals")
        if self.mode not in ("policy", "gait", "preview"):
            raise ValueError("mode must be policy, gait or preview")
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.step_time <= 0 or self.preview_horizon < .02 or self.preview_interval < .02:
            raise ValueError("Step and preview times must be positive control intervals")
        if self.preview_samples < 0 or self.lateral_limit < 0 or self.preview_workers < 1:
            raise ValueError("Sample count and command limits must be nonnegative")
        if self.preview_knots < 2 or self.preview_iterations < 1:
            raise ValueError("A sequence needs at least two knots and one search iteration")


class Supervisor:
    def __init__(self, brain, settings):
        self.brain, self.duck, self.cfg = brain, brain.duck, settings
        self.command = np.zeros(3)
        self.phase = "wait"
        self.kinematics = mujoco.MjData(self.duck.model)
        self.surface = None
        self.surface_time = -10.
        self.y_integral = 0.
        self.gait = None
        self.preview = None
        self.table = None
        if settings.bridge_observations:
            from microduck_lab.tasks.human_bridge.scripted_policy.baked_surface import surface_table
            self.table = surface_table()

    def terrain_targets(self, target):
        """Retarget foot heights to his back, without moving the live root."""
        from microduck_lab.tasks.human_bridge.scripted_policy.surface import Surface
        d, m, k = self.duck, self.duck.model, self.kinematics
        if self.surface is None:
            self.surface = Surface(d.world.ducks['he'])
        if d.world.t - self.surface_time >= .5:
            self.sx = np.arange(-.30, .451, .005)
            self.sz = np.array([max(self.surface.height(x, y) for y in (-.02, 0., .02)) for x in self.sx])
            self.surface_time = d.world.t
        height = lambda x: max(np.interp(x + offset, self.sx, self.sz) for offset in (-.012, 0., .012))
        ground = np.mean([height(p[0]) for p in d.feet()])
        k.qpos[:] = d.data.qpos
        k.qpos[d.qidx] = np.clip(target, d.lo, d.hi)
        mujoco.mj_forward(m, k)
        for side, site in enumerate(d.foot_sites):
            desired = k.site_xpos[site].copy()
            desired[2] += self.cfg.terrain * np.clip(height(desired[0]) - ground, -.04, .04)
            desired[1] += self.cfg.foot_center * np.clip(self.cfg.y_target - desired[1], -.03, .03)
            desired[1] += self.cfg.body_center * np.clip(d.pos()[1] - self.cfg.y_target
                            + .12 * d.data.qvel[d.dof+1], -.04, .04)
            rotation = k.site_xmat[site].reshape(3, 3).copy()
            ids = np.arange(0, 5) if side == 0 else np.arange(9, 14)
            self.solve_foot(k, site, ids, desired, rotation)
            target[ids] = k.qpos[d.qidx[ids]]
        return target

    def solve_foot(self, k, site, ids, desired, rotation):
        d, m = self.duck, self.duck.model
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(5):
            mujoco.mj_jacSite(m, k, jp, jr, site)
            current = k.site_xmat[site].reshape(3, 3)
            angle = sum(np.cross(current[:, i], rotation[:, i]) for i in range(3)) * .5
            j = np.vstack((jp[:, d.vidx[ids]], .03 * jr[:, d.vidx[ids]]))
            error = np.r_[desired - k.site_xpos[site], .03 * angle]
            dq = np.linalg.solve(j.T @ j + np.eye(5) * 1e-6, j.T @ error)
            k.qpos[d.qidx[ids]] = np.clip(k.qpos[d.qidx[ids]] + np.clip(dq, -.2, .2), d.lo[ids], d.hi[ids])
            mujoco.mj_forward(m, k)

    def lift_foot(self, target):
        """Raise the gait's higher foot using leg IK in scratch data.

        The live state is read only. Keep the commanded sole orientation.
        The other leg still receives the gait's original command.
        """
        d, m, k = self.duck, self.duck.model, self.kinematics
        k.qpos[:] = d.data.qpos
        k.qpos[d.qidx] = np.clip(target, d.lo, d.hi)
        mujoco.mj_forward(m, k)
        positions = k.site_xpos[list(d.foot_sites)].copy()
        local = positions @ d.R()
        side = int(np.argmax(local[:, 2]))
        dz = min(self.cfg.lift_cap, self.cfg.lift * max(0., local[side, 2] - local[1-side, 2]))
        desired = positions[side] + d.R()[:, 2] * dz
        site = d.foot_sites[side]
        rotation = k.site_xmat[site].reshape(3, 3).copy()
        ids = np.arange(0, 5) if side == 0 else np.arange(9, 14)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(5):
            mujoco.mj_jacSite(m, k, jp, jr, site)
            current = k.site_xmat[site].reshape(3, 3)
            angle = sum(np.cross(current[:, i], rotation[:, i]) for i in range(3)) * .5
            j = np.vstack((jp[:, d.vidx[ids]], .03 * jr[:, d.vidx[ids]]))
            error = np.r_[desired - k.site_xpos[site], .03 * angle]
            dq = np.linalg.solve(j.T @ j + np.eye(5) * 1e-6, j.T @ error)
            k.qpos[d.qidx[ids]] = np.clip(k.qpos[d.qidx[ids]] + np.clip(dq, -.2, .2), d.lo[ids], d.hi[ids])
            mujoco.mj_forward(m, k)
        target[ids] = k.qpos[d.qidx[ids]]
        return target

    def act(self, walking):
        d, c = self.duck, self.cfg
        if walking and c.mode == "gait":
            from microduck_lab.tasks.human_bridge.scripted_policy.gait import Gait
            if self.gait is None:
                self.gait = Gait(self)
            self.gait.act()
            self.phase = "step_left" if self.gait.step % 2 == 0 else "step_right"
            return
        if walking:
            self.phase = "cross"
            yaw = d.yaw()
            error = math.atan2(math.sin(yaw - HEADING), math.cos(yaw - HEADING))
            vel = d.data.qvel[d.dof:d.dof + 3]
            self.y_integral = np.clip(self.y_integral + .02 * (d.pos()[1] - c.y_target), -.1, .1)
            vy_world = np.clip(-c.lateral_gain * (d.pos()[1] - c.y_target)
                               - c.lateral_damping * vel[1] - c.integral_gain*self.y_integral,
                               -c.lateral_limit, c.lateral_limit)
            # Rotate a world velocity into her current body frame.
            vx = math.cos(yaw) * c.speed + math.sin(yaw) * vy_world
            vy = -math.sin(yaw) * c.speed + math.cos(yaw) * vy_world
            wz = np.clip(-c.heading_gain * error - c.yaw_damping * d.data.qvel[d.dof + 5], -1.2, 1.2)
            self.command[:] = vx, vy, wz
        else:
            self.command[:] = 0
        if walking and c.mode == "preview" and d.pos()[0] > -.19:
            from microduck_lab.tasks.human_bridge.scripted_policy.preview import Preview
            if self.preview is None:
                if c.preview_sequence:
                    from microduck_lab.tasks.human_bridge.scripted_policy.sequence import SequencePreview
                    self.preview = SequencePreview(self)
                else:
                    self.preview = Preview(self)
            self.command[:] = self.preview.command(self.command)
        with self.brain.quiet():
            if self.table is not None:
                from microduck_lab.tasks.human_bridge.scripted_policy.story import bridge_state
                self.brain.pol.body_cmd = bridge_state(d, self.table) if walking else np.zeros(6, np.float32)
            self.brain.pol.set_vel_cmd(*self.command)
        self.brain.act()
        if walking and self.preview is not None and c.preview_residual:
            self.preview.apply_residual(d, self.preview.current_parameters(), c)
        if walking and c.lift and c.lift_start < d.pos()[0] < .32:
            d.hold(self.lift_foot(d.target.copy()))
        if walking and (c.terrain or c.foot_center or c.body_center) and c.lift_start < d.pos()[0] < .32:
            d.hold(self.terrain_targets(d.target.copy()))
        if walking and c.balance:
            pitch = math.atan2(-d.R()[2, 0], d.R()[2, 2])
            rate = self.brain.pol.get_base_ang_vel()[1]
            correction = np.clip(c.balance * pitch + c.balance_damping * rate, -.5, .5)
            target = d.target.copy()
            target[4] += correction
            target[13] -= correction
            d.hold(target)
