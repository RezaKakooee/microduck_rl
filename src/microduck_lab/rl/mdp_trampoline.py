"""MDP terms for the trampoline flip (microduck_trampoline_flip_env_cfg.py).

State per env (env._tramp): the forward rotation integrated while the robot
is in the air, its frontier (max so far), the part of it already paid, and
the highest point reached above the bed. Updated once per control step by
whichever term reads it first.

Sign conventions (mdp.py's two styles): the *_cost functions return >= 0 and
take NEGATIVE weights; the rewards return >= 0 and take positive weights.
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv

from microduck_lab.rl.trampoline_scene import BED_TOP

FEET_SENSOR = "feet_bed_contact"
BODY_SENSOR = "body_bed_contact"
FLOOR_SENSOR = "robot_floor_contact"
STAND_ROOT_Z = 0.125      # trunk height above the soles, standing (CPU jump/trampoline worlds)
MIN_FLIGHT_S = 0.08       # a landing after at least this long in the air counts as a bounce


def _any(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _state(env: ManagerBasedRlEnv) -> dict:
    if not hasattr(env, "_tramp"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._tramp = dict(accum=z.clone(), frontier=z.clone(), paid=z.clone(),
                          apex=z.clone(), apex_paid=z.clone(), bounces=z.clone(), bounces_paid=z.clone(),
                          air_time=z.clone(), flight_apex=z.clone(), landed=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                          landed_apex=z.clone(), min_bounces=0.0, per_flight=False, last=-1)
    return env._tramp


def _update(env: ManagerBasedRlEnv) -> dict:
    """Integrate the forward pitch rate while airborne; count bounces; track the highest point.

    The rotation only counts once the robot has bounced min_bounces times (0
    in the flip task; the pump-flip task asks for a few bounces first).
    """
    s = _state(env)
    step = int(env.common_step_counter)
    if step == s["last"]:
        return s
    robot = env.scene["robot"]
    feet = _any(env, FEET_SENSOR)
    air = ~(feet | _any(env, BODY_SENSOR) | _any(env, FLOOR_SENSOR))
    landed = feet & (s["air_time"] >= MIN_FLIGHT_S)
    s["bounces"] = s["bounces"] + landed.float()
    s["air_time"] = torch.where(air, s["air_time"] + env.step_dt, torch.zeros_like(s["air_time"]))
    # Highest point of the current flight (root above standing on the bed at rest); kept at landing.
    h_now = torch.nan_to_num(robot.data.root_link_pos_w[:, 2] - (BED_TOP + STAND_ROOT_Z), nan=0.0)
    s["landed"] = landed
    s["air"], s["h_now"] = air, h_now
    s["landed_apex"] = torch.where(landed, s["flight_apex"], s["landed_apex"])
    s["flight_apex"] = torch.where(air, torch.maximum(s["flight_apex"], h_now), torch.full_like(h_now, -1.0))
    allowed = (s["bounces"] >= s["min_bounces"]).float()
    if s["per_flight"]:
        # A flip happens inside ONE flight: the count restarts at every touch-down.
        # Without this, run 4 tipped ~60 deg forward on each of ~6 bounces (and
        # back on the bed, uncounted) and "completed" flips it never did.
        s["accum"] = torch.where(air, s["accum"], torch.zeros_like(s["accum"]))
    # Body-frame pitch rate: > 0 turns nose down (a front flip).
    omega = torch.nan_to_num(robot.data.root_link_ang_vel_b[:, 1], nan=0.0)
    s["accum"] = s["accum"] + omega * env.step_dt * air.float() * allowed
    s["frontier"] = torch.maximum(s["frontier"], s["accum"])
    h = torch.nan_to_num(robot.data.root_link_pos_w[:, 2] - (BED_TOP + STAND_ROOT_Z), nan=0.0)
    s["apex"] = torch.maximum(s["apex"], h)
    s["last"] = step
    return s


def _gate(frontier: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = torch.clamp((frontier - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
    g = env.scene["robot"].data.projected_gravity_b
    return torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))


# ── Rewards ──────────────────────────────────────────────────────────────────

def flip_progress(env: ManagerBasedRlEnv, target_angle: float = 2 * math.pi,
                  max_paid_rate: float = 20.0) -> torch.Tensor:
    """Pay increments of the airborne rotation frontier, up to one full turn.

    Potential-based: a whole flip pays 1 x weight in total, turning on the bed
    or rocking below the frontier pays nothing. Faster than max_paid_rate
    rad/s forfeits the excess.
    """
    s = _update(env)
    new_paid = torch.clamp(s["frontier"], max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(s["paid"], max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    s["paid"] = torch.maximum(s["paid"], new_paid)
    return delta / (env.step_dt * target_angle)


def height_progress(env: ManagerBasedRlEnv, cap: float = 0.5) -> torch.Tensor:
    """Pay increments of the highest point reached (root above standing on the bed), up to cap m."""
    s = _update(env)
    new_paid = torch.clamp(s["apex"], max=cap)
    delta = torch.clamp(new_paid - torch.clamp(s["apex_paid"], max=cap), min=0.0)
    s["apex_paid"] = torch.maximum(s["apex_paid"], new_paid)
    return delta / (env.step_dt * cap)


def bounce_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pay each bounce, up to the number the flip waits for (one-off, weight per bounce)."""
    s = _update(env)
    cap = float(s["min_bounces"])
    new_paid = torch.clamp(s["bounces"], max=cap)
    delta = torch.clamp(new_paid - s["bounces_paid"], min=0.0)
    s["bounces_paid"] = torch.maximum(s["bounces_paid"], new_paid)
    return delta / env.step_dt


def fall_cost(env: ManagerBasedRlEnv, only_before_flip: bool = False) -> torch.Tensor:
    """1 on the step the robot falls (body on the bed, or anything on the floor); weight = cost per fall.

    Without it a crash scored the same as standing still (0), and the pump-flip
    run kept its old habit of diving into a flip straight from standing.
    only_before_flip: charge it only while the flip is not yet allowed (fewer
    than min_bounces bounces). Charged always (run 5), it also taxes every
    failed flip attempt, and both run-5 policies just bounced or stood safely.
    """
    fell = (_any(env, BODY_SENSOR) | _any(env, FLOOR_SENSOR)).float()
    if only_before_flip:
        s = _update(env)
        fell = fell * (s["bounces"] < s["min_bounces"]).float()
    return fell / env.step_dt


def bounce_height(env: ManagerBasedRlEnv, target: float = 0.12, pose_std: float | None = None) -> torch.Tensor:
    """At each landing, pay min(apex / target, 1) for that flight's highest point (weight per bounce).

    The bounce task's main term: every good bounce pays, not only the first
    few (the pump-flip runs earned 1 point per bounce up to 3, so standing
    still was the safe choice and they never pumped from rest).

    pose_std (rad): also multiply by exp(-(|q - default| / pose_std)^2), with
    q the 14 servo angles at the top of that flight. The flip policy takes
    over at the top and only flips from the standing (default) pose; bounce2
    flew with its neck 103 deg off it, and moving it back in the air tipped
    the trunk ~40 deg.
    """
    s = _update(env)
    pay = s["landed"].float() * torch.clamp(s["landed_apex"] / target, 0.0, 1.0)
    if pose_std is not None:
        from mjlab_microduck.tasks.mdp import _servo_default_joint_pos, _servo_joint_pos
        robot = env.scene["robot"]
        err = s.setdefault("apex_pose_err", torch.zeros(env.num_envs, device=env.device))
        pay = pay * torch.exp(-err / pose_std ** 2)      # the flight that just ended
        now = torch.nan_to_num(((_servo_joint_pos(env, robot) - _servo_default_joint_pos(env, robot)) ** 2).sum(-1), nan=1e3)
        at_top = s["air"] & (s["h_now"] >= s["flight_apex"])
        s["apex_pose_err"] = torch.where(at_top, now, torch.where(s["air"], err, torch.zeros_like(err)))
    return pay / env.step_dt


def upright(env: ManagerBasedRlEnv, std: float = 0.4) -> torch.Tensor:
    """Gaussian of the tilt, every step (the bounce task has no flip to oppose)."""
    return torch.exp(-(_tilt(env) / std) ** 2)


def upright_after_flip(env: ManagerBasedRlEnv, gate_lo: float = math.radians(300.0),
                       gate_hi: float = math.radians(340.0), std: float = 0.4) -> torch.Tensor:
    """Upright (Gaussian of the tilt), paid per step once the flip is (nearly) complete."""
    s = _update(env)
    return _gate(s["frontier"], gate_lo, gate_hi) * torch.exp(-(_tilt(env) / std) ** 2)


def roll_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sideways tilt: gravity along the body's y axis (0 in a clean sagittal flip)."""
    return env.scene["robot"].data.projected_gravity_b[:, 1] ** 2


def lateral_velocity_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Speed along the body's y axis (stays lateral while the body pitches)."""
    return env.scene["robot"].data.root_link_lin_vel_b[:, 1] ** 2


def off_center_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Squared distance of the trunk from the bed centre (m^2)."""
    return (env.scene["robot"].data.root_link_pos_w[:, :2] ** 2).sum(dim=-1)


# ── Observations (critic only: the actor keeps the 61-D contract) ────────────

def bed_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    bed = env.scene["bed"]
    return torch.stack([bed.data.joint_pos[:, 0], bed.data.joint_vel[:, 0]], dim=-1)


def height_above_bed(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.scene["robot"].data.root_link_pos_w[:, 2] - BED_TOP).unsqueeze(-1)


def flip_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    s = _update(env)
    return torch.stack([s["frontier"], s["accum"]], dim=-1) / (2 * math.pi)


def feet_on_bed(env: ManagerBasedRlEnv) -> torch.Tensor:
    found = env.scene.sensors[FEET_SENSOR].data.found
    return (found.view(found.shape[0], -1) > 0).float()


def root_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, :2]


# ── Terminations ─────────────────────────────────────────────────────────────

def body_on_bed(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Any robot part other than the feet touches the bed: a fall."""
    return _any(env, BODY_SENSOR)


def touches_floor(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Any robot part touches the floor: off the trampoline."""
    return _any(env, FLOOR_SENSOR)


# ── Curriculum ───────────────────────────────────────────────────────────────

def relative_event_param_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str,
                                    param_stages: list) -> torch.Tensor:
    """mdp.event_param_curriculum, with the stage steps counted from the start of THIS run.

    When a run starts from another run's checkpoint, mjlab restores that run's
    step counter (run 7 started at 4749 x 24 steps and skipped every stage).
    Here a jump of the counter (the restore) starts the stage clock again.
    """
    del env_ids
    step = int(env.common_step_counter)
    last = getattr(env, "_tramp_curr_last", None)
    if last is None or step < last or step - last > 5000:
        env._tramp_curr_start = step
    env._tramp_curr_last = step
    rel = step - env._tramp_curr_start
    current = param_stages[0]["params"]
    for stage in param_stages:
        if rel >= stage["step"]:
            current = stage["params"]
    env.event_manager.get_term_cfg(event_name).params.update(current)
    return torch.tensor(float(param_stages.index(next(st for st in param_stages if st["params"] is current))))


# ── Reset ────────────────────────────────────────────────────────────────────

def reset_trampoline_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    drop_prob: float = 0.5,
    midflip_prob: float = 0.5,
    stand_prob: float = 0.0,
    min_bounces: int = 0,
    per_flight: bool = False,
    stand_sag: float = 0.080,
    drop_height_range: tuple = (0.05, 0.5),
    xy_range: float = 0.05,
    tilt_max: float = math.radians(3.0),
    midflip_angle_range: tuple = (math.radians(60.0), math.radians(330.0)),
    midflip_height_range: tuple = (0.25, 0.6),
    midflip_omega_range: tuple = (4.0, 12.0),
    midflip_vz_range: tuple = (-1.0, 1.5),
    midflip_vx_range: tuple = (0.0, 0.6),
    midflip_x_range: tuple = (-0.2, 0.1),
    tuck_overrides: dict | None = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.03,
) -> None:
    """Spawn standing still on the settled bed, above it (a drop), or in the air mid-flip.

    Standing: the bed starts at its static sag (stand_sag, the nominal bed's)
    with the robot on it, still. Standing and drop spawns start with 0
    bounces; mid-flip spawns with min_bounces (their flip is allowed).

    Must run after reset_robot_joints (it starts from the default joint pose).
    The bed is put back at rest. Mid-flip spawns turn the robot `angle` into
    a front flip (nose down), spinning forward, legs lerped towards the tuck;
    the rotation state is preset to that angle, so only the rest of the turn
    is paid and the landing gate opens as it would in a real flip.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n = len(env_ids)
    dev = env.device
    robot = env.scene["robot"]
    s = _state(env)

    def uni(lo, hi):
        return torch.rand(n, device=dev) * (hi - lo) + lo

    total = max(drop_prob + midflip_prob + stand_prob, 1e-6)
    r = torch.rand(n, device=dev) * total
    stand = r < stand_prob
    mid = r >= stand_prob + drop_prob
    s["min_bounces"] = float(min_bounces)
    s["per_flight"] = bool(per_flight)

    pitch = torch.where(mid, uni(*midflip_angle_range), uni(-tilt_max, tilt_max))
    roll = uni(-tilt_max, tilt_max)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    quat = torch.stack([cr * cp, sr * cp, cr * sp, -sr * sp], dim=-1)   # roll then pitch, yaw 0

    root = torch.zeros(n, 13, device=dev)
    root[:, 0] = torch.where(mid, uni(*midflip_x_range), uni(-xy_range, xy_range))
    root[:, 1] = uni(-xy_range, xy_range)
    root[:, 2] = torch.where(mid, BED_TOP + uni(*midflip_height_range),
                             BED_TOP + STAND_ROOT_Z + uni(*drop_height_range))
    root[:, 2] = torch.where(stand, torch.full_like(root[:, 2], BED_TOP - stand_sag + STAND_ROOT_Z + 0.002), root[:, 2])
    root[:, 3:7] = quat
    root[:, 7] = torch.where(mid, uni(*midflip_vx_range), torch.zeros(n, device=dev))
    root[:, 9] = torch.where(mid, uni(*midflip_vz_range), torch.zeros(n, device=dev))
    root[:, 11] = torch.where(mid, uni(*midflip_omega_range), torch.zeros(n, device=dev))  # world y: nose down
    robot.write_root_state_to_sim(root, env_ids=env_ids)

    # Imported here: mjlab_microduck.tasks.mdp loads the task package, which
    # registers this env, which imports this module (a cycle at import time).
    from mjlab_microduck.tasks.mdp import _servo_joint_ids
    jpos = robot.data.default_joint_pos[env_ids].clone()
    servo = _servo_joint_ids(env, robot)
    if tuck_overrides:
        u = uni(*tuck_factor_range) * mid.float()
        for j, angle in tuck_overrides.items():
            col = servo[j]
            jpos[:, col] = jpos[:, col] + u * (angle - jpos[:, col])
    if joint_noise_std > 0:
        cols = torch.as_tensor(servo, device=dev, dtype=torch.long)
        before = jpos.clone()
        jpos[:, cols] += torch.randn(n, len(servo), device=dev) * joint_noise_std
        # Wide noise (FlipAnyPose) must stay inside the limits. Only the noise is clipped: a pose
        # already outside (the tuck's neck_pitch 1.15 > 0.92) is not moved further out, nor pulled in.
        lim = robot.data.soft_joint_pos_limits[env_ids]
        jpos = torch.clamp(jpos, torch.minimum(lim[..., 0], before), torch.maximum(lim[..., 1], before))
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)

    bed = env.scene["bed"]
    zero = torch.zeros(n, bed.data.joint_pos.shape[1], device=dev)
    bed_pos = zero.clone()
    bed_pos[:, 0] = torch.where(stand, torch.full((n,), -stand_sag, device=dev), torch.zeros(n, device=dev))
    bed.write_joint_state_to_sim(bed_pos, zero.clone(), env_ids=env_ids)
    s["bounces"][env_ids] = torch.where(mid, torch.full((n,), float(min_bounces), device=dev), torch.zeros(n, device=dev))
    s["bounces_paid"][env_ids] = s["bounces"][env_ids]
    s["air_time"][env_ids] = 0.0
    s["flight_apex"][env_ids] = -1.0
    s["landed"][env_ids] = False

    angle = torch.where(mid, pitch, torch.zeros(n, device=dev))
    s["accum"][env_ids] = angle
    s["frontier"][env_ids] = angle
    s["paid"][env_ids] = angle
    h0 = root[:, 2] - (BED_TOP + STAND_ROOT_Z)
    s["apex"][env_ids] = h0
    s["apex_paid"][env_ids] = h0
