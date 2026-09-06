"""Pose-tracking reward and reset terms for the Spiral and Arabesque tasks.

These nine functions used to live at the bottom of upstream
`mjlab_microduck/tasks/mdp.py`. They were 399 lines of ours inside a 7188-line
upstream file, so every future merge from upstream had to be resolved by hand.
They are ours, nothing upstream calls them, and they belong here.

`mdp` is still imported below: these terms build on upstream helpers, and the
env cfgs mix terms from both modules.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Four private helpers these terms share with the rest of upstream mdp.py.
# Imported by name so what this module still borrows is visible in one place.
from mjlab_microduck.tasks.mdp import (
    _DEFAULT_ASSET_CFG,
    _fallen_mask,
    _forward_progress_gate,
    _servo_joint_ids,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _leg_joint_id(env: "ManagerBasedRlEnv", asset: Entity, name: str) -> int:
    """Entity-local index of one joint, by exact name, cached.

    By NAME because the roller model interleaves `passive_*` wheel joints
    between the servos — see `_servo_joint_ids`.
    """
    cache = env.__dict__.setdefault("_leg_joint_id_cache", {})
    key = (id(asset), name)
    jid = cache.get(key)
    if jid is None:
        ids, _ = asset.find_joints(f"^{name}$")
        assert len(ids) == 1, f"expected exactly one joint named {name}, got {ids}"
        jid = ids[0]
        cache[key] = jid
    return jid


def _wheel_roll_gate(env: "ManagerBasedRlEnv", ref: float) -> torch.Tensor:
    """0->1 ramp in mean forward WHEEL speed, saturating at `ref`.

    `_forward_progress_gate` measures the TRUNK's forward velocity, which a duck
    toppling forward has in abundance. Gating a pose reward on it taught exactly
    that: spiral v1 learned to extend the free leg and dive, farming the pose
    term at wheel_speed 0.0 with 11x the falls. Wheels only turn if the robot is
    actually rolling on them, so this gate cannot be satisfied by falling.
    """
    asset: Entity = env.scene["robot"]
    ids = []
    for pat in ("passive_LF_?wheel", "passive_LR_?wheel",
                "passive_RF_?wheel", "passive_RR_?wheel"):
        found, _ = asset.find_joints(pat)
        ids.extend(found)
    spin = asset.data.joint_vel[:, ids].mean(dim=1)
    return (spin.clamp(min=0.0) / ref).clamp(max=1.0)


def _upright_gate(env: "ManagerBasedRlEnv", tilt_deg: float) -> torch.Tensor:
    """1 while upright, 0 once tilted past `tilt_deg`."""
    asset: Entity = env.scene["robot"]
    return 1.0 - _fallen_mask(env, asset, 0.0, tilt_deg)


def spiral_free_leg_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    wheel_ref: float = 3.0,
    tilt_gate_deg: float = 35.0,
    min_hold_s: float = 0.8,
    hip_extension: float = 1.1,
    hip_std: float = 0.5,
    knee_std: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the figure-skating spiral: glide on ONE wheel, free leg out behind.

        reward = single_support · forward_gate · hip_shape · knee_shape

    `glide_reward` already pays for coasting on one blade with quiet legs, but it
    is indifferent to what the lifted leg does — a tucked foot scores the same as
    an extended one. This term is what makes the pose an arabesque.

    Sign convention, measured on the roller model rather than assumed: the two
    legs mirror, and BOTH extend backward opposite to their HOME sign.
    Left  hip_pitch -0.458 (home) -> foot 21 mm FORWARD; +1.2 -> 113 mm BEHIND.
    Right hip_pitch +0.458 (home) -> foot 21 mm FORWARD; -1.2 -> 113 mm BEHIND.
    So the target is +hip_extension for a free LEFT leg and -hip_extension for a
    free RIGHT one, and getting that sign backwards trains a knee-to-chest tuck.

    Multiplicative, not additive: an additive stack has a compromise basin where
    a half-hearted lift scores most of the terms. Stds are deliberately wide so
    the current policy scores visibly and there is a gradient to climb.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, 2) LEFT, RIGHT
    air_time = sensor.data.current_air_time
    assert contact_time is not None and air_time is not None
    in_contact = contact_time > 0.0
    single = (torch.sum(in_contact.float(), dim=1) == 1).float()

    # Only pay for the pose once the leg has actually been up a while. A walking
    # swing passes through a backward-extended hip too: measured on spiral v1,
    # the mean free-leg hip during ordinary stroking was +0.99 rad against a
    # +1.10 target, so this term scored nearly full marks on a walk.
    single = single * (torch.max(air_time, dim=1).values > min_hold_s).float()

    # Free leg = the one NOT in contact (only meaningful when `single`).
    free_left = (~in_contact[:, 0]).float()

    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    l_hip = q[:, _leg_joint_id(env, asset, "left_hip_pitch")]
    r_hip = q[:, _leg_joint_id(env, asset, "right_hip_pitch")]
    l_knee = q[:, _leg_joint_id(env, asset, "left_knee")]
    r_knee = q[:, _leg_joint_id(env, asset, "right_knee")]

    hip_err = free_left * (l_hip - hip_extension) + (1.0 - free_left) * (
        r_hip + hip_extension
    )
    # Straight knee: home is ~0 for both, so the free knee should stay near 0.
    knee_err = free_left * l_knee + (1.0 - free_left) * r_knee

    hip_shape = torch.exp(-torch.square(hip_err) / hip_std**2)
    knee_shape = torch.exp(-torch.square(knee_err) / knee_std**2)

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    # Both required: the wheels must be turning AND the trunk must be upright.
    # Trunk velocity alone is satisfied by a forward topple (see _wheel_roll_gate).
    roll_gate = _wheel_roll_gate(env, wheel_ref)
    upright = _upright_gate(env, tilt_gate_deg)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return (
        single * forward_gate * roll_gate * upright * hip_shape * knee_shape * active
    )


def single_support_hold_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    wheel_ref: float = 3.0,
    tilt_gate_deg: float = 35.0,
    min_hold_s: float = 0.8,
    tau_s: float = 1.5,
) -> torch.Tensor:
    """Reward HOLDING single support, saturating — pay commitment, not chatter.

    Without this the spiral pose is worth the same held for 0.1 s as for 3 s, and
    the roller recipe's `skating_air_time` actively pays swing frequency, so the
    policy strokes rather than glides.

    Rate-limited by construction: the value is 1 - exp(-t_air/tau_s), which rises
    with time on one wheel and saturates near 1. No jackpot — arriving does not
    pay, staying does, and it cannot exceed 1 however long the hold runs.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    air_time = sensor.data.current_air_time
    assert contact_time is not None and air_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    # How long the lifted foot has been off the ground.
    held = torch.max(air_time, dim=1).values

    # LATCH. Without it this pays a continuous dribble for every walking swing:
    # measured on spiral v1, the policy alternated feet every 0.11 s, was in
    # single support 92% of the time, and collected the ramp the whole while. A
    # walk is single-support most of the time -- that is what walking IS -- so
    # only sustained holds can be allowed to score at all.
    ramp = torch.where(
        held > min_hold_s,
        1.0 - torch.exp(-(held - min_hold_s) / tau_s),
        torch.zeros_like(held),
    )

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)
    roll_gate = _wheel_roll_gate(env, wheel_ref)
    upright = _upright_gate(env, tilt_gate_deg)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * ramp * forward_gate * roll_gate * upright * active


def reward_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    param: str,
    stages: list[dict],
) -> torch.Tensor:
    """Step a single reward-term PARAM through stages (cf. `reward_weight`).

    Used to ramp the spiral's `min_hold_s` latch: at the final 1.5 s the reward
    is invisible to a policy that cannot yet balance on one wheel, so the latch
    starts short enough to be reachable and tightens as the skill consolidates.
    """
    del env_ids
    value = stages[0]["value"]
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            value = stage["value"]
    env.reward_manager.get_term_cfg(reward_name).params[param] = value
    return torch.tensor([value])


def reset_spiral_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    hip_extension: float = 1.1,
    speed_range: tuple = (0.30, 0.45),
    wheel_radius: float = 0.0175,
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reverse curriculum: spawn a fraction of episodes ALREADY in the spiral.

    Spiral v3 latched both pose rewards behind a minimum hold time and both read
    exactly 0.0 for 10000 iterations — the policy never once held a single wheel
    for even 0.30 s, so the reward was never observed and the terms were dead
    weight. Lowering the latch is not the fix: v1 showed that a latch loose
    enough to be reached by a walking stride is also satisfied BY a walking
    stride. The pose has to be handed to the policy instead.

    So a fraction of resets start rolling forward, upright, on ONE wheel, with
    the free leg already extended behind — the frontier the policy otherwise
    never samples. That fraction decays to zero, so by the end it has to enter
    the pose itself.

    Free leg is chosen per env, left or right, so neither side is privileged.
    Extension sign is mirrored: +hip_extension for a free LEFT leg, -for RIGHT
    (measured, see `spiral_free_leg_reward`).
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    if n == 0:
        return

    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.6}]
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]
    if fraction <= 0.0:
        return

    chosen = torch.rand(n, device=env.device) < fraction
    if not bool(chosen.any()):
        return
    sel = env_ids[chosen]
    m = int(sel.shape[0])

    free_left = torch.rand(m, device=env.device) < 0.5

    l_hip = _leg_joint_id(env, asset, "left_hip_pitch")
    r_hip = _leg_joint_id(env, asset, "right_hip_pitch")
    l_knee = _leg_joint_id(env, asset, "left_knee")
    r_knee = _leg_joint_id(env, asset, "right_knee")

    q = asset.data.joint_pos[sel].clone()
    # Free leg: extended straight out behind. Support leg: left at its reset pose.
    q[:, l_hip] = torch.where(free_left, torch.full_like(q[:, l_hip], hip_extension), q[:, l_hip])
    q[:, l_knee] = torch.where(free_left, torch.zeros_like(q[:, l_knee]), q[:, l_knee])
    q[:, r_hip] = torch.where(~free_left, torch.full_like(q[:, r_hip], -hip_extension), q[:, r_hip])
    q[:, r_knee] = torch.where(~free_left, torch.zeros_like(q[:, r_knee]), q[:, r_knee])
    asset.write_joint_position_to_sim(q, env_ids=sel)

    # Rolling entry, no slip: base velocity and wheel spin agree (cf.
    # reset_rolling_entry) so the wheels are not dragged at the first step.
    lo, hi = speed_range
    v = torch.rand(m, device=env.device) * (hi - lo) + lo
    root_vel = torch.zeros(m, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=sel)

    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel",
                 "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=sel)


def spiral_pose_match_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    hip_extension: float = 1.1,
    std_wide: float = 1.8,
    std_tight: float = 0.6,
    wheel_ref: float = 3.0,
    tilt_gate_deg: float = 35.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the spiral POSE, two-layer, instead of gating on how long it is held.

    Technique borrowed from jonathanhawkins/microduck-lab's `imitate` behaviour
    (Apache-2.0), whose `_mi_pose_match` uses a wide Gaussian plus a tight one:
    "the wide layer keeps a gradient when the body is nowhere near the clip
    (which is where every run starts), the tight layer pays for precision".

    That is precisely what our four previous attempts lacked. Each specified the
    spiral as a CONJUNCTION of hard gates (single support AND hold > latch AND
    hip near target...), which is binary: v1's version was satisfiable by a
    walking stride, v3's was never satisfied once in 10000 iterations. A
    two-layer Gaussian on the whole-body pose has a gradient everywhere, so the
    policy can climb toward the arabesque from wherever it currently is.

    The reference is a STATIC pose, so no clip timeline is needed: the duck's
    home pose with one leg extended straight out behind. Scored against BOTH
    mirror variants, best-of, so either leg may be the free one.

    Still gated on rolling and upright, for the reason their `_mi_on_feet`
    exists: a pose reward alone is just as matchable lying on the floor.
    """
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    q = asset.data.joint_pos[:, servo_ids]

    default = asset.data.default_joint_pos[:, servo_ids]

    # Joint order is the canonical 14-servo layout: 0-4 left leg
    # (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5-8 neck/head, 9-13 right leg.
    # Extension sign mirrors: +hip_extension for a free LEFT leg, - for RIGHT.
    ref_left = default.clone()
    ref_left[:, 2] = hip_extension
    ref_left[:, 3] = 0.0
    ref_right = default.clone()
    ref_right[:, 11] = -hip_extension
    ref_right[:, 12] = 0.0

    d2_left = torch.sum(torch.square(q - ref_left), dim=1)
    d2_right = torch.sum(torch.square(q - ref_right), dim=1)
    d2 = torch.minimum(d2_left, d2_right)

    match = 0.5 * torch.exp(-d2 / std_wide**2) + 0.5 * torch.exp(-d2 / std_tight**2)

    roll_gate = _wheel_roll_gate(env, wheel_ref)
    upright = _upright_gate(env, tilt_gate_deg)
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return match * roll_gate * upright * active


def arabesque_pose_match_reward(
    env: ManagerBasedRlEnv,
    hip_extension: float = 1.1,
    hip_roll_shift: float = 0.384,
    std_wide: float = 0.9,
    std_tight: float = 0.30,
    tilt_gate_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Two-layer tracking of a one-legged arabesque, on NORMAL feet.

    Differs from `spiral_pose_match_reward` in the one way that matters: the
    reference includes the SIDEWAYS WEIGHT SHIFT. Measured on the walk model,
    a reference that only extends the free leg leaves the centre of mass 42.8 mm
    outside the support foot — already falling at t=0, which no policy can hold.
    Shifting both hip_rolls to the support side brings the CoM to +2.0 mm against
    a foot spanning -34.1..+6.3 mm, i.e. inside the support polygon. That is a
    pose the robot can actually stand in, and it is what six earlier runs were
    never asked for.

    hip_roll's range is only +/-0.384 rad, and that limit is what bounds the
    shift; a scripted PD controller holding this pose manages ~0.96 s, so the
    pose is feasible and staying in it is a real control problem, not a trick.

    Two-layer Gaussian (wide for gradient, tight for precision) as in
    jonathanhawkins/microduck-lab's `imitate`. Scored best-of against both
    mirror variants so either leg may be the free one.
    """
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    q = asset.data.joint_pos[:, servo_ids]
    default = asset.data.default_joint_pos[:, servo_ids]

    # 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5-8 neck/head,
    # 9-13 right leg. Free LEFT => stand on the RIGHT => shift both rolls +.
    ref_free_left = default.clone()
    ref_free_left[:, 2] = hip_extension
    ref_free_left[:, 3] = 0.0
    ref_free_left[:, 1] = hip_roll_shift
    ref_free_left[:, 10] = hip_roll_shift

    ref_free_right = default.clone()
    ref_free_right[:, 11] = -hip_extension
    ref_free_right[:, 12] = 0.0
    ref_free_right[:, 1] = -hip_roll_shift
    ref_free_right[:, 10] = -hip_roll_shift

    d2 = torch.minimum(
        torch.sum(torch.square(q - ref_free_left), dim=1),
        torch.sum(torch.square(q - ref_free_right), dim=1),
    )
    match = 0.5 * torch.exp(-d2 / std_wide**2) + 0.5 * torch.exp(-d2 / std_tight**2)

    # A pose reward alone is just as matchable lying on the floor.
    return match * _upright_gate(env, tilt_gate_deg)
