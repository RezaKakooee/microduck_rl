"""Sideways beam task invariants (CPU only).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest src/microduck_lab/tests/test_sideways_beam_cfg.py -q
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

# Load the task package first (registers every task; see test_sideways_cfg.py).
import mjlab_microduck.tasks  # noqa: F401
from microduck_lab.rl.rl_policy import beam_terrain as T
from microduck_lab.rl.rl_policy import mdp_beam as M
from microduck_lab.rl.rl_policy import microduck_velocity_sideways_beam_env_cfg as B
from microduck_lab.rl.microduck_velocity_sideways_env_cfg import make_microduck_velocity_sideways_env_cfg
from microduck_lab.rl.rl_policy.warmstart_beam import BEAM_CRITIC_EXTRA, widen_critic


@pytest.fixture(scope="module")
def cfg():
    return B.make_microduck_velocity_sideways_beam_env_cfg()


# -- obs contract -----------------------------------------------------------------

def test_actor_obs_is_the_61d_contract(cfg):
    """Same actor terms, in the same order, as the sideways recipe (61D):
    48 proprioception + twist(3) + head(4) + body(6)."""
    base = make_microduck_velocity_sideways_env_cfg()
    assert list(cfg.observations["actor"].terms) == list(base.observations["actor"].terms)
    assert list(cfg.observations["actor"].terms)[-3:] == ["command", "head_command", "body_command"]
    assert len(cfg.commands["head_pose"].ranges) == 4 and len(cfg.commands["body_pose"].ranges) == 6


def test_critic_gets_beam_at_the_end(cfg):
    """warmstart_beam widens the critic's LAST columns, so the new term must be last."""
    assert list(cfg.observations["critic"].terms)[-1] == "beam"
    assert BEAM_CRITIC_EXTRA == 6 + len(M.PROFILE_AHEAD)


def test_head_and_body_commands_stay_tiny(cfg):
    """Tiny ranges; head tracking paid (0.5, gated) so the head inputs stay trained."""
    for lo, hi in cfg.commands["head_pose"].ranges:
        assert hi <= 0.07 and lo >= -0.07
    for lo, hi in cfg.commands["body_pose"].ranges:
        assert hi <= 0.05 and lo >= -0.05
    h = cfg.rewards["head_pose_tracking"]
    assert 0.5 <= h.weight <= 1.0 and h.func is M.gated
    assert h.params["inner_params"]["command_name"] == "head_pose"
    assert cfg.rewards["body_pose_tracking"].weight == 0.0


# -- rewards ------------------------------------------------------------------------

POSITIVE = ("progress", "track_lateral_velocity", "track_yaw_rate_tight", "upright", "pose",
            "air_time", "swing_lift", "standing_still", "head_pose_tracking")
COSTS = ("beam_offset", "heading", "failed", "action_rate_l2", "foot_swing_height", "foot_slip",
         "foot_slip_linear", "foot_clearance", "dof_pos_limits", "self_collisions")


def test_reward_signs(cfg):
    """Functions returning >= 0 costs get negative weights (every penalty logs <= 0)."""
    r = cfg.rewards
    for name in POSITIVE:
        assert r[name].weight > 0, name
    for name in COSTS:
        assert r[name].weight < 0, name
    for name in ("body_ang_vel", "angular_momentum"):
        assert r[name].weight <= 0
    # Self-negating microduck penalty: must never get a negative weight.
    assert r["head_pose_bias"].weight == 0.0
    assert "track_linear_velocity" not in r and "track_angular_velocity" not in r


def test_gated_terms_wrap_the_recipe_terms(cfg):
    for name in B.GATED:
        assert cfg.rewards[name].func is M.gated
        assert "inner" in cfg.rewards[name].params


def test_drift_costs_are_capped_below_a_fall(cfg):
    """Offset and heading costs saturate: 5 s of the worst of BOTH cost less than
    one fall, so after a drift staying up is never worse than falling. Near the
    axis they are still the plain squares (same gradient as before)."""
    env, robot, _ = _fake_env(n=3)
    M.tracker(env)
    o = env.scene.env_origins
    robot.data.root_link_pos_w[:, 1] = o[:, 1] + torch.tensor([0.002, 0.5, -3.0])
    yaw = torch.tensor([-math.pi / 2 + 0.01, math.pi / 2, 0.0])       # errors 0.01, pi, pi/2
    robot.data.root_link_quat_w[:] = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
    r = cfg.rewards
    off = M.beam_offset_sq(env, **r["beam_offset"].params)
    head = M.heading_err_sq(env, **r["heading"].params)
    cap_o, cap_h = r["beam_offset"].params["cap"], r["heading"].params["cap"]
    assert float(off.max()) <= cap_o**2 + 1e-12 and float(head.max()) <= cap_h**2 + 1e-12
    assert float(off[0]) == pytest.approx(0.002**2, rel=0.02)
    assert float(head[0]) == pytest.approx(0.01**2, rel=0.02)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    worst_per_s = abs(r["beam_offset"].weight) * cap_o**2 + abs(r["heading"].weight) * cap_h**2
    fall = abs(r["failed"].weight) * dt
    assert worst_per_s * 5.0 < fall, (worst_per_s * 5.0, fall)
    # monotone: more drift never costs less
    ys = torch.linspace(0, 0.2, 201)
    assert torch.all(torch.diff(M.saturating_sq(ys, cap_o)) >= 0)


def test_failure_cost_after_dt(cfg):
    """mjlab multiplies each term by dt: -500 x 0.02 = -10 per failure (bridge v11b: -0.1)."""
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert abs(dt - 0.02) < 1e-9
    assert cfg.rewards["failed"].weight * dt <= -5.0


def test_story_motor_model(cfg):
    """The story's motors (world.py): no action delay, 7.4 V, sag 0.1, kp 200, no obs
    delay. The other DR stays (s4_fix A/B: only the motor model mattered)."""
    from microduck_lab.tasks.human_bridge.rl_policy import world as Wd
    assert B.STORY_MOTORS
    for a in cfg.scene.entities["robot"].articulation.actuators:
        assert (a.delay_min_lag, a.delay_max_lag) == (0, 0)
        assert a.vin_range == (Wd.BAM_VIN, Wd.BAM_VIN)
        assert a.vin_drop_gain_range == (Wd.BAM_VIN_DROP_GAIN, Wd.BAM_VIN_DROP_GAIN)
        assert a.kp_fw == Wd.BAM_KP_FW and a.vin_min == Wd.BAM_VIN_MIN
    for name, term in cfg.observations["actor"].terms.items():
        assert getattr(term, "delay_max_lag", 0) == 0, name
    for name in ("randomize_com", "randomize_joint_friction", "encoder_bias", "foot_friction",
                 "expand_bam_friction_fields"):
        assert name in cfg.events, name
    assert cfg.observations["actor"].enable_corruption
    base = make_microduck_velocity_sideways_env_cfg()        # the recipe itself is untouched
    assert base.scene.entities["robot"].articulation.actuators[0].delay_max_lag > 0


def test_no_inherited_step_curriculum(cfg):
    for name in B.INHERITED_CURRICULA:
        assert name not in cfg.curriculum
    assert set(cfg.curriculum) == {"terrain_levels"}
    assert cfg.rewards["action_rate_l2"].weight == B.ACTION_RATE


# -- terminations ---------------------------------------------------------------

def test_terminations(cfg):
    t = cfg.terminations
    assert "fell_over" not in t
    assert "out_of_terrain_bounds" not in t     # fires on the outer tiles (0.3 m margin)
    assert abs(t["tilted"].params["limit_angle"] - math.acos(0.9)) < 1e-9
    for name in ("body_contact", "foot_off", "dropped", "nan_state", "time_out"):
        assert name in t
    assert t["reached_end"].time_out and t["time_out"].time_out
    assert not t["body_contact"].time_out and not t["foot_off"].time_out
    assert cfg.sim.nconmax >= 100          # pooled; measured peak 2.1 contacts per env
    assert cfg.episode_length_s >= 30.0


def test_contact_pattern_resolves_on_the_robot():
    """Her 35 non-foot story-solid parts get names the sensor matches; the two
    foot shells too (allowed contacts), and they pass the CollisionCfg."""
    import re
    spec = M.beam_walk_spec()
    names = [g.name for g in spec.geoms]
    hit = [n for n in names if re.match(r"^bodyhit_.*", n)]
    shell = [n for n in names if re.match(r"^footshell_.*", n)]
    assert len(hit) == 35 and len(shell) == 2
    assert not any("sole" in n or "foot_collision" in n for n in hit)
    robot = B.make_microduck_velocity_sideways_beam_env_cfg().scene.entities["robot"]
    for c in robot.collisions:
        c.edit_spec(spec)
    for g in spec.geoms:
        if g.name.startswith(("bodyhit_", "footshell_")):
            assert (g.contype, g.conaffinity) == (0, M.SOLID_CONAFFINITY), g.name
        if g.name in ("left_foot_collision", "right_foot_collision"):
            assert (g.contype, g.conaffinity) == (1, 1)
    m = spec.compile()
    for g in range(m.ngeom):          # body bits (set at compile) let them collide
        if m.geom_conaffinity[g] & 8:
            assert m.body_conaffinity[m.geom_bodyid[g]] & 8


def test_contact_bits_only_hit_the_terrain():
    ct, ca = M.TERRAIN_CONTYPE, 1
    part = (0, M.SOLID_CONAFFINITY)
    touch = lambda a, b: bool((a[0] & b[1]) or (b[0] & a[1]))
    assert touch(part, (ct, ca))              # parts vs terrain
    assert not touch(part, (1, 1))            # parts vs her soles
    assert not touch(part, (2, 2))            # parts vs self-collision geoms
    assert not touch(part, part)
    assert touch((1, 1), (ct, ca))            # soles still stand on the terrain


def test_sensors(cfg):
    s = {x.name: x for x in cfg.scene.sensors}
    c = s["body_terrain_contact"]
    assert c.primary.pattern == r"^bodyhit_.*" and c.secondary.pattern == "terrain"
    offs, _ = s["foot_height_scan"].pattern.generate_rays(None, "cpu")
    # Rays along her forward (across the beam), within the 54 mm sole.
    assert offs[:, 1].abs().max() == 0 and offs[:, 0].abs().max() <= 0.027 + 1e-6
    assert len(offs) >= 3
    # A foot with no beam under it must not read its height above z = 0 (~0.3 m).
    assert s["foot_height_scan"].max_distance <= 0.1


def test_swing_terms_keep_the_v4_values(cfg):
    """The foot site is at the sole bottom (measured), so v4's targets already mean real lift."""
    base = make_microduck_velocity_sideways_env_cfg().rewards
    assert cfg.rewards["foot_swing_height"].params["target_height"] == base["foot_swing_height"].params["target_height"]
    ip = cfg.rewards["swing_lift"].params["inner_params"]
    assert (ip["low"], ip["high"]) == (base["swing_lift"].params["low"], base["swing_lift"].params["high"])
    assert M.SITE_Z == 0.0


# -- command law --------------------------------------------------------------------

def _command(cfg, n=4):
    c = object.__new__(M.BeamSteeringCommand)
    c.cfg = cfg.commands["twist"]
    c.vx0 = torch.full((n,), c.cfg.vx0)
    c.wz0 = torch.full((n,), c.cfg.wz0)
    c.vx_noise = torch.zeros(n)
    c.wz_noise = torch.zeros(n)
    return c


def test_steering_law_signs(cfg):
    """Facing -y, her forward is world -y: trunk at y > 0 (behind the axis) steps
    forward (vx > 0); turned CCW (heading error > 0) turns back (wz < 0)."""
    c = _command(cfg)
    k = c.cfg
    assert (k.k_y, k.k_yaw, k.vx0, k.wz0, k.vx_max, k.wz_max) == (5.0, 2.0, -0.055, 0.06, 0.3, 0.4)
    vx, wz = c.steer(torch.tensor([0.0, 0.02, -0.02, 0.5]), torch.tensor([0.0, 0.1, -0.1, 2.0]))
    assert torch.allclose(vx[:3], torch.tensor([-0.055, 0.045, -0.155]))    # -20 mm no longer clips
    assert vx[3] == pytest.approx(0.3) and wz[3] == pytest.approx(-0.4)
    # vx (her only offset sensor) stays unclipped over +-40 mm
    y = torch.linspace(-0.04, 0.04, 81)
    c4 = _command(cfg, n=81)
    vx, _ = c4.steer(y, torch.zeros(81))
    assert torch.allclose(vx, k.vx0 + k.k_y * y)
    assert wz[1] < 0 < wz[2]
    # Her left (+vy in her frame) is world +x when she faces -y.
    left = (-math.sin(k.heading), math.cos(k.heading))
    assert left[0] > 0.999
    assert 0 < k.vy_range[0] < k.vy_range[1] <= 0.14
    assert k.rel_standing_envs == 0.0          # zero commands only while waiting on the start platform


def test_every_spawn_walks_at_once(cfg):
    """The story's walker never gets a zero command (its standing policy covers
    the wait), so no spawn waits: the steering law runs from the first step."""
    assert cfg.events["reset_base"].params["wait_s"] == (0.0, 0.0)
    n = 400
    env, robot, _ = _fake_env(kind="flat", row=9, n=n)
    robot.data.default_root_state = torch.zeros(n, 13)
    robot.data.default_root_state[:, 3] = 1.0
    robot.write_root_link_pose_to_sim = lambda pose, env_ids: robot.data.root_link_pos_w.__setitem__(
        env_ids, pose[:, :3])
    robot.write_root_link_velocity_to_sim = lambda vel, env_ids: None
    tw = cfg.commands["twist"]
    cmd = object.__new__(M.BeamSteeringCommand)
    cmd.cfg, cmd._env, cmd.robot = tw, env, robot
    for name in ("vy_base", "vx0", "wz0", "vx_noise", "wz_noise"):
        setattr(cmd, name, torch.zeros(n))
    cmd.vy_base[:] = 0.11
    cmd.vx0[:] = tw.vx0
    cmd.wz0[:] = tw.wz0
    cmd.is_standing_env = torch.zeros(n, dtype=torch.bool)
    cmd.vel_command_b = torch.zeros(n, 3)
    cmd.vel_command_w = torch.zeros(n, 3)
    env.command_manager = SimpleNamespace(get_command=lambda name: cmd.vel_command_b)
    p = dict(cfg.events["reset_base"].params)
    M.reset_on_beam(env, torch.arange(n), **p)
    tr = M.tracker(env)
    x, _, _ = tr.rel()
    on_start = x < T.X_BEAM0
    assert 0.1 < on_start.float().mean() < 0.4                       # p_start 0.25
    assert torch.all(tr.wait == 0.0)
    cmd._update_command()
    assert torch.all(cmd.vel_command_b[:, 1] > 0.1)


# -- terrain ----------------------------------------------------------------------

def test_beam_widths_follow_the_rows():
    for kind in T.KINDS:
        if kind == "his_back":
            continue
        widths = []
        for row in range(T.NUM_ROWS):
            segs = T.beam_profile(kind, row)
            beam = [s for s in segs if T.X_BEAM0 <= s[0] and s[1] <= T.X_END]
            widths.append(min(s[4] for s in beam))
        assert widths[0] >= 0.142 and widths[-1] <= 0.030
        assert all(a >= b for a, b in zip(widths, widths[1:])), (kind, widths)
        assert min(widths) >= T.W_MIN - 1e-9


def test_row0_is_flat_and_features_grow():
    for kind in T.KINDS:
        _, top, _, gap = T.sample_profile(T.beam_profile(kind, 0))
        assert np.ptp(top) == 0 and not gap.any()
    last = T.NUM_ROWS - 1
    jump = lambda kind, row: np.abs(np.diff(T.sample_profile(T.beam_profile(kind, row))[1])).max()
    assert 0.015 < jump("steps", last) <= T.STEP_MAX + 1e-9
    assert T.sample_profile(T.beam_profile("pits", last))[3].sum() * 0.005 > 0.05


def test_every_tile_reads_its_own_row():
    """Replays the generator's draws (curriculum mode: for col, for row: U)."""
    gen = T.make_beam_terrain_cfg()
    rng = np.random.default_rng(gen.seed)
    for _ in gen.sub_terrains:
        for row in range(gen.num_rows):
            d = (row + rng.uniform()) / gen.num_rows
            assert int(np.floor(d * gen.num_rows + 1e-6)) == row


def test_ramps_stay_below_20_deg():
    for row in range(T.NUM_ROWS):
        for x0, x1, h0, h1, _ in T.beam_profile("ramps", row):
            assert math.degrees(math.atan2(abs(h1 - h0), x1 - x0)) <= T.RAMP_MAX_DEG + 1e-6


def test_his_back_at_full_difficulty_is_his_profile():
    """F1w, measured (s4_fix/surface_f1w.json): 30 mm hole, his feet 16 mm down,
    shins as two rails (narrowest |y| 19-26 mm), trunk +4.7 mm, rounded head
    sloping to -22.6 mm, far pit 12.5 / 15 mm, far ledge -12.5 mm."""
    segs = T.beam_profile("his_back", T.NUM_ROWS - 1)
    first = [s for s in segs if s[0] > T.X_BEAM0][0]
    assert first[0] - T.X_BEAM0 == pytest.approx(0.030)
    assert first[2] == pytest.approx(-0.016)
    rail = [s for s in segs if len(s) > 5 and len(s[5]) == 2]
    assert rail and all(b[1] > 0 for s in rail for b in T.seg_boxes(s))
    inner = min(abs(b[0]) - b[1] / 2 for s in rail for b in T.seg_boxes(s))
    outer = min(abs(b[0]) + b[1] / 2 for s in rail for b in T.seg_boxes(s))
    assert inner == pytest.approx(0.011) and outer == pytest.approx(0.026)
    copy1 = segs[1:1 + sum(1 for it in T.HIS_BACK if it[0] != "gap")]
    assert max(max(s[2], s[3]) for s in copy1) == pytest.approx(0.0047)
    assert min(s[3] for s in copy1) == pytest.approx(-0.0226)
    ledges = [s for s in segs[1:-1] if s[4] == T.PLATFORM_W]
    assert ledges[0][2] == pytest.approx(-0.0125) and ledges[1][2] == pytest.approx(-0.025)
    heads = [s for s in segs if s[4] < 0.1 and s[3] - s[2] < -0.02]    # the head slopes
    pits = [round((b[0] - a[1]) * 1000, 1) for a, b in zip(heads, ledges)]
    assert pits[:2] == [12.5, 15.0]


def test_small_feature_columns():
    """narrow_steps / narrow_mixed / crowned: width follows the row down to 28 mm,
    but no step, ramp rise, bump or gap is over 12 mm."""
    cap = T.SMALL_MAX + 1e-9
    for kind in T.SMALL_KINDS:
        assert kind in T.KINDS
        w9 = min(s[4] for s in T.beam_profile(kind, T.NUM_ROWS - 1) if T.X_BEAM0 <= s[0] and s[1] <= T.X_END)
        assert w9 <= 0.030
        for row in range(T.NUM_ROWS):
            segs = T.beam_profile(kind, row)
            for a, b in zip(segs, segs[1:]):
                assert abs(b[3] - b[2]) <= cap and b[0] - a[1] <= cap, (kind, row, a, b)
                if b[0] - a[1] < 1e-9 or b[0] - a[1] <= cap:
                    assert abs(b[2] - a[3]) <= cap, (kind, row, a, b)


def test_crowned_top():
    """Flat top w, then on each side two steps down: 3-8 mm over 10-15 mm."""
    segs = T.beam_profile("crowned", T.NUM_ROWS - 1)
    beam = [s for s in segs if T.X_BEAM0 <= s[0] and s[1] <= T.X_END]
    for s in beam:
        boxes = T.seg_boxes(s)
        assert len(boxes) == 3 and boxes[0] == (0.0, s[4], 0.0)
        drop, side = -boxes[2][2], (boxes[2][1] - boxes[0][1]) / 2
        assert 0.003 - 1e-9 <= drop <= 0.008 + 1e-9 and 0.010 - 1e-9 <= side <= 0.015 + 1e-9
        assert boxes[1][2] == pytest.approx(-drop / 2)
    assert T.seg_boxes(T.beam_profile("crowned", 0)[1]) == ((0.0, T.beam_profile("crowned", 0)[1][4], 0.0),)


@pytest.mark.parametrize("kind,row", [("crowned", 9), ("his_back", 9), ("his_back", 4)])
def test_tile_geometry_matches_profile_cross_sections(kind, row):
    """Crowned tops and rails: one geom per box, at the box's y, width and top."""
    sub = T.BeamLaneTerrainCfg(kind=kind, size=(T.TILE_X, T.TILE_Y))
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    sub.function((row + 0.5) / T.NUM_ROWS, spec, np.random.default_rng(0))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    boxes = [(s, b) for s in sub.profile(row) for b in T.seg_boxes(s)]
    assert m.ngeom == len(boxes)
    for g, (s, (y, bw, dz)) in enumerate(boxes):
        assert m.geom_size[g][1] == pytest.approx(bw / 2)
        assert d.geom_xpos[g][1] == pytest.approx(T.TILE_Y / 2 + y)
        if s[2] == s[3]:
            assert d.geom_xpos[g][2] + m.geom_size[g][2] == pytest.approx(T.H0 + s[2] + dz, abs=1e-6)


def test_tile_geometry_matches_profile():
    """The boxes the generator builds have the profile's widths and tops."""
    sub = T.BeamLaneTerrainCfg(kind="steps", size=(T.TILE_X, T.TILE_Y))
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    row = 6
    out = sub.function((row + 0.5) / T.NUM_ROWS, spec, np.random.default_rng(0))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    segs = sub.profile(row)
    assert m.ngeom == len(segs)
    for g, (x0, x1, h0, h1, w) in enumerate(segs):
        assert m.geom_size[g][1] == pytest.approx(w / 2)
        assert d.geom_xpos[g][2] + m.geom_size[g][2] == pytest.approx(T.H0 + max(h0, h1), abs=1e-3)
    assert out.origin[1] == pytest.approx(T.TILE_Y / 2) and out.origin[2] == pytest.approx(T.H0)


def test_spawn_points_stand_on_flat_ground():
    for kind in T.KINDS:
        for row in (0, 5, 9):
            segs = T.beam_profile(kind, row)
            xs, top, _, gap = T.sample_profile(segs)
            for x in T.spawn_points(segs, 16, T.X_END - 1.0):
                m = (xs >= x - 0.065) & (xs <= x + 0.065)
                assert np.ptp(top[m]) < 1e-9 and not gap[m].any(), (kind, row, x)
                assert x <= T.X_END - 1.0 + 1e-9


# -- tracker logic on a fake env ------------------------------------------------------

class _Scene(dict):
    pass


def _fake_env(kind="steps", row=5, n=2):
    gen = T.make_beam_terrain_cfg()
    types = torch.full((n,), T.KINDS.index(kind), dtype=torch.long)
    terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=gen),
                              terrain_levels=torch.full((n,), row, dtype=torch.long),
                              terrain_types=types)
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros(n, 3), root_link_quat_w=torch.tensor([[math.cos(-math.pi / 4), 0, 0,
                                                                          math.sin(-math.pi / 4)]] * n),
        site_pos_w=torch.zeros(n, 2, 3)))
    scene = _Scene(robot=robot)
    scene.terrain = terrain
    scene.env_origins = torch.tensor([[1.0, 2.0, T.H0]] * n)
    cmd = torch.tensor([[0.0, 0.1, 0.0]] * n)
    env = SimpleNamespace(scene=scene, device="cpu", num_envs=n, step_dt=0.02, common_step_counter=0,
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    return env, robot, cmd


def _step(env, robot, x):
    env.common_step_counter += 1
    robot.data.root_link_pos_w[:, 0] = env.scene.env_origins[:, 0] + torch.as_tensor(x)


def test_ratchet_pays_new_ground_only():
    env, robot, _ = _fake_env()
    tr = M.tracker(env)
    tr.start(torch.arange(2), torch.tensor([0.5, 0.5]))
    paid = 0.0
    # 2 mm per 20 ms step = 0.1 m/s, below the cap (1.25 x vy_cmd): forward, rock back, forward again
    for x in (0.502, 0.504, 0.500, 0.498, 0.504, 0.506):
        _step(env, robot, [x, 0.5])
        paid += float(M.progress_ratchet(env)[0]) * env.step_dt
    assert paid == pytest.approx(0.006, abs=1e-6)
    assert float(M.progress_ratchet(env)[1]) == 0.0


def test_ratchet_does_not_pay_rushing():
    """Ground gained faster than speed_cap x vy_cmd (a lunge, a fall toward +x)
    is paid only up to the cap."""
    env, robot, cmd = _fake_env()
    tr = M.tracker(env)
    tr.start(torch.arange(2), torch.tensor([0.5, 0.5]))
    _step(env, robot, [0.53, 0.5])                        # 30 mm in one step = 1.5 m/s
    vy = float(env.command_manager.get_command("twist")[0, 1])
    assert float(M.progress_ratchet(env, speed_cap=1.25)[0]) == pytest.approx(1.25 * vy, rel=1e-5)


def test_gate_pays_slow_careful_steps():
    """gate_frac 0.3: new ground at 40 % of the commanded speed keeps the gate open."""
    assert M.TRACKER_PARAMS["gate_frac"] == pytest.approx(0.3)
    env, robot, cmd = _fake_env()
    tr = M.tracker(env)
    tr.start(torch.arange(2), torch.tensor([0.5, 0.5]))
    for i in range(150):                               # 3 s at 0.04 m/s (env 0) and 0.02 m/s (env 1)
        _step(env, robot, [0.5 + 0.0008 * (i + 1), 0.5 + 0.0004 * (i + 1)])
        g = tr.gate()
    assert g[0] > 0.99 and 0.5 < g[1] < 0.8


def test_gate_closes_when_marching_in_place():
    env, robot, cmd = _fake_env()
    tr = M.tracker(env)
    tr.start(torch.arange(2), torch.tensor([0.5, 0.5]))
    for i in range(100):                               # 2 s: env 0 walks 0.1 m/s, env 1 rocks
        _step(env, robot, [0.5 + 0.002 * (i + 1), 0.5 + 0.004 * (i % 2)])
        g = tr.gate()
    assert g[0] > 0.99 and g[1] < 0.1
    cmd[1] = 0.0                                       # zero command: always open
    assert tr.gate()[1] == 1.0


def test_foot_off_and_dropped():
    env, robot, _ = _fake_env(kind="pits", row=9)
    tr = M.tracker(env)
    segs = T.beam_profile("pits", 9)
    gap_x = next(a[1] for a, b in zip(segs, segs[1:]) if b[0] - a[1] > 0.01)   # a gap start
    flat_x = T.X_BEAM0 + 0.05
    o = env.scene.env_origins
    site = robot.data.site_pos_w
    site[:, :, 0] = o[:, None, 0] + flat_x
    site[:, :, 2] = o[:, None, 2] + M.SITE_Z                     # both soles flat on the platform
    assert not M.foot_off(env, asset_cfg=SimpleNamespace(name="robot", site_ids=[0, 1])).any()
    site[1, 0, 0] = o[1, 0] + gap_x + 0.01                        # env 1: left foot down the gap
    h = float(tr.lookup(tr.top, torch.tensor([gap_x + 0.01, gap_x + 0.01]))[1])
    site[1, 0, 2] = o[1, 2] + h + M.SITE_Z - 0.03
    off = M.foot_off(env, asset_cfg=SimpleNamespace(name="robot", site_ids=[0, 1]))
    assert off.tolist() == [False, True]
    robot.data.root_link_pos_w[:] = o + torch.tensor([flat_x, 0.0, 0.122])
    robot.data.root_link_pos_w[1, 2] = o[1, 2] + 0.03
    assert M.dropped(env).tolist() == [False, True]


def test_beam_move_masks():
    p = torch.tensor([1.0, 0.5, 0.1, 0.1, 1.0])
    judged = torch.tensor([True, True, True, False, False])
    up, down = M.beam_move_masks(p, judged, 0.8, 0.2)
    assert up.tolist() == [True, False, False, False, False]
    assert down.tolist() == [False, False, True, False, False]
    # 1 m of new ground, but the episode ended by a failure: no promotion
    up, _ = M.beam_move_masks(torch.tensor([1.0, 1.0]), torch.tensor([True, True]), 0.8, 0.2,
                              failed=torch.tensor([True, False]))
    assert up.tolist() == [False, True]
    # a short random first episode that timed out does not demote
    _, down = M.beam_move_masks(p, judged, 0.8, 0.2, torch.tensor([True, True, False, True, True]))
    assert not down.any()


# -- warm start / runner --------------------------------------------------------------

def test_widen_critic_sets_new_stats_and_count():
    """The 18 new inputs get measured mean / var; the shared count drops from v4's 98M."""
    sd = {"mlp.0.weight": torch.randn(8, 5), "obs_normalizer._mean": torch.randn(1, 5),
          "obs_normalizer._var": torch.rand(1, 5) + 0.5, "obs_normalizer._std": torch.rand(1, 5) + 0.5,
          "obs_normalizer.count": torch.tensor(98402304)}
    old_mean = sd["obs_normalizer._mean"].clone()
    new = widen_critic(sd, 3, mean=[0.1, 0.2, 0.3], var=[4.0, 1.0, 0.25], count=2_000_000)
    assert torch.allclose(new["obs_normalizer._mean"][0, :5], old_mean[0])
    assert torch.allclose(new["obs_normalizer._mean"][0, 5:], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(new["obs_normalizer._std"][0, 5:], torch.tensor([2.0, 1.0, 0.5]))
    assert int(new["obs_normalizer.count"]) == 2_000_000
    from microduck_lab.rl.rl_policy.warmstart_beam import CRITIC_COUNT
    assert CRITIC_COUNT <= 5_000_000        # new inputs must still move: rate 4096 / count per step


def test_widen_critic_keeps_old_outputs():
    torch.manual_seed(0)
    sd = {"mlp.0.weight": torch.randn(8, 5), "obs_normalizer._mean": torch.randn(1, 5),
          "obs_normalizer._var": torch.rand(1, 5) + 0.5, "obs_normalizer._std": torch.rand(1, 5) + 0.5}
    old = {k: v.clone() for k, v in sd.items()}
    new = widen_critic(sd, 3)
    x = torch.randn(4, 5)
    x3 = torch.cat([x, torch.randn(4, 3)], dim=1)
    f = lambda s, v: ((v - s["obs_normalizer._mean"]) / s["obs_normalizer._std"]) @ s["mlp.0.weight"].T
    assert torch.allclose(f(old, x), f(new, x3), atol=1e-6)


def test_runner_cfg():
    rl = B.MicroduckSidewaysBeamRlCfg
    assert rl.experiment_name == "velocity_sideways_beam"
    assert rl.algorithm.gamma == pytest.approx(0.995)
    assert rl.algorithm.symmetry_cfg is None
    from microduck_lab.rl.microduck_velocity_sideways_env_cfg import MicroduckSidewaysRlCfg
    assert MicroduckSidewaysRlCfg.experiment_name == "velocity_sideways"   # base untouched


def test_progress_pay_is_capped_at_the_commanded_speed(cfg):
    """Uncapped, 5 PPO iterations made her 54% faster with more falls
    (s5_review): ground gained faster than 1.25 x vy_cmd is not paid."""
    t = cfg.rewards["progress"]
    assert t.func is M.progress_ratchet
    assert t.params["speed_cap"] == B.PROGRESS_SPEED_CAP == 1.25


def test_scan_miss_reads_above_the_swing_targets_but_stays_small():
    """A ray over a gap reads SCAN_MAX_DISTANCE. It must stay above the swing
    target and lift range (normal steps read true) but small, so a landing
    beside a gap is not charged 400x a normal one (it was 0.10 m)."""
    from microduck_lab.rl import microduck_velocity_sideways_env_cfg as side
    assert side.SWING_HEIGHT_TARGET < B.SCAN_MAX_DISTANCE <= 0.05
    assert side.LIFT_RANGE[1] < B.SCAN_MAX_DISTANCE
