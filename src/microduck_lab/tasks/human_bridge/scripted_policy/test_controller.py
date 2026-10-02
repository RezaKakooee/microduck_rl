"""CPU checks without physics rollouts or rendering.

Physical trials belong in run.py so each one gets a GPU video.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_lab.tasks.human_bridge.scripted_policy.world import World
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.scripted_policy.controller import Settings, Supervisor, HEADING
from microduck_lab.tasks.human_bridge.scripted_policy.run import reserve
from microduck_lab.tasks.human_bridge.scripted_policy.surface import Surface
from microduck_lab.tasks.human_bridge.scripted_policy.metrics import audit
from microduck_lab.tasks.human_bridge.scripted_policy.preview import Preview


def test_video_numbers_are_never_reused(tmp_path):
    (tmp_path / "scripted_try1.mp4").touch()
    (tmp_path / "scripted_try2.lock").touch()
    assert reserve(tmp_path).name == "scripted_try3"
    assert reserve(tmp_path).name == "scripted_try4"


@pytest.mark.parametrize("kwargs", [dict(mode="typo"), dict(speed=float("nan")),
                                   dict(preview_horizon=0), dict(step_time=0)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


@pytest.fixture(scope="module")
def world():
    return World(L.design())


@pytest.mark.parametrize("kwargs", [{}, dict(lift=1), dict(terrain=1, body_center=.5),
                                   dict(balance=1), dict(mode="gait")])
def test_controller_changes_targets_only(world, kwargs):
    duck = world.ducks["she"]
    duck.place((-.15, .01, .46), yaw=HEADING, q=np.zeros(14))
    world.t = 0
    duck.hold(np.zeros(14))
    pol = SimpleNamespace(set_vel_cmd=lambda *args: None,
                          get_base_ang_vel=lambda: np.zeros(3))
    brain = SimpleNamespace(duck=duck, pol=pol, quiet=nullcontext,
                            act=lambda: duck.hold(np.zeros(14)))
    control = Supervisor(brain, Settings(**kwargs))
    q, v = world.data.qpos.copy(), world.data.qvel.copy()
    mass = world.model.body_mass.copy()
    control.act(True)
    np.testing.assert_array_equal(world.data.qpos, q)
    np.testing.assert_array_equal(world.data.qvel, v)
    np.testing.assert_array_equal(world.model.body_mass, mass)
    assert not world.data.xfrc_applied.any()
    assert not world.data.qfrc_applied.any()
    assert np.isfinite(duck.target).all()
    if kwargs.get("mode") != "gait":
        # At yaw -pi/2, local left moves toward +world x.
        assert control.command[1] == pytest.approx(.12)
        assert control.command[0] > 0  # Return from positive world y.


def test_surface_keeps_the_gap(world):
    surface = Surface(world.ducks["he"])
    assert surface.height(.5, 0) == pytest.approx(L.far_z)
    assert surface.height(-.3, 0) == pytest.approx(L.near_z)
    assert surface.height(.05, .2) == 0


def clear_trace():
    trace = np.zeros((201, 19))
    trace[:, 0] = np.arange(201)*.02 + 10
    trace[:, 3] = .4
    trace[:, 4] = .9
    trace[:, 6] = .1
    trace[:, 7] = .3
    return trace


def test_task_check_does_not_require_extra_landing_margin():
    trace = clear_trace()
    trace[:, 7] = .256
    trace[:, 4] = .65
    result = audit(trace)
    assert result['task_limits']
    assert result['final_clearance_seconds'] == pytest.approx(4.)


@pytest.mark.parametrize("column,value", [(3, .364), (4, .5), (6, -.35), (7, .255)])
def test_task_thresholds(column, value):
    trace = clear_trace()
    trace[-1, column] = value
    assert not audit(trace)['task_limits']


def test_earlier_fall_cannot_be_hidden_by_a_good_end_pose():
    trace = clear_trace()
    trace[3, 4] = .49
    assert not audit(trace)['task_limits']


def test_old_trace_uses_its_recorded_scene_thresholds():
    trace = clear_trace()
    trace[:, 3] = .368
    assert audit(trace, drop_z=.365, far_edge=.25)['task_limits']
    assert not audit(trace, drop_z=.3725, far_edge=.25)['task_limits']


def test_clearance_timer_restarts_after_moving_back():
    trace = clear_trace()
    trace[100, 7] = .25
    result = audit(trace)
    assert result['task_limits']
    assert result['final_clearance_seconds'] == pytest.approx(1.98)


def test_sequence_interpolation_and_end_hold():
    sequence = np.array([[0., 1.], [2., 3.], [4., 5.]])
    np.testing.assert_allclose(Preview.parameters_at(sequence, .2, .8), [1., 2.])
    np.testing.assert_allclose(Preview.parameters_at(sequence, 2., .8), [4., 5.])
    np.testing.assert_allclose(Preview.parameters_at(sequence, -.1, .8), [0., 1.])
    np.testing.assert_array_equal(sequence, [[0., 1.], [2., 3.], [4., 5.]])


def test_residual_lift_only_changes_motor_targets(world):
    duck = world.ducks['she']
    duck.place((-.15, 0., .46), yaw=HEADING, q=np.zeros(14))
    duck.hold(np.zeros(14))
    state = world.data.qpos.copy()
    velocity = world.data.qvel.copy()
    Preview.apply_residual(duck, np.array([0., .15, 0., .1, 0., 0., .4]),
                           Settings(preview_balance=2.))
    np.testing.assert_array_equal(world.data.qpos, state)
    np.testing.assert_array_equal(world.data.qvel, velocity)
    assert not np.array_equal(duck.target, np.zeros(14))
    assert not world.data.xfrc_applied.any()


def test_bridge_imports_stay_inside_our_package():
    import ast
    from pathlib import Path
    for path in Path(__file__).parent.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.startswith('microduck_lab.tasks.human_bridge.'):
                    assert name == 'microduck_lab.tasks.human_bridge.scripted_policy' or name.startswith('microduck_lab.tasks.human_bridge.scripted_policy.'), (path, name)
                if name.startswith('microduck_lab.rl.'):
                    assert name.startswith('microduck_lab.rl.human_bridge.scripted_policy.'), (path, name)


def test_missing_policy_fails_before_a_trial():
    from pathlib import Path
    from microduck_lab.tasks.human_bridge.scripted_policy.runtime import require_policy
    path = Path(__file__).resolve().parents[5] / 'local_storage/hb_dev/scripted_policy/policies/nonexistent_test.onnx'
    with pytest.raises(FileNotFoundError, match='Missing Codex policy'):
        require_policy(path)


def test_policy_outside_codex_is_rejected():
    from microduck_lab.tasks.human_bridge.scripted_policy.runtime import require_policy
    with pytest.raises(ValueError, match='Policy must be supplied inside'):
        require_policy('/not-a-codex-policy.onnx')


def test_body_support_restarts_the_landing_timer(monkeypatch):
    from microduck_lab.tasks.human_bridge.scripted_policy import run as runner
    trial = runner.Trial.__new__(runner.Trial)
    trial.world = SimpleNamespace(t=10.)
    trial.she = SimpleNamespace(up=lambda: .99)
    trial.control = SimpleNamespace(cfg=Settings(landing_margin=0., landing_up=.9,
        landing_support_limit=1., settle_with_preview=True))
    trial.events = []
    trial.phase, trial.t_phase = 'cross', 5.
    force = [0.]
    monkeypatch.setattr(runner, 'rearmost', lambda duck: .4)
    monkeypatch.setattr(runner, 'nonfoot_support', lambda duck: force[0])
    trial.check_landing()
    assert trial.phase == 'across' and trial.t_phase == 10.
    trial.world.t = 11.
    force[0] = 2.
    trial.check_landing()
    assert trial.phase == 'cross'
    trial.world.t = 12.
    force[0] = 0.
    trial.check_landing()
    assert trial.phase == 'across' and trial.t_phase == 12.
    trial.world.t = 13.
    trial.she.up = lambda: .8
    trial.check_landing()
    assert trial.phase == 'cross'


def test_standing_handoff_waits_for_clearance_height_and_speed(monkeypatch):
    from microduck_lab.tasks.human_bridge.scripted_policy import run as runner
    trial = runner.Trial.__new__(runner.Trial)
    trial.world = SimpleNamespace(t=10.)
    position = np.array([.5, 0., .41])
    velocity = np.array([.5, 0., 0.])
    trial.she = SimpleNamespace(up=lambda: .99, pos=lambda: position,
                               dof=0, data=SimpleNamespace(qvel=velocity))
    trial.control = SimpleNamespace(cfg=Settings(landing_margin=.08, landing_up=.95,
        landing_support_limit=1., landing_entry_speed=.3, landing_entry_height=.42))
    trial.events = []
    trial.phase, trial.t_phase = 'cross', 5.
    monkeypatch.setattr(runner, 'rearmost', lambda duck: .4)
    monkeypatch.setattr(runner, 'nonfoot_support', lambda duck: 0.)
    trial.check_landing()
    assert trial.phase == 'cross'
    velocity[:] = 0.
    trial.check_landing()
    assert trial.phase == 'cross'
    position[2] = .44
    trial.check_landing()
    assert trial.phase == 'across'


@pytest.mark.parametrize('start_x', [-.1, .34, .36, .6])
def test_terminal_goal_prefers_stopping_over_platform_overshoot(start_x):
    preview = Preview.__new__(Preview)
    preview.cfg = Settings(preview_finish=True, preview_progress=40., preview_finish_speed_cost=4.)
    stop = preview.terminal_value(start_x, .55, .99, np.zeros(3))
    overshoot = preview.terminal_value(start_x, 1.15, .99, np.array([1., 0., 0.]))
    fast_arrival = preview.terminal_value(start_x, .55, .99, np.array([1., 0., 0.]))
    assert stop > overshoot
    assert stop > fast_arrival


def test_standing_recovery_resets_timer_without_restarting_gait(monkeypatch):
    from microduck_lab.tasks.human_bridge.scripted_policy import run as runner
    trial = runner.Trial.__new__(runner.Trial)
    trial.world = SimpleNamespace(t=10.)
    up = [.99]
    trial.she = SimpleNamespace(up=lambda: up[0])
    trial.control = SimpleNamespace(cfg=Settings(landing_margin=0., landing_up=.95,
        landing_support_limit=1., settle_with_preview=False))
    trial.events = []
    trial.phase, trial.t_phase = 'cross', 5.
    monkeypatch.setattr(runner, 'rearmost', lambda duck: .4)
    monkeypatch.setattr(runner, 'nonfoot_support', lambda duck: 0.)
    trial.check_landing()
    assert trial.phase == 'across'
    trial.world.t = 11.
    up[0] = .9
    assert trial.check_landing() is None
    assert trial.phase == 'across' and trial.t_phase == pytest.approx(11.02)
    trial.world.t = 11.02
    up[0] = .99
    trial.check_landing()
    assert trial.phase == 'across' and trial.t_phase == pytest.approx(11.02)


@pytest.mark.parametrize('foot', [0, 1])
def test_independent_lift_changes_only_selected_leg_targets(world, foot):
    duck = world.ducks['she']
    duck.hold(np.zeros(14))
    qpos, qvel = world.data.qpos.copy(), world.data.qvel.copy()
    command = np.zeros(8)
    command[6+foot] = .4
    Preview.apply_residual(duck, command)
    selected, other = (slice(2, 5), slice(11, 14)) if foot == 0 else (slice(11, 14), slice(2, 5))
    assert np.any(duck.target[selected] != 0.)
    np.testing.assert_array_equal(duck.target[other], np.zeros(3))
    assert sum(duck.target[selected]) == pytest.approx(0.)
    np.testing.assert_array_equal(world.data.qpos, qpos)
    np.testing.assert_array_equal(world.data.qvel, qvel)


def test_independent_lift_requires_sequence_planning():
    with pytest.raises(ValueError, match='Independent lifts require'):
        Settings(preview_independent_lift=True)


def test_validation_detects_changed_landing_checks(tmp_path):
    import json
    from microduck_lab.tasks.human_bridge.scripted_policy.validation import summarize
    source_names = ('run.py', 'controller.py', 'preview.py', 'sequence.py',
                    'predict_workers.py', 'metrics.py', 'runtime.py', 'gait.py', 'surface.py')
    result = dict(settings={}, success=True, failure=None, dx=0., dy=0.,
                  min_up=1., min_z=.44, max_he_up=.01, end={},
                  source_sha256=dict.fromkeys(source_names, 'same'),
                  policy_sha256='policy', input_sha256={'world.py': 'world'})
    paths = [tmp_path / 'first.json', tmp_path / 'second.json']
    for path in paths:
        path.write_text(json.dumps(result))
        np.savez(path.with_suffix('.npz'), trace=clear_trace())
    assert summarize(paths)['same_controller']
    result['source_sha256']['run.py'] = 'changed-landing-checks'
    paths[1].write_text(json.dumps(result))
    assert not summarize(paths)['same_controller']
