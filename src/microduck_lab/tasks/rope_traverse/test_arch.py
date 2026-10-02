"""CPU checks for the leg-hook hang and the inchworm (no video)."""
import numpy as np
import pytest

from microduck_lab.tasks.rope_traverse.fit import I
from microduck_lab.tasks.rope_traverse.inchworm import Gait, Inchworm
from microduck_lab.tasks.rope_traverse.traverse import Trial, judge


def test_hang_is_carried_by_the_two_leg_hooks_only():
    trial = Trial(cycles=1)
    trial.start()
    k = trial.duck
    for _ in range(100):                      # 2 s holding the start pose
        k.hold(trial.q0); trial.w.step(); trial.record('hold')
    last = trial.rows[-1]
    assert last['floor'] == 0. and last['other'] == 0. and last['head'] == 0.
    assert last['left'] > .3 * trial.weight and last['right'] > .3 * trial.weight
    assert k.com()[2] < trial.w.rope_config.height - trial.w.rope_config.sag
    assert judge(trial.rows, trial.weight, trial.t_gait)['hold_pass']


def test_one_cycle_moves_along_the_rope_on_the_legs():
    trial = Trial(cycles=1)
    result = trial.run()
    assert result['travel_m'] > .005
    assert not result['floor_contact'] and not result['other_rope_contact']
    assert result['broken_hold_ticks'] == 0 and result['both_hooks_at_end']


def test_directions_are_mirror_images():
    left, right = Inchworm(Gait(), 1, 1), Inchworm(Gait(), 1, -1)
    swap = {}
    for name, i in I.items():
        if name.startswith('left_'): swap[i] = I['right_' + name[5:]]
        elif name.startswith('right_'): swap[i] = I['left_' + name[6:]]
        else: swap[i] = i
    for t in np.linspace(0, left.duration, 40):
        a, b = left(t), right(t)
        mirrored = np.array([a[swap[i]] for i in range(14)])
        for name, i in I.items():
            if name.endswith('hip_roll'):
                assert b[i] == pytest.approx(-mirrored[i])     # roll axes do not mirror
            elif name.startswith(('left_', 'right_')):
                assert b[i] == pytest.approx(-mirrored[i])     # mirrored joints change sign
            else:
                assert b[i] == pytest.approx(a[i])


def test_every_step_keeps_both_legs_as_hooks():
    """No step opens a hook wider than the sliding shape: the rope stays
    between thigh and foot of both legs all the time."""
    g = Gait()
    worm = Inchworm(g, 2, 1)
    for _, _, q in worm.steps:
        for side, sign in (('left', 1), ('right', -1)):
            assert sign * q[I[side + '_knee']] >= min(g.free_knee, g.carry_knee) - 1e-9


def test_judge_rejects_floor_head_and_short_travel():
    w = 7.4
    row = lambda t, arc, **kw: dict(dict(t=t, arc=arc, left=4., right=4., head=0., other=0., floor=0., torque=[0.]), **kw)
    rows = [row(.02 * i, .0002 * i) for i in range(1000)]   # 20 s, 0.2 m
    assert judge(rows, w, 0.)['traverse_pass']
    assert not judge(rows[:300], w, 0.)['traverse_pass']       # 6 cm only
    for key in ('floor', 'head', 'other'):
        bad = [dict(r) for r in rows]; bad[500][key] = 3.
        assert not judge(bad, w, 0.)['traverse_pass']
    one_leg = [dict(r) for r in rows]; one_leg[-1]['right'] = 0.
    assert not judge(one_leg, w, 0.)['traverse_pass']
