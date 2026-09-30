import numpy as np
import pytest
from microduck_lab.rl.scripted_policy.salmon_jump.run import Candidate, pose, target_at, summarize, reserve, INDEX


def trace():
    a=np.zeros((400,14));a[:,0]=np.arange(1,401)*.02
    a[:,3]=.06;a[:,5]=1.;a[:,6]=.07;a[:,9]=7.4
    return a


def test_rest_and_grounded_recovery_do_not_count_as_jump():
    a=trace()
    assert not summarize(a)['success']
    a[200:,3]=.115;a[200:,4]=1.;a[200:,6]=.14
    a[200:,8]=7.4;a[200:,9]=0.
    assert not summarize(a)['success']


def test_airborne_phase_needs_upward_takeoff_then_feet_only_landing():
    a=trace();a[200:205,9]=0.;a[200:205,10]=.01;a[200:205,6]=.11
    a[205:,3]=.115;a[205:,4]=1.;a[205:,6]=.14
    a[205:,8]=7.4;a[205:,9]=0.
    assert not summarize(a)['success']  # unsupported falling is not a launch
    a[200:205,13]=.3
    assert summarize(a)['success']
    a[300:,9]=1.
    assert not summarize(a)['success']


def test_single_sample_contact_loss_is_not_jump():
    a=trace();a[200,9]=0.;a[200,10]=.01;a[200,13]=.3
    assert summarize(a)['longest_airborne_s']==pytest.approx(.02)
    assert not summarize(a)['success']


def test_named_symmetric_targets_and_continuous_start():
    home=np.zeros(14);c=Candidate()
    np.testing.assert_array_equal(target_at(c,home,0.),home)
    q=pose(home,[.3,.5,-.7,.2,.1])
    for name in ['hip_pitch','knee','ankle']:
        assert q[INDEX['left_'+name]] == -q[INDEX['right_'+name]]
    assert q[INDEX['neck_pitch']]==.2
    np.testing.assert_array_equal(home,np.zeros(14))


def test_invalid_durations_and_unique_video_names(tmp_path):
    with pytest.raises(ValueError): Candidate(durations=(.2,0.,.1))
    assert reserve(tmp_path).name=='salmon_try1'
    assert reserve(tmp_path).name=='salmon_try2'


def test_multistage_targets_and_interpolation_modes():
    home=np.zeros(14)
    poses=((.2,.3,.4,.5,.6),)*5
    for mode in ('smoothstep','linear','step'):
        c=Candidate(poses=poses,durations=(.1,)*5,interpolation=mode)
        np.testing.assert_allclose(target_at(c,home,.5),pose(home,poses[-1]))
        np.testing.assert_allclose(target_at(c,home,1.3),home,atol=1e-12)
    c=Candidate(interpolation='step')
    np.testing.assert_allclose(target_at(c,home,0.),pose(home,c.poses[0]))
    with pytest.raises(ValueError): Candidate(poses=poses)
    with pytest.raises(ValueError): Candidate(interpolation='teleport')


def test_jump_after_standing_recovery_is_not_salmon():
    a=trace()
    a[180:,3]=.115;a[180:,4]=1.;a[180:,6]=.14
    a[180:,8]=7.4;a[180:,9]=0.;a[180:,12]=1.
    a[200:205,8]=0.;a[200:205,10]=.01;a[200:205,13]=.3
    assert not summarize(a)['success']
    # Tilting after the standing-policy handoff does not bypass the guard.
    a[200:205,4]=.6
    assert not summarize(a)['success']


def test_takeoff_speed_measured_at_contact_loss_not_clearance_threshold():
    a=trace()
    a[200:205,9]=0.;a[200:205,10]=.01;a[200:205,6]=.11
    a[200,10]=.0007;a[200,13]=.21
    a[201:205,13]=[-.01,-.2,-.4,-.6]
    a[200:205,4]=.86  # a near-upright takeoff is still a launch
    a[205:,3]=.115;a[205:,4]=1.;a[205:,6]=.14
    a[205:,8]=7.4;a[205:,9]=0.
    r=summarize(a)
    assert r['success']
    assert r['takeoff_com_vz']==.21
    assert r['qualifying_airborne_s']==pytest.approx(.1)
    # Grounded standing followed by a jump must not count as a back-to-feet jump.
    a[180:200,3]=.115;a[180:200,4]=1.;a[180:200,8]=7.4;a[180:200,9]=0.
    assert not summarize(a)['success']


def test_variable_length_search_keeps_controller_settings():
    from microduck_lab.rl.scripted_policy.salmon_jump.search import unpack
    x=np.r_[np.zeros(25),np.full(5,.15),.3]
    c=unpack(x,{'interpolation':'step','recovery':'airborne'})
    assert len(c.poses)==len(c.durations)==5
    assert c.interpolation=='step' and c.recovery=='airborne'


def test_body_stumble_followed_by_recovery_is_not_clean_landing():
    a=trace();a[200:205,9]=0.;a[200:205,10]=.01;a[200:205,6]=.11
    a[200:205,13]=.3
    a[205:,3]=.115;a[205:,4]=1.;a[205:,6]=.14
    a[205:,8]=7.4;a[205:,9]=0.
    assert summarize(a)['success']
    a[220,9]=2.
    r=summarize(a)
    assert r['final_standing_s']>2
    assert not r['clean_landing'] and not r['success']


def test_catch_controller_may_start_before_flight_without_a_standing_pause():
    a=trace();a[190:,12]=1.
    a[200:205,9]=0.;a[200:205,10]=.01;a[200:205,6]=.11;a[200:205,13]=.3
    a[205:,3]=.115;a[205:,4]=1.;a[205:,6]=.14;a[205:,8]=7.4;a[205:,9]=0.
    assert summarize(a)['success']


def test_separated_clearance_samples_do_not_qualify_a_launch():
    a=trace();a[200:205,9]=0.;a[200:205,10]=.001;a[200:205,6]=.11
    a[200,13]=.3;a[200,10]=.01;a[204,10]=.01
    # A later falling flight must not supply the launch's missing duration.
    a[210:215,9]=0.;a[210:215,10]=.01;a[210:215,13]=-.3
    a[215:,3]=.115;a[215:,4]=1.;a[215:,6]=.14;a[215:,8]=7.4;a[215:,9]=0.
    assert summarize(a)['takeoff_com_vz'] is None
