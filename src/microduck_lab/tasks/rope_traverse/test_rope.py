import mujoco
import numpy as np
from microduck_lab.tasks.rope_traverse.scene import build,Scene
from microduck_lab.tasks.rope_traverse.run import summarize


def test_anchors_only_connect_rope_to_world_and_jaw_is_bounded():
    w=build();m=w.model;d=w.data
    assert m.neq==1 and m.eq_type[0]==mujoco.mjtEq.mjEQ_CONNECT
    for name in ['rope_tip','rope_anchor']:
        site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name)
        body=int(m.site_bodyid[site]);assert body==0 or mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body).startswith('rope_')
    tip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'rope_tip');anchor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'rope_anchor')
    np.testing.assert_allclose(d.site_xpos[tip],d.site_xpos[anchor],atol=1e-12)
    jaw=w.ducks['she'].mouth
    assert m.actuator_forcelimited[jaw]
    np.testing.assert_allclose(m.actuator_forcerange[jaw],[-.25,.25])
    assert len(w.rope_geoms)==32
    assert not np.any(m.body_gravcomp)


def test_floor_support_and_incidental_body_contact_do_not_pass():
    a=np.zeros((200,12));a[:,0]=np.arange(200)*.02;a[:,3]=.35;a[:,4]=.2
    a[:,6]=3.;a[:,7]=4.;a[:,8]=4.
    assert summarize(a,.5)['hang_pass']
    assert not summarize(a,.5)['traverse_pass']
    a[:,10]=8.
    assert not summarize(a,.5)['hang_pass']
    a[:,10]=0.;a[:,9]=8.
    assert not summarize(a,.5)['hang_pass']


def test_one_foot_does_not_count_as_a_three_point_hold():
    a=np.zeros((200,12));a[:,0]=np.arange(200)*.02;a[:,3]=.35;a[:,4]=.2
    a[:,6]=3.;a[:,7]=4.
    result=summarize(a,.5)
    assert not result['hang_pass']
    assert result['sustained_support_s']>2.


def test_balance_is_setup_only_and_puts_com_under_rope():
    from microduck_lab.tasks.rope_traverse.fit import place,balance_about_rope
    import pytest
    w=build();place(w,np.array([-2.,-1.,1.,0.,0.,0.,0.,0.,0.]))
    balance_about_rope(w)
    assert abs(w.ducks['she'].com()[1])<1e-8
    assert w.ducks['she'].com()[2]<w.rope_config.height-w.rope_config.sag
    w.start()
    with pytest.raises(RuntimeError):balance_about_rope(w)


def test_reach_scheduler_releases_only_one_grip_at_a_time():
    from microduck_lab.tasks.rope_traverse.run import command
    from microduck_lab.tasks.rope_traverse.fit import HOME,I
    home=HOME.copy()
    q,mouth,_=command(home,1.3,'traverse',.16)
    assert mouth==.5
    for name,index in I.items():
        if name.startswith(('left_','right_')):assert q[index]==home[index]
    for t,stationary_side in [(1.9,'right_'),(2.5,'left_')]:
        q,mouth,_=command(home,t,'traverse',.16)
        assert mouth==0.
        for name,index in I.items():
            if name.startswith(stationary_side):assert q[index]==home[index]
    np.testing.assert_array_equal(home,HOME)
