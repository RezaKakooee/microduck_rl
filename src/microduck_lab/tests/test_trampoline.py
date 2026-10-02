"""The trampoline bed must be passive, must not add energy, and must hold the feet."""
import numpy as np
import pytest

from microduck_lab.tasks.jump.world import Upright
from microduck_lab.tasks.trampoline.world import (
    BED_MASS, TrampolineWorld, bounce_report, fit_damping, rigid_rebound)

MASS = 0.7372
K = MASS * 9.81 / 0.020


def test_bed_is_not_actuated():
    w = TrampolineWorld()
    m = w.model
    assert w.bed_q not in set(m.jnt_qposadr[m.actuator_trnid[:, 0]].tolist())
    assert len(w.names) == 14


def test_bed_contact_does_not_add_energy():
    # No damper: the weight loses only the impact on the light bed.
    theory = (MASS / (MASS + BED_MASS)) ** 2
    assert rigid_rebound(K, 0.0, MASS) == pytest.approx(theory, abs=0.02)


def test_damper_fit_hits_the_asked_rebound():
    c = fit_damping(K, MASS, 0.6)
    assert rigid_rebound(K, c, MASS) == pytest.approx(0.6, abs=0.005)


def test_stiff_bed_needs_a_smaller_step():
    with pytest.raises(ValueError):
        TrampolineWorld(timestep=0.005, decimation=4)


def test_drop_lands_on_the_feet_and_stays_upright():
    w = TrampolineWorld()
    w.place(w.stand, drop_mm=20.0)
    up = Upright(w)
    for _ in range(int(1.5 / w.control_dt)):
        w.step(up(w.stand))
    r = bounce_report(w.log)
    assert not r["fell"]
    assert r["max_foot_depth_mm"] < 2.0     # the feet never pass into the bed
    assert r["bounces"] >= 2                 # the drop, then at least one rebound
    assert np.isfinite(np.array(w.log)).all()


def test_level_keeps_the_soles_flat_in_the_air():
    import mujoco
    from microduck_lab.tasks.trampoline.pump import Pump, PumpParams
    w = TrampolineWorld()
    pump = Pump(w, PumpParams())
    m, d = w.model, w.data
    for axis in ([0, 1, 0], [1, 0, 0]):
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array(axis, float), np.radians(6))
        d.qpos[w.root:w.root + 7] = [0, 0, 0.4, *quat]
        d.qpos[w.qidx] = pump.level(pump.pose(20.0))
        mujoco.mj_kinematics(m, d)
        for g in w.foot_boxes:
            assert d.geom_xmat[g].reshape(3, 3)[2, 2] > np.cos(np.radians(0.5))


def test_pumping_makes_the_bounces_grow_and_stays_up():
    from microduck_lab.tasks.trampoline.pump import PumpParams, run
    r, _ = run(PumpParams(), seconds=8.0, tail_s=3.0)
    assert not r["fell"]
    assert r["max_drift_mm"] < 80
    first = r["flights"][0]["flight_ms"]
    assert r["tail"]["flight_ms"] > 150            # the passive rebound gave 84 ms
    assert r["tail"]["flight_ms"] > 1.5 * first     # the bounces grow


def test_front_flip_lands_on_the_feet():
    # The best flip so far (clip 16): a full turn in the air, feet first on the
    # bed. It still falls about 1.7 s later (off the front edge), so the test
    # only checks the landing.
    from microduck_lab.tasks.trampoline.flip import FlipParams, run
    p = FlipParams(target_deg=300, open_ratio=0.6, open_depth=40, open_neck=-0.5,
                   recover_amp=0, lean=0, release_delay=0.04, pre_x=0.0)
    out, _ = run(p, seconds=11.5)
    e = out["events"]
    assert e["landing_on_bed"]
    assert 300 <= e["landing_rotation_deg"] <= 380
    assert e["upright_after_landing_s"] > 0.3


def test_hybrid_pump_then_rl_flip():
    # Scripted pump from standing, legs to the standing pose on a high bounce,
    # then the RL flip policy (run 2's ONNX) takes over at the top (clip 53).
    import os
    from microduck_lab.tasks.trampoline.hybrid import RUN2_ONNX, run
    if not os.path.exists(RUN2_ONNX):
        pytest.skip("run 2 ONNX not present (logs/ is not in git)")
    out, _, _ = run(switch_h=0.25, seconds=13.0)
    assert out["switched_at_s"] is not None
    assert out["flip"] and out["flip_flight"]["by"] == "rl"
    assert out["flip_flight"]["rotation_deg"] >= 300 and out["flip_flight"]["landed_on_feet"]
    assert out["fell_at_s"] is None
