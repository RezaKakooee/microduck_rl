#!/usr/bin/env python3
"""Asymmetric board play — rock the board UNEVENLY left and right.

The symmetric rocker (`task_balance_board_rocking.py`) drives a pure sinusoid:

    reference(t) = Re( state_wave * exp(i*2*pi*f*t) )

so the board roll is a clean +/-6.7 deg sine at a constant 1.2 Hz. That is a
limit cycle, and it is left/right symmetric by construction: every push one way
is mirrored by an identical push the other way, so the average lateral impulse
is exactly zero and the roller returns to the same place every cycle.

This script breaks that symmetry three ways, then measures whether the duck can
still hold the board. All three modulate the SAME solved wave, and are applied
to the reference and the feedforward together so the two stay consistent:

  1. skew  -- unequal dwell. Warp the phase so the duck lingers on one side and
     snaps through the other:
         theta(t) = 2*pi*f*t + skew * sin(2*pi*f*t)
     The path through state space is unchanged; only the speed along it varies.

  2. bias  -- unequal amplitude. Scale the two halves differently:
         amp(theta) = 1 + bias * cos(theta)
     bias = 0.25 gives +25% one way and -25% the other.

  3. drift -- break the exact periodicity, so it is not a clean limit cycle:
         amp *= 1 + drift * sin(2*pi*t / drift_period)

The feedforward was solved for a CONSTANT rate 2*pi*f, so warping the phase
leaves a torque mismatch that the LQR feedback has to absorb. That is the real
question this script answers: how much asymmetry the feedback can carry.

    uv run python src/microduck_lab/tasks/balance_board/rocking_asym.py --sweep
    uv run python src/microduck_lab/tasks/balance_board/rocking_asym.py --skew 0.6 --bias 0.25 \
        --seconds 20 --azimuth 180 --video videos/balance_board/board_rocking_asym.mp4

Does not touch the symmetric script, its policy file, or its videos.
"""

import argparse
from pathlib import Path

import numpy as np


from microduck_lab.tasks.balance_board import lqr
from microduck_lab.tasks.balance_board import rocking

DEFAULT_POLICY = "videos/balance_board/board_rocking_policy.npz"


class AsymmetricRocking(rocking.RockingExpert):
    """The solved periodic rocker, driven with an uneven phase and amplitude."""

    name = "asymmetric_rocking_lqr"

    def __init__(self, base, *, skew=0.0, bias=0.0, drift=0.0, drift_period=7.0, **kw):
        super().__init__(base, base.state_wave, base.action_wave, **{
            k: base.motion[k] for k in ("amplitude_deg", "frequency_hz", "ramp_seconds")
        } | kw)
        self.asym = dict(skew=float(skew), bias=float(bias),
                         drift=float(drift), drift_period=float(drift_period))
        if not 0.0 <= abs(self.asym["bias"]) < 1.0:
            raise ValueError("bias must be in (-1, 1); 1 would zero one half")
        if self.asym["drift_period"] <= 0:
            raise ValueError("drift_period must be positive")

    def __call__(self, state, dt):
        if not np.isclose(dt, lqr.DT):
            raise ValueError(f"Expert requires {lqr.DT} s control interval")
        t = float(state["time"])
        a = self.asym
        fraction = np.clip(t / self.motion["ramp_seconds"], 0, 1)
        envelope = fraction * fraction * (3 - 2 * fraction)

        base_phase = 2 * np.pi * self.motion["frequency_hz"] * t
        theta = base_phase + a["skew"] * np.sin(base_phase)      # unequal dwell
        amp = 1.0 + a["bias"] * np.cos(theta)                    # unequal amplitude
        if a["drift"]:
            amp *= 1.0 + a["drift"] * np.sin(2 * np.pi * t / a["drift_period"])

        phase = np.exp(1j * theta)
        scale = envelope * amp
        reference = scale * (self.state_wave * phase).real
        feedforward = scale * (self.action_wave * phase).real
        x = lqr.state_error(self.model, self.q0, state["qpos"], state["qvel"])
        command = np.clip(self.u0 + feedforward - self.gain @ (x - reference),
                          self.lower, self.upper)
        if not np.isfinite(command).all():
            raise ValueError("Non-finite observation or action")
        return command, float(np.max(np.abs(command - self.u0)))


_BOARD = None


def _board():
    """One Board per process; building it is the slow part."""
    global _BOARD
    if _BOARD is None:
        _BOARD = lqr.tb.Board("A")
        lqr.prepare(_BOARD)
    return _BOARD


def load(policy=DEFAULT_POLICY, frequency_hz=None, amplitude_deg=8.0,
         ramp_seconds=3.0, **asym):
    """Load the saved 1.2 Hz wave, or SOLVE a new one at another frequency.

    The wave is solved for one specific frequency: `state_wave` and
    `action_wave` are the periodic solution of the linearised robot/plank/roller
    dynamics at that rate. Merely driving the saved wave faster would leave the
    feedforward solving the wrong problem, so a new speed means a new solve.
    """
    board = _board()
    if frequency_hz is None:
        base = rocking.RockingExpert.load(policy, board)
    else:
        base = rocking.design(board, amplitude_deg=amplitude_deg,
                              frequency_hz=frequency_hz, ramp_seconds=ramp_seconds)
    return AsymmetricRocking(base, **asym)


def measure(trace, ramp_s):
    """Left/right asymmetry of the board roll, after the ramp."""
    t, roll = trace[:, 0], np.degrees(trace[:, 1])
    m = t >= t[0] + ramp_s
    r = roll[m]
    pos, neg = r[r > 0], r[r < 0]
    return dict(
        peak_pos=float(pos.max()) if pos.size else 0.0,
        peak_neg=float(neg.min()) if neg.size else 0.0,
        mean=float(r.mean()),
        frac_pos=float((r > 0).mean()),
    )


def one(seconds=20.0, policy=DEFAULT_POLICY, trace_path=None, **kw):
    asym = {k: kw.pop(k) for k in ("skew", "bias", "drift", "drift_period") if k in kw}
    design_kw = {k: kw.pop(k) for k in ("frequency_hz", "amplitude_deg", "ramp_seconds")
                 if k in kw}
    expert = load(policy, **design_kw, **asym)
    result, trace = lqr.evaluate(expert, seconds=seconds, return_trace=True, **kw)
    result["balance_success"] = result["success"]
    result.update(rocking.score_motion(trace, expert.motion, result["seconds"]))
    result["asym"] = measure(np.asarray(trace), expert.motion["ramp_seconds"])
    if trace_path:
        np.savetxt(trace_path, trace, delimiter=",", comments="",
                   header="time,board_roll,roller_position")
    return result


def speed_sweep(seconds=20.0, skew=0.9, bias=0.45, drift=0.3, amplitude_deg=8.0):
    """How fast can it rock? Each row solves a fresh wave at that frequency."""
    print(f"{'Hz':>5} {'held':>7} {'cycles':>9} {'peak +':>7} {'peak -':>7} {'result':>8}")
    for f in (1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0):
        try:
            r = one(seconds=seconds, frequency_hz=f, amplitude_deg=amplitude_deg,
                    skew=skew, bias=bias, drift=drift)
        except Exception as exc:                     # a solve can fail outright
            print(f"{f:5.1f} {'-':>7} {'-':>9} {'-':>7} {'-':>7}   solve failed: {exc}")
            continue
        a = r["asym"]
        print(f"{f:5.1f} {r['held']:7.2f} {r['completed_rocking_cycles']:4d}/"
              f"{r['expected_cycles']:<4d} {a['peak_pos']:+7.1f} {a['peak_neg']:+7.1f} "
              f"{'PASS' if r['balance_success'] else 'FELL':>8}")


def sweep(seconds=20.0):
    print(f"{'skew':>5} {'bias':>5} {'drift':>6} {'held':>7} {'cycles':>7} "
          f"{'peak +':>7} {'peak -':>7} {'mean':>6} {'t>0':>5}")
    rows = []
    for skew, bias, drift in [(0, 0, 0), (0.3, 0, 0), (0.6, 0, 0), (0.9, 0, 0),
                              (0, 0.25, 0), (0, 0.45, 0),
                              (0.6, 0.25, 0), (0.6, 0.25, 0.2), (0.9, 0.45, 0.3)]:
        r = one(seconds=seconds, skew=skew, bias=bias, drift=drift)
        a = r["asym"]
        rows.append((skew, bias, drift, r))
        print(f"{skew:5.2f} {bias:5.2f} {drift:6.2f} {r['held']:7.2f} "
              f"{r['completed_rocking_cycles']:3d}/{r['expected_cycles']:<3d} "
              f"{a['peak_pos']:+7.1f} {a['peak_neg']:+7.1f} {a['mean']:+6.2f} "
              f"{a['frac_pos']:5.2f}  {'' if r['balance_success'] else 'FELL'}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--speed-sweep", action="store_true")
    p.add_argument("--frequency-hz", type=float, default=None,
                   help="Solve a NEW wave at this rate (default: load the saved 1.2 Hz one)")
    p.add_argument("--amplitude-deg", type=float, default=8.0)
    p.add_argument("--skew", type=float, default=0.6)
    p.add_argument("--bias", type=float, default=0.25)
    p.add_argument("--drift", type=float, default=0.0)
    p.add_argument("--drift-period", type=float, default=7.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--video")
    p.add_argument("--azimuth", type=float, default=180.0)   # 180 = FRONT
    p.add_argument("--trace")
    a = p.parse_args()
    if a.speed_sweep:
        speed_sweep(seconds=a.seconds, skew=a.skew, bias=a.bias, drift=a.drift,
                    amplitude_deg=a.amplitude_deg)
        return
    if a.sweep:
        sweep(seconds=a.seconds)
        return
    extra = {} if a.frequency_hz is None else dict(frequency_hz=a.frequency_hz,
                                                   amplitude_deg=a.amplitude_deg)
    r = one(seconds=a.seconds, policy=a.policy, skew=a.skew, bias=a.bias,
            drift=a.drift, drift_period=a.drift_period, **extra,
            video=a.video, azimuth=a.azimuth, trace_path=a.trace)
    s = r["asym"]
    print(f"{a.frequency_hz or 1.2:.1f} Hz | held {r['held']:.2f}s of {a.seconds:.0f}  "
          f"cycles {r['completed_rocking_cycles']}/{r['expected_cycles']}")
    print(f"board roll: peak {s['peak_pos']:+.1f} / {s['peak_neg']:+.1f} deg, "
          f"mean {s['mean']:+.2f} deg, {100*s['frac_pos']:.0f}% of the time on the + side")
    print("PASS" if r["balance_success"] else f"FAIL: {r['fail']}")


if __name__ == "__main__":
    main()
