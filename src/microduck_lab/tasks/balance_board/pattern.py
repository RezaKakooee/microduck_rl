#!/usr/bin/env python3
"""Varied board play — a routine of DIFFERENT moves, not one shape repeating.

The symmetric rocker drives `Re(wave * exp(i*2*pi*f*t))`: one shape, forever.
The asymmetric one skews and biases that shape, so the cycle is lopsided — but
it is still the SAME lopsided cycle every time. Looking at the roll trace, the
pattern repeats identically.

Here the four motion parameters become functions of time:

    theta(t) = integral of 2*pi*f(t) dt          <- phase, INTEGRATED, so the
                                                    rate can change smoothly
    theta_eff = theta + skew(t) * sin(theta)     <- unequal dwell
    amp       = A(t) * (1 + bias(t)*cos(theta_eff))

    reference   = amp * Re(state_wave  * exp(i*theta_eff))
    feedforward = amp * Re(action_wave * exp(i*theta_eff))

The phase must be INTEGRATED rather than evaluated as `2*pi*f*t`: with f varying,
the latter jumps the phase every time f changes, which the board cannot follow.

Two programs:

  `choreography` — a fixed routine of named moves (slow and wide, fast and
      small, a long lean to one side, a build-up, a flutter), cross-faded so the
      parameters never step. Each move respects the measured speed/swing
      envelope: ~12 deg is safe at 1.2 Hz, ~6 deg at 2 Hz, ~4 deg at 3 Hz
      (see docs/tasks/balance_board_rocking_asym.md).

  `wander` — smooth pseudo-random modulation: each parameter is a sum of sines
      with incommensurate periods, so the motion never repeats. Deterministic
      from `--seed`; no RNG is called inside the control loop.

The feedforward is solved for ONE rate (1.2 Hz), so any deviation leaves a
torque mismatch the LQR feedback must absorb. That already happens in the
asymmetric version — skew 0.9 swings the instantaneous rate by +/-90% and it
holds — which is why re-solving per segment is not needed here.

    uv run python src/microduck_lab/tasks/balance_board/pattern.py --program choreography --seconds 34
    uv run python src/microduck_lab/tasks/balance_board/pattern.py --program wander --seconds 30 --seed 3
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np


from microduck_lab.tasks.balance_board import lqr
from microduck_lab.tasks.balance_board import rocking
from microduck_lab.tasks.balance_board.rocking_asym import DEFAULT_POLICY, _board  # noqa: E402

# name, duration, frequency Hz, amplitude deg, skew, bias
ROUTINE = [
    ("slow and wide",      7.0, 1.0, 12.0, 0.9, 0.20),
    ("quick and small",    5.0, 2.0,  6.0, 0.3, 0.00),
    ("long lean, one way", 5.0, 0.9,  8.0, 0.9, 0.55),
    ("build up",           6.0, 1.4,  9.0, 0.6, 0.10),
    ("flutter",            4.0, 3.0,  4.0, 0.2, 0.00),
    ("slow and wide again", 7.0, 1.1, 11.0, 0.9, 0.25),
]
BLEND_S = 1.0     # cross-fade between moves, so nothing steps

# The same routine, plus ONE extra move at the end that is deliberately outside
# the measured envelope, so the duck loses the board. Amplitude 12 at 1.2 Hz is
# the largest that holds; 13 lets go at ~16 s and 14 at ~9 s, always by a FOOT
# LIFTING OFF THE PLANK past about 20 deg of tilt (no ankle-roll joint, so the
# sole tilts with the board). 18 deg of commanded swing is well past that.
FINALE = ("too far", 8.0, 1.2, 18.0, 0.9, 0.30)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Routine:
    """Piecewise-constant moves, cross-faded.

    `loop=False` holds the last move instead of wrapping round, which is what a
    routine ending in a deliberate fall needs.
    """

    def __init__(self, moves=ROUTINE, blend=BLEND_S, loop=True):
        self.moves, self.blend, self.loop = list(moves), float(blend), bool(loop)
        self.edges = np.cumsum([0.0] + [m[1] for m in self.moves])
        self.total = float(self.edges[-1])

    def label(self, t):
        tt = t % self.total if self.loop else min(t, self.total - 1e-6)
        i = int(np.searchsorted(self.edges, tt, "right") - 1)
        return self.moves[min(max(i, 0), len(self.moves) - 1)][0]

    def __call__(self, t):
        tt = t % self.total if self.loop else min(t, self.total - 1e-6)
        i = int(np.searchsorted(self.edges, tt, "right") - 1)
        i = min(max(i, 0), len(self.moves) - 1)
        cur = np.array(self.moves[i][2:], dtype=float)
        # fade in from the previous move over the first `blend` seconds
        into = tt - self.edges[i]
        if into < self.blend:
            prev = np.array(self.moves[(i - 1) % len(self.moves)][2:], dtype=float)
            cur = prev + (cur - prev) * _smoothstep(into / self.blend)
        return cur


class Wander:
    """Each parameter = centre + sum of sines with incommensurate periods."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # irrational-ish period ratios so nothing lines up again
        self.periods = np.array([6.1, 9.7, 14.3, 21.9]) * (1 + 0.1 * rng.random(4))
        self.phase = rng.random((4, 4)) * 2 * np.pi
        self.centre = np.array([1.30, 8.5, 0.55, 0.15])          # f, A, skew, bias
        self.swing = np.array([0.45, 3.0, 0.35, 0.20])
        self.lo = np.array([0.85, 4.5, 0.10, -0.35])
        self.hi = np.array([2.00, 12.0, 0.95, 0.50])

    def label(self, t):
        return "wander"

    def __call__(self, t):
        w = np.sin(2 * np.pi * t / self.periods[None, :] + self.phase).mean(axis=1)
        return np.clip(self.centre + self.swing * w * 1.6, self.lo, self.hi)


class PatternRocking(rocking.RockingExpert):
    """The solved wave, driven by a time-varying programme."""

    name = "pattern_rocking_lqr"

    def __init__(self, base, program, ramp_seconds=3.0):
        super().__init__(base, base.state_wave, base.action_wave,
                         amplitude_deg=base.motion["amplitude_deg"],
                         frequency_hz=base.motion["frequency_hz"],
                         ramp_seconds=ramp_seconds)
        self.program = program
        self.base_amp = base.motion["amplitude_deg"]
        self.theta = 0.0
        self._last_t = None
        self.log = []

    def __call__(self, state, dt):
        if not np.isclose(dt, lqr.DT):
            raise ValueError(f"Expert requires {lqr.DT} s control interval")
        t = float(state["time"])
        if self._last_t is not None and t < self._last_t - 1e-9:
            raise ValueError("time went backwards; the phase integral is invalid")
        self._last_t = t

        f, amp_deg, skew, bias = self.program(t)
        # Integrate the phase. Evaluating 2*pi*f*t directly would jump the phase
        # whenever f changes, and the board cannot follow a phase step.
        self.theta += 2 * np.pi * f * dt
        theta_eff = self.theta + skew * np.sin(self.theta)

        envelope = _smoothstep(t / self.motion["ramp_seconds"])
        scale = envelope * (amp_deg / self.base_amp) * (1.0 + bias * np.cos(theta_eff))
        phase = np.exp(1j * theta_eff)
        reference = scale * (self.state_wave * phase).real
        feedforward = scale * (self.action_wave * phase).real
        x = lqr.state_error(self.model, self.q0, state["qpos"], state["qvel"])
        command = np.clip(self.u0 + feedforward - self.gain @ (x - reference),
                          self.lower, self.upper)
        if not np.isfinite(command).all():
            raise ValueError("Non-finite observation or action")
        self.log.append((t, f, amp_deg, skew, bias))
        return command, float(np.max(np.abs(command - self.u0)))


def build(program="choreography", seed=0, policy=DEFAULT_POLICY):
    base = rocking.RockingExpert.load(policy, _board())
    if program == "wander":
        prog = Wander(seed)
    elif program == "finale":
        # identical to `choreography`, with ONE move added at the end
        prog = Routine(ROUTINE + [FINALE], loop=False)
    else:
        prog = Routine()
    return PatternRocking(base, prog)


def one(seconds=34.0, program="choreography", seed=0, trace_path=None, **kw):
    expert = build(program, seed)
    result, trace = lqr.evaluate(expert, seconds=seconds, return_trace=True, **kw)
    result["balance_success"] = result["success"]
    trace = np.asarray(trace)
    t, roll = trace[:, 0] - trace[0, 0], np.degrees(trace[:, 1])
    m = t >= 3.0
    result["peak_pos"] = float(roll[m].max())
    result["peak_neg"] = float(roll[m].min())
    result["program"] = program
    if trace_path:
        np.savetxt(trace_path, trace, delimiter=",", comments="",
                   header="time,board_roll,roller_position")
        np.savetxt(str(trace_path).replace(".csv", "_prog.csv"),
                   np.asarray(expert.log), delimiter=",", comments="",
                   header="time,frequency_hz,amplitude_deg,skew,bias")
    return result


def render_including_the_fall(program="finale", seconds=40.0, after_fall_s=4.0,
                              video=None, azimuth=180.0, seed=0,
                              policy=DEFAULT_POLICY):
    """Like lqr.evaluate, but KEEP SIMULATING after the failure is detected.

    `lqr.evaluate` breaks out of its loop the instant a failure condition trips,
    which is correct for scoring but ends a recording at the moment the fall
    STARTS. The duck is still upright in the last frame. Here the failure time
    is recorded and the episode continues for `after_fall_s` more seconds so the
    duck actually reaches the floor on camera.

    Only for rendering. Everything measured elsewhere uses lqr.evaluate.
    """
    expert = build(program, seed, policy)
    cfg = expert.config
    board = lqr.tb.Board("A", radius=cfg["radius"], plank_len=cfg["plank_len"],
                         plank_thk=cfg["plank_thk"])
    m, d = board.model, board.data
    lqr.prepare(board)
    mujoco.mj_forward(m, d)
    start = d.time
    rec = lqr.tb.duck_sim.Recorder(m, distance=.8, azimuth=azimuth, elevation=-12) if video else None
    if rec:
        rec.maybe_capture(0, d)

    fail, fail_t, held = None, None, 0.0
    absent = {"left": 0.0, "right": 0.0}
    total = seconds + after_fall_s
    for i in range(round(total / lqr.DT)):
        command, _ = expert({"qpos": d.qpos, "qvel": d.qvel, "time": i * lqr.DT}, lqr.DT)
        d.ctrl[:] = command
        for _ in range(lqr.tb.DECIMATION):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            elapsed = float(d.time - start)
            if fail is None:
                pf, fp, ff, bad = board.contacts()
                for side in absent:
                    absent[side] = 0.0 if fp[side] else absent[side] + m.opt.timestep
                if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                    fail = "non-finite physics state"
                elif pf:
                    fail = "plank touched floor"
                elif any(ff.values()):
                    fail = "foot touched floor"
                elif bad:
                    fail = f"non-foot robot body contact: {bad}"
                elif max(absent.values()) > lqr.tb.FOOT_GRACE_S + 1e-9:
                    fail = "foot left plank for more than 0.10 s"
                if fail:
                    fail_t = elapsed
                else:
                    held = elapsed
        if rec:
            rec.maybe_capture(i + 1, d)
        if fail is not None and elapsed > fail_t + after_fall_s:
            break
    if rec:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        rec.write(video)
        rec.renderer.close()
    trunk_z = float(d.xpos[board.trunk][2])
    return dict(held=held, fail=fail, fail_t=fail_t, final_trunk_z=trunk_z,
                landed=trunk_z < 0.09, video=video)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", choices=("choreography", "wander", "finale"),
                   default="choreography",
                   help="finale = the choreography plus one final move past the limit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=34.0)
    p.add_argument("--video")
    p.add_argument("--azimuth", type=float, default=180.0)   # 180 = FRONT
    p.add_argument("--trace")
    p.add_argument("--show-fall", action="store_true",
                   help="keep simulating after the fall so the duck reaches the floor")
    p.add_argument("--after-fall", type=float, default=4.0)
    a = p.parse_args()
    if a.show_fall:
        r = render_including_the_fall(program=a.program, seconds=a.seconds,
                                      after_fall_s=a.after_fall, video=a.video,
                                      azimuth=a.azimuth, seed=a.seed)
        print(f"{a.program}: rode {r['held']:.2f}s, fell at {r['fail_t']:.2f}s ({r['fail']})")
        print(f"kept filming {a.after_fall:.0f}s more; final trunk height "
              f"{1000*r['final_trunk_z']:.0f} mm -> {'on the floor' if r['landed'] else 'NOT down yet'}")
        return
    r = one(seconds=a.seconds, program=a.program, seed=a.seed,
            video=a.video, azimuth=a.azimuth, trace_path=a.trace)
    print(f"{a.program}: held {r['held']:.2f}s of {a.seconds:.0f}, "
          f"board roll peak {r['peak_pos']:+.1f} / {r['peak_neg']:+.1f} deg")
    if a.program == "finale":
        print(f"fell at {r['held']:.2f}s ({r['fail']})" if not r["balance_success"]
              else "did NOT fall — the finale move was not extreme enough")
    else:
        print("PASS" if r["balance_success"] else f"FAIL: {r['fail']}")


if __name__ == "__main__":
    main()
