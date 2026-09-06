#!/usr/bin/env python3
"""Actively rock the loose balance board while keeping the duck on both feet.

A periodic reference is solved from the coupled robot/plank/roller dynamics.
All movement comes from the existing 14 servos. The platform is never forced,
animated, pinned or kicked to create the rocking. Full simulator state is
required, just as for the static LQR expert.

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking --suite
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking --video videos/balance_board/board_rocking.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microduck_lab.tasks.balance_board import lqr


class RockingExpert(lqr.LQRExpert):
    """Periodic feedforward plus whole-body feedback about the moving reference."""
    name = "active_rocking_lqr"

    def __init__(self, base, state_wave, action_wave, *, amplitude_deg=8.0,
                 frequency_hz=1.2, ramp_seconds=3.0):
        super().__init__(base.model, base.q0, base.u0, base.gain,
                         base.config, base.diagnostics)
        self.state_wave = np.asarray(state_wave, dtype=complex).copy()
        self.action_wave = np.asarray(action_wave, dtype=complex).copy()
        self.motion = dict(amplitude_deg=amplitude_deg, frequency_hz=frequency_hz,
                           ramp_seconds=ramp_seconds)
        if not all(np.isfinite(v) and v > 0 for v in self.motion.values()):
            raise ValueError("Amplitude, frequency and ramp duration must be positive and finite")
        if self.state_wave.shape != (2 * self.model.nv,) or self.action_wave.shape != (14,):
            raise ValueError("Periodic reference dimensions do not match the robot")
        if not (np.isfinite(self.state_wave).all() and np.isfinite(self.action_wave).all()):
            raise ValueError("Periodic reference contains non-finite values")

    def __call__(self, state, dt):
        if not np.isclose(dt, lqr.DT):
            raise ValueError(f"Expert requires {lqr.DT} s control interval")
        t = float(state["time"])
        fraction = np.clip(t / self.motion["ramp_seconds"], 0, 1)
        envelope = fraction * fraction * (3 - 2 * fraction)
        phase = np.exp(2j * np.pi * self.motion["frequency_hz"] * t)
        reference = envelope * (self.state_wave * phase).real
        feedforward = envelope * (self.action_wave * phase).real
        x = lqr.state_error(self.model, self.q0, state["qpos"], state["qvel"])
        command = np.clip(self.u0 + feedforward - self.gain @ (x - reference),
                          self.lower, self.upper)
        if not np.isfinite(command).all():
            raise ValueError("Non-finite observation or action")
        return command, float(np.max(np.abs(command - self.u0)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q0=self.q0, u0=self.u0, gain=self.gain,
                            state_wave=self.state_wave, action_wave=self.action_wave,
                            config=json.dumps(self.config), motion=json.dumps(self.motion),
                            diagnostics=json.dumps(self.diagnostics))

    @classmethod
    def load(cls, path, board):
        with np.load(path, allow_pickle=False) as z:
            config = json.loads(str(z["config"]))
            if config != lqr.configuration(board):
                raise ValueError("Saved rocking policy does not match this board")
            base = lqr.LQRExpert(board.model, z["q0"], z["u0"], z["gain"], config,
                                 json.loads(str(z["diagnostics"])))
            return cls(base, z["state_wave"], z["action_wave"], **json.loads(str(z["motion"])))


def design(board, *, amplitude_deg=8.0, frequency_hz=1.2, ramp_seconds=3.0):
    """Find a physically coupled sinusoid; merely tilting a pose is insufficient.

    For z=exp(j*w*dt), H=(zI-A+BK)^-1 B maps a periodic forcing to state
    amplitude. F=I-KH maps it to the resulting servo-target amplitude.
    Minimize state/effort cost subject to the board's roll amplitude. Complex
    amplitudes retain the phase shifts required to move the loose roller.
    """
    if not all(np.isfinite(v) and v > 0 for v in (amplitude_deg, frequency_hz, ramp_seconds)):
        raise ValueError("Motion parameters must be positive and finite")
    base = lqr.design(board)
    a, b = base.dynamics
    m = board.model
    n = 2 * m.nv
    z = np.exp(2j * np.pi * frequency_hz * lqr.DT)
    response = np.linalg.solve(z * np.eye(n) - (a - b @ base.gain), b)
    action_response = np.eye(14) - base.gain @ response
    q = np.eye(n) * .01
    q[:m.nv, :m.nv] = np.eye(m.nv)
    for v in (board.vadr, board.pvadr, board.cvadr):
        q[v:v + 3, v:v + 3] = np.eye(3) * 100
        q[v + 3:v + 6, v + 3:v + 6] = np.eye(3) * 10
    weight = response.conj().T @ q @ response + .1 * action_response.conj().T @ action_response
    constraint = response[board.pvadr + 3]
    direction = np.linalg.solve(weight, constraint.conj())
    forcing = direction * (-np.radians(amplitude_deg)) / (constraint @ direction)
    return RockingExpert(base, response @ forcing, action_response @ forcing,
                         amplitude_deg=amplitude_deg, frequency_hz=frequency_hz,
                         ramp_seconds=ramp_seconds)


def score_motion(trajectory, motion, seconds):
    """Every complete cycle must tilt visibly BOTH ways and move the roller.

    A long still hold, a single push, or a motion that fades late in the video
    cannot pass. Ignore only the explicitly specified initial amplitude ramp.
    """
    period = 1.0 / motion["frequency_hz"]
    ramp = motion["ramp_seconds"]
    count = max(0, int(np.floor((seconds - ramp) / period + 1e-8)))
    threshold = .7 * motion["amplitude_deg"]
    cycles = []
    for i in range(count):
        start, stop = ramp + i * period, ramp + (i + 1) * period
        samples = trajectory[(trajectory[:, 0] >= start - 1e-8) &
                             (trajectory[:, 0] <= stop + 1e-8)]
        complete = len(samples) >= int(period / lqr.DT) - 1
        lo = float(np.degrees(samples[:, 1].min())) if len(samples) else 0.0
        hi = float(np.degrees(samples[:, 1].max())) if len(samples) else 0.0
        travel = float(np.ptp(samples[:, 4]) * 1000) if len(samples) else 0.0
        passed = complete and lo <= -threshold and hi >= threshold and travel >= 5.0
        cycles.append(dict(start_s=start, end_s=stop, min_tilt_deg=lo,
                           max_tilt_deg=hi, roller_travel_mm=travel, passed=bool(passed)))
    return {"motion": dict(motion), "required_peak_each_side_deg": threshold,
            "expected_cycles": count, "completed_rocking_cycles": sum(c["passed"] for c in cycles),
            "motion_success": bool(count > 0 and all(c["passed"] for c in cycles)),
            "cycles": cycles}


def evaluate(expert, **kwargs):
    result, trajectory = lqr.evaluate(expert, return_trace=True, **kwargs)
    result["balance_success"] = result["success"]
    result.update(score_motion(trajectory, expert.motion, result["seconds"]))
    result["success"] = result["balance_success"] and result["motion_success"]
    if result["balance_success"] and not result["motion_success"]:
        result["fail"] = "balanced, but did not rock visibly in every complete cycle"
    return result


def suite(expert):
    cases = [{"seconds": 60.0}]
    cases += [{"seed": seed, "seconds": 20.0} for seed in range(10)]
    cases += [{"push": p, "push_at": t, "seconds": 20.0}
              for p in (-.01, .01) for t in (5.0, 5.2)]
    cases += [{"friction": f, "seconds": 20.0} for f in (1.0, 1.8)]
    results = []
    for case in cases:
        result = evaluate(expert, **case)
        results.append(result)
        print(f"{len(results):2d}/{len(cases)} {case}: {result['held']:.3f}s, "
              f"{result['completed_rocking_cycles']}/{result['expected_cycles']} cycles, "
              f"{result['fail'] or 'PASS'}", flush=True)
    return dict(passed=sum(r["success"] for r in results), total=len(results), episodes=results)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--amplitude-deg", type=float, default=8.0)
    p.add_argument("--frequency-hz", type=float, default=1.2)
    p.add_argument("--ramp-seconds", type=float, default=3.0)
    p.add_argument("--video")
    p.add_argument("--azimuth", type=float, default=180.0)
    p.add_argument("--trace")
    p.add_argument("--output")
    p.add_argument("--save-policy")
    p.add_argument("--load-policy")
    p.add_argument("--suite", action="store_true")
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error("seconds must be positive and finite")
    board = lqr.tb.Board("A")
    lqr.prepare(board)
    expert = RockingExpert.load(args.load_policy, board) if args.load_policy else design(
        board, amplitude_deg=args.amplitude_deg, frequency_hz=args.frequency_hz,
        ramp_seconds=args.ramp_seconds)
    result = suite(expert) if args.suite else evaluate(
        expert, seconds=args.seconds, video=args.video, azimuth=args.azimuth, trace=args.trace)
    if args.save_policy:
        expert.save(args.save_policy)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "cycles")}, indent=2))
    if (args.suite and result["passed"] != result["total"]) or (not args.suite and not result["success"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
