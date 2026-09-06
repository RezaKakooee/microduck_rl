#!/usr/bin/env python3
"""Whole-body LQR expert for the free-roller balance task, at 50 Hz.

Uses the unchanged Board physics, robot, current limits and one-second held
setup. Finite differences identify the complete 20 ms transition (including
contacts and servo dynamics); a finite-horizon Riccati recursion produces a
constant state-feedback policy. No training, extra joints or larger board.

This is a SIMULATION expert: observations include qpos and qvel for the robot,
plank and free roller. It is not a 61D ONNX policy for robotd. The controller
only returns the 14 position targets; it never writes simulator state/forces.

  .venv/bin/python -m microduck_lab.tasks.balance_board.lqr
  .venv/bin/python -m microduck_lab.tasks.balance_board.lqr --suite --output videos/balance_board/board_lqr_validation.json
  MUJOCO_GL=egl .venv/bin/python -m microduck_lab.tasks.balance_board.lqr --seconds 30 --push .02 --video videos/balance_board/board_lqr.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.tasks.balance_board import board as tb

DT = tb.CONTROL_DT


def prepare(board: tb.Board) -> None:
    """The original protocol: place, hold one second with upright PD, release."""
    board.place()
    board.set_held(True)
    stand = tb.PD("A")
    for _ in range(round(tb.SETTLE_S / DT)):
        state = board.state()
        state["released"] = False
        board.data.ctrl[:] = stand(state, DT)[0]
        for _ in range(tb.DECIMATION):
            board.hold_board()
            mujoco.mj_step(board.model, board.data)
        board.hold_board()
        mujoco.mj_forward(board.model, board.data)
    board.set_held(False)
    mujoco.mj_forward(board.model, board.data)


def configuration(board: tb.Board) -> dict:
    m = board.model
    return {
        "orientation": "A", "radius": board.radius, "plank_len": board.plank_len,
        "plank_thk": board.plank_thk, "plank_width": tb.PLANK_WID,
        "roller_mass": float(m.body_mass[board.cyl_b]),
        "plank_mass": float(m.body_mass[board.plank_b]),
        "robot_mass": float(m.body_mass[board.robot_bodies].sum()),
        "physics_dt": m.opt.timestep, "control_dt": DT,
        "mujoco_version": mujoco.__version__, "fixed_roller": False,
    }


def state_error(model, reference, qpos, qvel):
    x = np.empty(2 * model.nv)
    mujoco.mj_differentiatePos(model, x[:model.nv], 1.0, reference, qpos)
    x[model.nv:] = qvel
    return x


class LQRExpert:
    """Constant feedback u = u0 - K x, followed by the actual control limits."""
    name = "whole_body_lqr"

    def __init__(self, model, q0, u0, gain, config, diagnostics=None, dynamics=None):
        self.model = model
        self.q0 = np.asarray(q0).copy()
        self.u0 = np.asarray(u0).copy()
        self.gain = np.asarray(gain).copy()
        self.config = dict(config)
        self.diagnostics = diagnostics or {}
        self.dynamics = dynamics
        if self.gain.shape != (14, 2 * model.nv):
            raise ValueError("Expected a 14-action gain matrix for this model")
        self.lower = model.actuator_ctrlrange[:, 0].copy()
        self.upper = model.actuator_ctrlrange[:, 1].copy()
        unlimited = ~model.actuator_ctrllimited.astype(bool)
        self.lower[unlimited], self.upper[unlimited] = -np.inf, np.inf

    def __call__(self, state, dt):
        if not np.isclose(dt, DT):
            raise ValueError(f"Expert requires {DT} s control interval")
        x = state_error(self.model, self.q0, state["qpos"], state["qvel"])
        command = np.clip(self.u0 - self.gain @ x, self.lower, self.upper)
        if not np.isfinite(command).all():
            raise ValueError("Non-finite observation or action")
        return command, float(np.max(np.abs(command - self.u0)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q0=self.q0, u0=self.u0, gain=self.gain,
                            config=json.dumps(self.config), diagnostics=json.dumps(self.diagnostics))

    @classmethod
    def load(cls, path, board):
        with np.load(path, allow_pickle=False) as z:
            config = json.loads(str(z["config"]))
            if config != configuration(board):
                raise ValueError("Saved expert physics/configuration does not match this board")
            return cls(board.model, z["q0"], z["u0"], z["gain"], config,
                       json.loads(str(z["diagnostics"])))


def linearize(board: tb.Board, epsilon=1e-5):
    """Differentiate the complete 20 ms transition in independent scratch data."""
    m, d = board.model, board.data
    q0, u0 = d.qpos.copy(), d.ctrl.copy()
    scratch = mujoco.MjData(m)
    n = 2 * m.nv

    def transition(x, u):
        mujoco.mj_resetData(m, scratch)
        scratch.qpos[:] = q0
        mujoco.mj_integratePos(m, scratch.qpos, x[:m.nv], 1.0)
        scratch.qvel[:] = x[m.nv:]
        scratch.ctrl[:] = u0 + u
        for _ in range(tb.DECIMATION):
            mujoco.mj_step(m, scratch)
        return state_error(m, q0, scratch.qpos, scratch.qvel)

    axes_x, axes_u = np.eye(n) * epsilon, np.eye(m.nu) * epsilon
    zero_x, zero_u = np.zeros(n), np.zeros(m.nu)
    a = np.column_stack([(transition(v, zero_u) - transition(-v, zero_u)) / (2 * epsilon)
                         for v in axes_x])
    b = np.column_stack([(transition(zero_x, v) - transition(zero_x, -v)) / (2 * epsilon)
                         for v in axes_u])
    return q0, u0, a, b


def design(board: tb.Board, effort=0.1, horizon=1000, epsilon=1e-5) -> LQRExpert:
    """Identify coupled contact/servo dynamics and solve finite-horizon LQR.

    The nominal state is the original held-board standing pose. Identification
    leaves the live simulation untouched. Feedback rejects the small vertical
    contact transient on release.
    """
    m = board.model
    n = 2 * m.nv
    q0, u0, a, b = linearize(board, epsilon)
    q = np.eye(n) * 0.01
    q[:m.nv, :m.nv] = np.eye(m.nv)
    for v in (board.vadr, board.cvadr, board.pvadr):
        q[v:v + 3, v:v + 3] = np.eye(3) * 100
        q[v + 3:v + 6, v + 3:v + 6] = np.eye(3) * 10
    q[board.pvadr + 3, board.pvadr + 3] = 100
    r = np.eye(m.nu) * effort
    # A loose roller has nearly neutral coordinates (including axial position
    # and spin). A finite horizon avoids pretending every gauge is stabilizable
    # to an exact world pose, as an infinite-horizon DARE would require.
    p = q.copy()
    for _ in range(horizon):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        p = q + a.T @ p @ (a - b @ gain)
        p = (p + p.T) * 0.5
    diagnostics = {
        "open_loop_spectral_radius": float(np.abs(np.linalg.eigvals(a)).max()),
        "feedback_spectral_radius": float(np.abs(np.linalg.eigvals(a - b @ gain)).max()),
        "riccati_horizon_steps": horizon, "effort_weight": effort,
        "finite_difference_epsilon": epsilon,
    }
    return LQRExpert(m, q0, u0, gain, configuration(board), diagnostics, dynamics=(a, b))


def evaluate(expert, *, seconds=10.0, push=0.0, push_at=2.0, seed=None,
             friction=None, mass_scale=1.0, video=None, azimuth=180.0, trace=None,
             return_trace=False):
    """Fresh episode with contact checks at EVERY 5 ms physics step.

    Random stress applies base/joint velocity disturbances at release. A push
    is a single base lateral velocity kick, the same convention as tb.run.
    Model mass/friction changes happen after nominal controller design.
    With return_trace=True, return (result, trajectory) for motion scoring.
    """
    cfg = expert.config
    board = tb.Board("A", radius=cfg["radius"], plank_len=cfg["plank_len"],
                     plank_thk=cfg["plank_thk"])
    m, d = board.model, board.data
    prepare(board)
    if friction is not None:
        m.geom_friction[:, 0] = friction
        m.pair_friction[:, :2] = friction
    if mass_scale != 1.0:
        m.body_mass[board.robot_bodies] *= mass_scale
        m.body_inertia[board.robot_bodies] *= mass_scale
        mujoco.mj_setConst(m, mujoco.MjData(m))
    if seed is not None:
        rng = np.random.default_rng(seed)
        d.qvel[board.vadr:board.vadr + 3] += rng.uniform(-.01, .01, 3)
        d.qvel[board.vadr + 3:board.vadr + 6] += rng.uniform(-.05, .05, 3)
        servo_dofs = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        d.qvel[servo_dofs] += rng.uniform(-.02, .02, 14)
    mujoco.mj_forward(m, d)
    start = d.time
    recorder = tb.duck_sim.Recorder(m, distance=.8, azimuth=azimuth, elevation=-12) if video else None
    if recorder:
        recorder.maybe_capture(0, d)
    out = {"controller": expert.name, "seconds": seconds, "held": 0.0,
           "success": False, "fail": None, "fail_t": None,
           "push": push, "push_at": push_at, "push_applied": False, "seed": seed,
           "friction_override": friction, "robot_mass_scale": mass_scale,
           "max_plank_tilt_deg": 0.0, "max_trunk_tilt_deg": 0.0,
           "max_roller_offset_mm": 0.0, "min_plank_floor_clearance_mm": float("inf"),
           "max_foot_airtime_s": 0.0, "max_action_delta_rad": 0.0,
           "plank_floor_contacts": 0, "foot_floor_contacts": 0, "bad_body_contacts": 0,
           "roller_contact_loss_s": 0.0, "configuration": cfg}
    absent = {"left": 0.0, "right": 0.0}
    roller_absent = 0.0
    rows = []
    for i in range(round(seconds / DT)):
        if push and not out["push_applied"] and i * DT >= push_at:
            d.qvel[board.vadr + board.axis] += push
            out["push_applied"] = True
        command, size = expert({"qpos": d.qpos, "qvel": d.qvel, "time": i * DT}, DT)
        d.ctrl[:] = command
        out["max_action_delta_rad"] = max(out["max_action_delta_rad"], size)
        for _ in range(tb.DECIMATION):
            mujoco.mj_step(m, d)
            # mj_step integrates qpos after computing xpos/contact. Refresh so
            # geometry, tilt and contact checks describe the SAME instant.
            mujoco.mj_forward(m, d)
            elapsed = float(d.time - start)
            state = board.state()
            pf, fp, ff, bad = board.contacts()
            for side in absent:
                absent[side] = 0.0 if fp[side] else absent[side] + m.opt.timestep
            roller_contact = any({c.geom1, c.geom2} == {board.plank_g, board.cyl_g}
                                 for c in d.contact)
            roller_absent = 0.0 if roller_contact else roller_absent + m.opt.timestep
            rot = d.geom_xmat[board.plank_g].reshape(3, 3)
            bottom = d.geom_xpos[board.plank_g, 2] - np.abs(rot[2]) @ m.geom_size[board.plank_g]
            out["min_plank_floor_clearance_mm"] = min(out["min_plank_floor_clearance_mm"], 1000 * bottom)
            out["max_foot_airtime_s"] = max(out["max_foot_airtime_s"], *absent.values())
            out["roller_contact_loss_s"] = max(out["roller_contact_loss_s"], roller_absent)
            out["max_plank_tilt_deg"] = max(out["max_plank_tilt_deg"], abs(np.degrees(state["plank_tilt"])))
            out["max_trunk_tilt_deg"] = max(out["max_trunk_tilt_deg"], abs(np.degrees(state["trunk_tilt"])))
            out["max_roller_offset_mm"] = max(out["max_roller_offset_mm"], 1000 * abs(state["s"]))
            out["plank_floor_contacts"] += pf
            out["foot_floor_contacts"] += sum(ff.values())
            out["bad_body_contacts"] += int(bad is not None)
            failure = None
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                failure = "non-finite physics state"
            elif pf or bottom <= 0:
                failure = "plank touched floor"
            elif any(ff.values()):
                failure = "foot touched floor"
            elif bad:
                failure = f"non-foot robot body contact: {bad}"
            elif max(absent.values()) > tb.FOOT_GRACE_S + 1e-9:
                failure = "foot left plank for more than 0.10 s"
            elif roller_absent > tb.FOOT_GRACE_S + 1e-9:
                failure = "plank left roller for more than 0.10 s"
            elif d.xpos[board.trunk, 2] - (d.xpos[board.plank_b, 2] + cfg["plank_thk"] / 2) < tb.FALL_HEIGHT:
                failure = "trunk below standing height threshold"
            if failure:
                out["fail"], out["fail_t"] = failure, elapsed
                break
            out["held"] = min(elapsed, seconds)
        if recorder:
            recorder.maybe_capture(i + 1, d)
        rows.append([min(d.time - start, seconds), state["plank_tilt"], state["trunk_tilt"],
                     state["s"], state["cyl_pos"][1], bottom, fp["left"], fp["right"]])
        if out["fail"]:
            break
    out["success"] = out["fail"] is None and out["held"] >= seconds - 1e-8
    if out["success"]:
        out["held"] = seconds
    if recorder:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        recorder.write(video)
        recorder.renderer.close()
        out["video"] = str(video)
    if trace:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(trace, rows, delimiter=",", comments="",
                   header="time_s,plank_roll_rad,trunk_roll_rad,roller_offset_m,roller_world_y_m,plank_clearance_m,left_contacts,right_contacts")
    return (out, np.asarray(rows)) if return_trace else out


def suite(expert):
    cases = [{"seconds": 60.0}]
    cases += [{"push": p} for p in (-.05, -.02, -.01, .01, .02, .05)]
    cases += [{"seed": seed} for seed in range(20)]
    cases += [{"friction": f, "push": .01} for f in (.8, 1.0, 1.2, 1.8)]
    cases += [{"mass_scale": scale, "push": .01} for scale in (.9, 1.1)]
    results = []
    for case in cases:
        result = evaluate(expert, **case)
        results.append(result)
        print(f"{len(results):2d}/{len(cases)} {case}: {result['held']:.3f}s, {result['fail'] or 'PASS'}", flush=True)
    return {"passed": sum(r["success"] for r in results), "total": len(results),
            "diagnostics": expert.diagnostics, "episodes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--radius", type=float, default=tb.CYL_RADIUS)
    parser.add_argument("--push", type=float, default=0.0)
    parser.add_argument("--push-at", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--video")
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--trace")
    parser.add_argument("--output")
    parser.add_argument("--save-policy")
    parser.add_argument("--load-policy")
    parser.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.radius <= 0:
        parser.error("seconds and radius must be positive")
    board = tb.Board("A", radius=args.radius)
    prepare(board)
    expert = LQRExpert.load(args.load_policy, board) if args.load_policy else design(board)
    if args.save_policy:
        expert.save(args.save_policy)
    result = suite(expert) if args.suite else evaluate(
        expert, seconds=args.seconds, push=args.push, push_at=args.push_at, seed=args.seed,
        video=args.video, azimuth=args.azimuth, trace=args.trace)
    print(json.dumps(result if not args.suite else {k: v for k, v in result.items() if k != "episodes"}, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if (args.suite and result["passed"] != result["total"]) or (not args.suite and not result["success"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
