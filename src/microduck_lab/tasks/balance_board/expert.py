#!/usr/bin/env python3
"""Expert balance controller for the rocker board: open the stance, then hold
the board with DIFFERENTIAL LEG LENGTH.

Why this is different from the PD in task_balance_board.py
----------------------------------------------------------
That controller has one channel: lean the body sideways with both hip_rolls
(`c[L_HIP_ROLL] -= u; c[R_HIP_ROLL] -= u`). On a roller board that barely
works, because the ankle axis sits above the roller contact, so leaning does
not move the support under the body. Best measured: 2.22 s of 10.

A person on a wobble board does something else: they stand WIDE and press down
alternately with each foot. The torque about the roller is

    tau = (F_left - F_right) * d          d = half the foot separation

so two things matter, and neither is leaning:

  1. `d` -- the stance width. MEASURED on this model, opening both hip_rolls
     outward (left +, right -) from HOME:

        open (rad)   0.00   0.10   0.20   0.30   0.40   0.47
        foot sep(mm) 83.6  103.3  122.3  140.4  157.4  168.6
        sole tilt    0.0    5.7   11.5   17.2   22.9   26.9  deg

     0.30 rad gives 1.7x the lever of the home stance. The sole tilt is the
     cost: this robot has no ankle-roll joint, so an opened leg puts its sole
     on an edge (the same limitation that makes a one-legged pose impossible,
     see docs/project_journey/05).

  2. `F_left - F_right` -- made by changing one leg's LENGTH. Pressing a leg
     longer loads that foot. The three sagittal joints all tilt the sole
     equally (d(sole_pitch)/dq = +0.996, -0.996, +0.996 for hip_pitch, knee,
     ankle), so the sole-flat constraint is hip_pitch - knee + ankle = 0, which
     leaves a 2-D family. Maximising length change inside it gives

        SHORTEN_LEFT  = (hip_pitch, knee, ankle) = (0.521, 0.805, 0.284)
        SHORTEN_RIGHT = -(0.520, 0.805, 0.285)

     worth 16.9 mm of leg length per radian with the sole EXACTLY flat
     (verified: 0.5 rad -> -9.9 mm, 0.00 deg of tilt). Note the knee carries
     most of it; a minimum-norm least-squares solve zeroes the knee and yields
     only 2.8 mm/rad, which is the wrong answer.

The upright stand PD from task_balance_board.PD is kept as-is underneath: the
bare servos cannot hold HOME even on the floor (they fold at ~0.8 s).

    uv run python src/microduck_lab/tasks/balance_board/expert.py --grid
    uv run python src/microduck_lab/tasks/balance_board/expert.py --open 0.30 --kp 6 --kd 0.3
"""

import argparse
import itertools

import numpy as np


from microduck_lab.tasks.balance_board import board as tb
from microduck_lab.tasks.skating.ice_experts.harness import (  # noqa: E402
    HOME, L_HIP_ROLL, R_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE,
    R_HIP_PITCH, R_KNEE, R_ANKLE,
)

# Measured; see the module docstring. Positive coefficient = SHORTEN that leg.
SHORTEN_L = {L_HIP_PITCH: 0.521, L_KNEE: 0.805, L_ANKLE: 0.284}
SHORTEN_R = {R_HIP_PITCH: -0.520, R_KNEE: -0.805, R_ANKLE: -0.285}
MM_PER_RAD = 16.9

HIP_ROLL_LIMIT = 0.384


class Expert:
    """Open stance + differential leg length, over the stand PD."""

    name = "expert"

    def __init__(self, orientation="A", open_stance=0.30, kp=6.0, kd=0.3,
                 ks=0.0, kv=0.0, sign=1.0, u_max=0.8, stand=None, lean_kp=0.0, lean_kd=0.0):
        assert orientation == "A", "differential leg length acts about the ROLL axis"
        self.o = orientation
        self.open = open_stance
        self.kp, self.kd, self.ks, self.kv = kp, kd, ks, kv
        self.sign, self.u_max = sign, u_max
        self.lean_kp, self.lean_kd = lean_kp, lean_kd
        self.stand = dict(tb.STAND) if stand is None else dict(stand)
        self.prev = None

    def __call__(self, s, dt):
        if self.prev is None:
            self.prev = (s["plank_tilt"], s["trunk_tilt"], s["s"], s["trunk_other"])
        d_tilt = (s["plank_tilt"] - self.prev[0]) / dt
        d_trunk = (s["trunk_tilt"] - self.prev[1]) / dt
        d_s = (s["s"] - self.prev[2]) / dt
        d_other = (s["trunk_other"] - self.prev[3]) / dt
        self.prev = (s["plank_tilt"], s["trunk_tilt"], s["s"], s["trunk_other"])

        c = HOME.copy()

        # 1. Open the legs. Constant, and it also runs during the held settle so
        #    the duck is already wide when the board is released.
        c[L_HIP_ROLL] += self.open
        c[R_HIP_ROLL] -= self.open

        # 2. Stand PD, the same channels as tb.PD (roll -> both hip_rolls,
        #    pitch -> ankles), so the trunk stays up.
        kp_r, kd_r = self.stand["roll"]
        kp_p, kd_p = self.stand["pitch"]
        lean = kp_r * s["trunk_tilt"] + kd_r * d_trunk
        if s.get("released", True):
            # the PD's own channel: lean the body with both hip_rolls
            lean += self.lean_kp * s["plank_tilt"] + self.lean_kd * d_tilt
        c[L_HIP_ROLL] -= lean
        c[R_HIP_ROLL] -= lean
        v = kp_p * s["trunk_other"] + kd_p * d_other
        c[L_ANKLE] += v
        c[R_ANKLE] -= v

        # 3. Board feedback -> differential leg length. Zero while the board is
        #    still held, exactly as tb.PD does.
        u = 0.0
        if s.get("released", True):
            u = (self.kp * s["plank_tilt"] + self.kd * d_tilt
                 - self.ks * s["s"] - self.kv * d_s)
            u = float(np.clip(self.sign * u, -self.u_max, self.u_max))
            for j, w in SHORTEN_L.items():
                c[j] += u * w
            for j, w in SHORTEN_R.items():
                c[j] -= u * w

        c[L_HIP_ROLL] = float(np.clip(c[L_HIP_ROLL], -HIP_ROLL_LIMIT, HIP_ROLL_LIMIT))
        c[R_HIP_ROLL] = float(np.clip(c[R_HIP_ROLL], -HIP_ROLL_LIMIT, HIP_ROLL_LIMIT))
        return c, u


def one(open_stance, kp, kd, sign, seconds=10.0, push=0.0, verbose=False,
        u_max=0.8, lean_kp=0.0, lean_kd=0.0, **kw):
    ctrl = Expert(open_stance=open_stance, kp=kp, kd=kd, sign=sign, u_max=u_max,
                  lean_kp=lean_kp, lean_kd=lean_kd)
    return tb.run("A", ctrl, seconds=seconds, push=push, verbose=verbose, **kw)


def grid(seconds=10.0):
    best = (0.0, None)
    print(f"{'open':>5} {'sign':>5} {'kp':>5} {'kd':>5} {'held':>7} {'max tilt':>9}")
    for op, sg, kp, kd in itertools.product((0.0, 0.15, 0.30),
                                            (1.0, -1.0),
                                            (1.0, 4.0, 10.0),
                                            (0.0, 0.15, 0.4)):
        r = one(op, kp, kd, sg, seconds=seconds)
        held = r["held"]
        if held > best[0]:
            best = (held, (op, sg, kp, kd))
            print(f"{op:5.2f} {sg:+5.0f} {kp:5.1f} {kd:5.2f} {held:7.2f} "
                  f"{np.degrees(r['max_tilt']):9.1f}   <- best")
    print(f"\nbest {best[0]:.2f}s at open/sign/kp/kd = {best[1]}")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", action="store_true")
    p.add_argument("--open", dest="open_stance", type=float, default=0.30)
    p.add_argument("--kp", type=float, default=6.0)
    p.add_argument("--kd", type=float, default=0.3)
    p.add_argument("--sign", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--cam-distance", type=float, default=0.8)
    a = p.parse_args()
    if a.grid:
        grid(seconds=a.seconds)
        return
    r = one(a.open_stance, a.kp, a.kd, a.sign, seconds=a.seconds, push=a.push,
            verbose=True, video=a.video, azimuth=a.azimuth, cam_distance=a.cam_distance)
    print(f"HELD {r['held']:.2f}")


if __name__ == "__main__":
    main()
