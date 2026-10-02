"""Flip: bounce up, turn a full circle in the air, land on the feet, keep bouncing.

Phases:

1. pump: the pumping controller (pump.py, push law on the big bed) until
   `flip_at` seconds, so the bounces are high.
2. flip stance: on the next landing. With `release_bottom` the upright PDs
   keep working until the bed reaches its lowest point; then they stop and
   the feet go `lean` mm ahead of the CoM. Without the PDs the trunk tips
   forward at the soft hips under the bed's push: a front flip is this
   robot's natural direction (measured: 6-8 rad/s nose down with lean 0).
   Feet behind the CoM (lean < 0) add to it. For a backflip, direction = -1
   and lean > 0. Releasing at the landing instead turned the duck 72 deg on
   the bed, so it left the bed tilted and flew 600 mm forward.
3. flip air: after take-off the duck tucks (knees, hips, head in). That
   halves its pitch inertia (49.6 -> 24.8 x 1e-4 kg m^2, measured), so it
   spins about twice as fast. At `open_deg` of rotation it opens again, and
   near upright it levels the soles for the landing.
4. recover: the pumping controller again, with a smaller leg swing.

The rotation is the trunk's pitch, unwrapped, counted from the start of the
flip stance. A real robot would integrate its gyro.

    uv run python -m microduck_lab.tasks.trampoline.flip
    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.trampoline.flip --video
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from microduck_lab.tasks.jump.world import tilt_angles
from microduck_lab.tasks.trampoline.pump import Pump, PumpParams
from microduck_lab.tasks.trampoline.world import COL, ROBOT_HEIGHT, TrampolineWorld

# The big bed of the 3-4x height attempt (a light, bouncy, soft, deep bed).
BED = dict(frame="wide", sag_mm=80.0, rebound=0.95, rebound_drop_mm=500.0, bed_mass=0.01)
# The pumping setting that bounced about 1.3 robot heights on it (clip 13).
# Pumping with position feedback: 25 s without a fall at 1.5 robot heights,
# drift under 140 mm (the first big-bed setting that stays up).
PUMP = PumpParams(push=1.0, mid=20.0, amp=20.0, push_travel=50.0, hip_kp=3.0, hip_kd=0.4,
                  place_gain=0.1, place_i=0.0, place_x=0.05, air_hip=1.0, target_ms=float("nan"))


@dataclass
class FlipParams:
    pump: PumpParams = field(default_factory=lambda: PumpParams(**asdict(PUMP)))
    flip_at: float = 10.0       # s: the first landing after this starts the flip
    # A gymnast takes off a little behind the centre when the trick travels
    # forward. From pre_at the pumping aims at x = pre_x (m) instead of 0.
    pre_at: float = 5.0
    pre_x: float = -0.2
    direction: float = 1.0      # +1 front flip (nose down), -1 backflip
    lean: float = 0.0           # mm: feet ahead of the CoM in the flip stance
    release_bottom: float = 0.0  # 1: keep the PDs on until the bed's lowest point
    release_delay: float = 0.05  # s after the landing (when release_bottom = 0)
    tuck_deg: float = 30.0      # start the tuck after this much rotation
    open_deg: float = 290.0     # open again after this much rotation (open_mode 0)
    # open_mode 1: open when the rotation so far plus (spin now x open_ratio x
    # time left) reaches target_deg. Time left comes from the take-off speed
    # (flight = 2 vz / g). open_ratio = spin after opening / spin in the tuck
    # (the inverse of the inertia ratio, 24.8 / 49.6).
    open_mode: float = 1.0
    open_ratio: float = 0.5
    target_deg: float = 360.0
    level_deg: float = 60.0     # level the soles when this close to upright
    tuck_depth: float = 40.0    # mm: crouch of the tuck
    tuck_hip: float = 0.8       # rad: extra hip flexion in the tuck
    tuck_neck: float = 0.8      # rad: neck and head in the tuck
    tuck_head: float = 0.6
    recover_amp: float = 10.0   # mm: leg swing after the landing
    recover_brake: float = 0.0  # 1: after the landing, take energy out (pump.brake)
    recover_place_x: float = 0.05   # position feedback after the landing
    recover_place_x_max: float = 0.03   # and its limit (m)
    # The pose after opening: the pumping crouch at open_depth mm (0..40, with
    # the pump's foot placement, which matters: without it the best landing
    # stood 0.23 s instead of 1.56 s), neck and head turned by open_neck (rad,
    # < 0 = away from the tuck). A longer, open body turns slower.
    open_depth: float = 40.0
    open_neck: float = 0.0
    # The flip's push throws the duck forward (it turns about its feet on the
    # bed). To land where it took off, the bounce before the flip lands with
    # the feet pre_lean mm ahead of the CoM, so it arrives moving backward.
    pre_lean: float = 0.0


def unwrapped_pitch(w, prev):
    """Trunk pitch (rad, nose down > 0), continued from `prev` across +/-pi."""
    R = w.data.xmat[w.trunk].reshape(3, 3)
    a = float(np.arctan2(-R[2, 0], R[0, 0]))
    if prev is None:
        return a
    return prev + (a - prev + np.pi) % (2 * np.pi) - np.pi


class Flip:
    def __init__(self, world, p: FlipParams):
        self.w, self.p = world, p
        self.pump = Pump(world, p.pump)
        tuck = world.crouch(p.tuck_depth / 1000).copy()
        tuck[world.act["left_hip_pitch"]] += p.tuck_hip
        tuck[world.act["right_hip_pitch"]] -= p.tuck_hip
        tuck[world.act["neck_pitch"]] += p.tuck_neck
        tuck[world.act["head_pitch"]] += p.tuck_head
        self.tuck = tuck
        self.phase = "pump"
        self.pitch = None
        self.start = None           # pitch at the start of the flip stance
        self.was_air = False
        self.events = {}
        self.stance_t = None
        self.takeoff_t = None
        self.flight_pred = None
        self.opened = False

    def rotation_deg(self):
        """Rotation since the flip started, in the flip's direction (deg, >= 0 when going the right way)."""
        if self.start is None:
            return 0.0
        return float(np.degrees(np.sign(self.p.direction) * (self.pitch - self.start)))

    def pre_window(self):
        """Time before flip_at in which the pre-lean bounce happens (about one bounce)."""
        return 0.7

    def release(self):
        """Stop the upright PDs and lean: the flip's turning starts here."""
        self.start = self.pitch
        self.events["release"] = round(self.w.data.time, 3)
        self.pump.extra_ahead, self.pump.balance = self.p.lean / 1000, False

    def __call__(self):
        w, p = self.w, self.p
        self.pitch = unwrapped_pitch(w, self.pitch)
        feet, other = w.touching()
        air = not feet and not other
        landed, took_off = self.was_air and not air, air and not self.was_air
        self.was_air = air
        t = w.data.time

        if self.phase == "pump":
            if t >= p.pre_at:
                self.pump.target_x = p.pre_x
            if p.pre_lean and t >= p.flip_at - self.pre_window():
                self.pump.extra_ahead = p.pre_lean / 1000
            q = self.pump()
            if landed and t >= p.flip_at:
                self.phase, self.stance_t = "flip_stance", t
                self.events["flip_stance"] = round(t, 3)
            return q

        if self.phase == "flip_stance":
            if self.start is None:
                if p.release_bottom and w.data.qvel[w.bed_v] > 0:
                    self.release()
                elif not p.release_bottom and t - self.stance_t >= p.release_delay - 1e-9:
                    self.release()
            if took_off:
                if self.start is None:
                    self.release()
                self.phase, self.takeoff_t = "flip_air", t
                vz = float(w.data.subtree_linvel[w.trunk][2])
                self.flight_pred = max(2 * vz / 9.81, 0.0)
                self.events["takeoff_vz"] = round(vz, 2)
                self.events["takeoff_vx"] = round(float(w.data.subtree_linvel[w.trunk][0]), 3)
                self.events["takeoff_x_mm"] = round(float(w.data.subtree_com[w.trunk][0]) * 1000, 1)
                self.events["predicted_flight_ms"] = round(self.flight_pred * 1000, 1)
                self.events["takeoff"] = round(t, 3)
                self.events["takeoff_rotation_deg"] = round(self.rotation_deg(), 1)
                self.events["takeoff_spin_rad_s"] = round(float(np.sign(p.direction) * w.data.qvel[w.root + 4]), 2)
            else:
                return self.pump()

        if self.phase == "flip_air":
            self.pump()     # keeps its state (speeds, flight timer) up to date
            if landed:
                self.phase = "recover"
                self.events["landing"] = round(t, 3)
                self.events["landing_rotation_deg"] = round(self.rotation_deg(), 1)
                self.events["landing_x_mm"] = round(float(w.data.subtree_com[w.trunk][0]) * 1000, 1)
                self.events["landing_spin_rad_s"] = round(float(np.sign(p.direction) * w.data.qvel[w.root + 4]), 2)
                self.events["landing_vx"] = round(float(w.data.subtree_linvel[w.trunk][0]), 3)
                self.events["landing_on_bed"] = bool(feet)
                self.pump.extra_ahead, self.pump.balance = 0.0, True
                self.pump.target_x = 0.0
                self.pump.p.amp = p.recover_amp
                self.pump.p.brake = p.recover_brake
                self.pump.p.place_x = p.recover_place_x
                self.pump.p.place_x_max = p.recover_place_x_max
                self.pump.progress = 0.0
                return self.pump()
            rot = self.rotation_deg()
            if not self.opened:
                if p.open_mode:
                    spin = float(np.degrees(np.sign(p.direction) * w.data.qvel[w.root + 4]))
                    left = max(self.flight_pred - (t - self.takeoff_t), 0.0)
                    self.opened = rot >= p.tuck_deg and rot + spin * p.open_ratio * left >= p.target_deg
                else:
                    self.opened = rot >= p.open_deg
                if self.opened:
                    self.events["open_at_deg"] = round(rot, 1)
                    self.events["open_at_s_after_takeoff"] = round(t - self.takeoff_t, 3)
            if rot < p.tuck_deg:
                return self.pump.pose(self.pump.p.mid + self.pump.p.amp)
            if not self.opened:
                return self.tuck
            q = self.pump.pose(p.open_depth).copy()
            q[w.act["neck_pitch"]] += p.open_neck
            q[w.act["head_pitch"]] += p.open_neck
            if 360.0 - rot < p.level_deg:
                q = self.pump.level(q)
            return q

        return self.pump()      # recover


def run(p: FlipParams, seconds=14.0, film=None, **world_kw):
    kw = dict(BED, **world_kw)
    w = TrampolineWorld(**kw)
    w.place(w.stand, drop_mm=p.pump.drop)
    ctrl = Flip(w, p)
    if film is not None:
        w.on_substep = film(w)
    rot, xy = [], []
    for _ in range(int(round(seconds / w.control_dt))):
        w.step(ctrl())
        rot.append((w.data.time, ctrl.rotation_deg()))
        xy.append(w.data.subtree_com[w.trunk][:2].copy())
    a = np.array(w.log)
    ev = dict(ctrl.events)
    out = dict(events=ev, params=asdict(p))
    # Failure: any body part on the bed or floor at any time; after the flip's
    # landing also a tilt over 45 deg. In the flip itself the duck is upside down.
    body = a[:, COL["n_other"]] > 0
    tilt = (a[:, COL["tilt"]] > 45) & (a[:, 0] >= ev.get("landing", np.inf))
    bad = body | tilt
    out["fell"] = bool(bad.any())
    out["fell_at"] = round(float(a[np.argmax(bad), 0]), 3) if bad.any() else None
    out["fall_reason"] = None if not bad.any() else ("body contact" if body[np.argmax(bad)] else "tilt > 45 deg")
    if "takeoff" in ev and "landing" in ev:
        seg = a[(a[:, 0] >= ev["takeoff"]) & (a[:, 0] <= ev["landing"])]
        ev["flight_ms"] = round((ev["landing"] - ev["takeoff"]) * 1000, 1)
        ev["soles_max_m"] = round(float((seg[:, COL["sole_gap"]] + seg[:, COL["bed_z"]]).max()), 3)
        ev["robot_heights"] = round(ev["soles_max_m"] / ROBOT_HEIGHT, 2)
        after = a[(a[:, 0] >= ev["landing"]) & (a[:, 0] <= ev["landing"] + 0.3)]
        ev["tilt_0.3s_after_landing_deg"] = round(float(after[:, COL["tilt"]].max()), 1) if len(after) else None
        end = out["fell_at"] if out["fell"] else a[-1, 0]
        ev["upright_after_landing_s"] = round(float(end - ev["landing"]), 2)
    out["max_drift_mm"] = round(float(np.linalg.norm(np.array(xy), axis=1).max() * 1000), 1)
    return out, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(FlipParams()).items():
        if k != "pump":
            ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    ap.add_argument("--pump", default="{}", help="JSON changes to the pumping params")
    ap.add_argument("--world", default="{}", help="JSON changes to the bed")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--video", action="store_true", help="film the run (needs MUJOCO_GL=egl)")
    ap.add_argument("--slow", type=float, nargs=2, default=None, help="slow-motion window (s)")
    ap.add_argument("--name", default="flip")
    a = ap.parse_args()
    pump = PumpParams(**dict(asdict(PUMP), **json.loads(a.pump)))
    p = FlipParams(pump=pump, **{k: getattr(a, k) for k in asdict(FlipParams()) if k != "pump"})
    film = None
    if a.video:
        from microduck_lab.tasks.trampoline.video import BedFilm
        film = lambda w: BedFilm(w, slow_window=a.slow)   # noqa: E731
    out, w = run(p, seconds=a.seconds, film=film, **json.loads(a.world))
    out.pop("params")
    print(json.dumps(out))
    if a.video:
        print("videos ->", *w.on_substep.write(a.name))


if __name__ == "__main__":
    main()
