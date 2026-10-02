"""Scripted pumping: the duck makes its trampoline bounces grow.

Energy goes in only when the legs push while the bed pushes back, so the
legs follow the phase of the BED:

    leg length = mid + amp * sin(bed phase + lead)

The bed phase is 0 at the bottom of the bed's travel, +90 deg moving up
through its rest point, 180 deg at the top. With lead = 0 the legs get
longer fastest at the bottom, where the bed force is highest. The servos lag
under load, so `lead` (deg) starts the motion earlier. In the air the legs
bend (mid + amp shorter): that costs nothing, as there is no load.

Four more parts keep the duck on its feet and on the bed. Each fixed a
failure seen in the sim (videos/trampoline/07, 08):

1. In the air, the soles are held level (`level`). The upright PD would
   tilt them by 5x the trunk tilt, and the duck landed on its toes or heels.
2. On the bed, the hips hold the trunk upright (`hip_kp`, `hip_kd`). The
   ankle PD alone let the trunk tip forward about 7 deg in every stance.
3. Foot placement (`place_*`). The CoM sits 6-7 mm behind the feet, so the
   duck hopped backward about 10 mm per bounce, off the bed edge. A slow
   correction after each landing works. A fast one (Raibert style) did not:
   the hips lag about 100 ms.
4. A flight-time limit (`target_ms`). Without it a bouncier bed pumps up
   until the duck falls.

The first version switched on the CoM vertical speed instead of the bed
phase. It failed: a 30 mm crouch takes the servos about 200 ms, so the CoM
speed followed the duck's own squat (about 2 Hz), not the bed (3.5 Hz).

Inputs: bed height and speed while on the bed, "in the air", flight time,
trunk tilt and pitch rate, and the forward speed in flight. A real robot
could estimate the bed from its feet (trunk height from the IMU minus the
leg length from the joint encoders), and the rest from the IMU (in the air
the accelerometer reads 0 g).

The defaults are the best setting of search.py (2026-09-30).

    uv run python -m microduck_lab.tasks.trampoline.pump --seconds 20
    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.trampoline.pump --video
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from microduck_lab.tasks.jump.world import Upright, tilt_angles
from microduck_lab.tasks.trampoline.world import COL, ROBOT_HEIGHT, TrampolineWorld, bounce_report


@dataclass
class PumpParams:
    mid: float = 10.0       # mean crouch: hips this many mm lower than standing
    amp: float = 20.0       # leg length swing, +/- mm
    lead: float = 150.0     # phase lead, deg
    drop: float = 20.0      # start: dropped from this height (mm) onto the bed
    start_s: float = 0.5    # hold the stand pose this long after the drop
    level: float = 1.0      # 1: soles level in the air; 0: upright PD in the air (the old way)
    # Foot placement (x only). Feet this far ahead of the CoM (mm):
    #     place_offset + integral + place_gain * (last flight's forward speed)
    # At each landing the integral adds place_i * (that flight's forward speed).
    # place_gain = nan turns placement off (feet where the crouch puts them,
    # about 6-7 mm ahead of the CoM, and the duck drifts backward).
    place_offset: float = 3.0
    place_gain: float = 0.0     # s
    place_i: float = 0.05       # s per bounce
    # Position feedback (sim only: a real robot would need a camera or similar
    # to know where it is on the bed). Feet further ahead by place_x x (CoM x -
    # target_x); the target is Pump.target_x (m), 0 = bed centre.
    place_x: float = 0.0
    place_x_max: float = 0.03   # m: limit on that term. Far from the centre, 29 mm pushed a duck over backward
    # Trunk pitch held with the hips while on the bed (rad per rad, rad per rad/s).
    # The upright PD only uses the ankles: that turns the whole duck about its
    # feet, but under the bounce load the trunk tips forward at the soft hips.
    hip_kp: float = 3.0
    hip_kd: float = 0.05
    # Bounce height limit. After each flight the leg swing is scaled by
    #     clip(1 - (flight - target_ms) / REG_BAND_MS, 0, 1)
    # so it pumps fully below the target and not at all 40 ms above it.
    # Without it a bouncier bed pumps up until the duck falls. nan = off.
    target_ms: float = 200.0
    # Push law (push = 1) instead of the phase law, for big bounces on a soft,
    # deep bed. The bottom of such a bounce loads the legs with 5-6x body
    # weight, more than bent legs hold (the phase law folded there). So: bend
    # in the air (mid + amp), straighten to mid - amp while the bed goes down
    # (over push_travel mm of bed travel, looking push_lead s ahead for the
    # servo lag), and stay straight until take-off. The legs work against the
    # rising bed force, and are strongest at the bottom.
    # In the air, how the soles are kept level when the trunk pitches:
    #   0 = ankles only (the legs turn with the trunk, so a trunk tipped back
    #       swings the feet forward: 24 mm at 15 deg in a long flight);
    #   1 = hips only (the legs stay vertical in the world, soles level too).
    # Between and above 1: hips by air_hip x pitch, ankles do the rest.
    air_hip: float = 0.0
    push: float = 0.0
    push_travel: float = 100.0
    push_lead: float = 0.05
    # Brake (push = 1 and brake = 1): the push law backwards, to kill the
    # bounce ("checking" the bed). Straight legs in the air, and they bend
    # while the bed pushes up, so the legs take energy out.
    brake: float = 0.0


REG_BAND_MS = 40.0


class Pump:
    def __init__(self, world, p: PumpParams):
        self.w, self.p = world, p
        # Crouch poses on a 0.5 mm grid, interpolated each step. For each,
        # also where the feet are in x relative to the CoM (m), and how far
        # they move back per rad of hip swing (m/rad), trunk upright.
        self.depths = np.arange(p.mid - p.amp, p.mid + p.amp + 0.25, 0.5)
        self.poses = np.array([world.crouch(d / 1000) for d in self.depths])
        base = np.array([self._feet_ahead(q, 0.0) for q in self.poses])
        self.feet_ahead = base
        self.feet_per_rad = (base - np.array([self._feet_ahead(q, 0.05) for q in self.poses])) / 0.05
        self.up = Upright(world)
        self.omega = 2 * np.pi * world.bed_hz
        self.rest = -world.sag_mm / 1000
        self.vx = 0.0
        self.integral = 0.0
        self.was_air = False
        self.takeoff_t = 0.0
        self.scale = 1.0
        self.progress = 0.0
        # Hooks for a caller that takes over (flip.py): extra feet-ahead
        # offset (m), and whether the upright PDs run on the bed.
        self.extra_ahead = 0.0
        self.balance = True
        self.target_x = 0.0

    def _feet_ahead(self, q, delta):
        """Mean foot-box x minus CoM x (m) in pose q with hip swing delta, trunk upright."""
        import mujoco
        w = self.w
        d = mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        d.qpos[w.root:w.root + 7] = [0, 0, 0.3, 1, 0, 0, 0]
        d.qpos[w.qidx] = self.swing(q, delta)
        mujoco.mj_kinematics(w.model, d)
        mujoco.mj_comPos(w.model, d)
        return float(np.mean([d.geom_xpos[g][0] for g in w.foot_boxes]) - d.subtree_com[w.trunk][0])

    def swing(self, q, delta):
        """Swing both legs by `delta` rad at the hips; the ankles turn back so the soles keep their angle."""
        h, k, a = self.w.left_leg(q)
        return self.w.leg_pose(h + delta, k, a - delta, base=q)

    def pose(self, depth_mm):
        """Crouch pose at this depth, with the feet placed ahead of the CoM as asked."""
        q = np.array([np.interp(depth_mm, self.depths, self.poses[:, j]) for j in range(self.poses.shape[1])])
        if np.isnan(self.p.place_gain):
            return q
        want = self.p.place_offset / 1000 + self.integral + self.p.place_gain * self.vx + self.extra_ahead
        if self.p.place_x:
            want += float(np.clip(self.p.place_x * (float(self.w.data.subtree_com[self.w.trunk][0]) - self.target_x),
                                  -self.p.place_x_max, self.p.place_x_max))
        ahead = np.interp(depth_mm, self.depths, self.feet_ahead)
        per_rad = np.interp(depth_mm, self.depths, self.feet_per_rad)
        return self.swing(q, float(np.clip((ahead - want) / per_rad, -0.4, 0.4)))

    def phase(self):
        """Bed phase (rad): 0 at the bottom, pi/2 going up through rest, pi at the top."""
        d = self.w.data
        x = d.qpos[self.w.bed_q] - self.rest
        v = d.qvel[self.w.bed_v]
        return float(np.arctan2(v / self.omega, -x))

    def level(self, q):
        """Turn the ankles and hip rolls so the soles are level in the world.

        For the air. The upright PD would add 5x the trunk tilt at the
        ankles; in the air that cannot turn the trunk, it only tilts the
        soles (4 deg of trunk pitch gave a 23 deg sole at landing). The
        signs are the upright PD's with gain -1 (checked by kinematics).
        """
        w = self.w
        r, pitch = tilt_angles(w.data.qpos[w.root + 3:w.root + 7])
        k = self.p.air_hip
        q = q.copy()
        q[w.act["left_hip_pitch"]] -= k * pitch
        q[w.act["right_hip_pitch"]] += k * pitch
        q[w.act["left_ankle"]] -= (1 - k) * pitch
        q[w.act["right_ankle"]] += (1 - k) * pitch
        q[w.act["left_hip_roll"]] += r
        q[w.act["right_hip_roll"]] += r
        return q

    def __call__(self):
        w, p = self.w, self.p
        if w.data.time < p.start_s:
            return self.up(w.stand)
        feet, other = w.touching()
        air = not feet and not other
        if air and not self.was_air:
            self.takeoff_t = w.data.time
        if self.was_air and not air:
            # Landing: nudge the feet toward the way the duck was drifting.
            self.integral = float(np.clip(self.integral + p.place_i * self.vx, -0.01, 0.01))
            flight_ms = (w.data.time - self.takeoff_t) * 1000
            if not np.isnan(p.target_ms) and flight_ms > 10:
                self.scale = float(np.clip(1 - (flight_ms - p.target_ms) / REG_BAND_MS, 0.0, 1.0))
        self.was_air = air
        if air:
            # Forward speed is constant in flight: that is the one to correct.
            self.vx = float(w.data.subtree_linvel[w.trunk][0])
            self.progress = 0.0
            q = self.pose(p.mid + (-1 if p.brake else 1) * self.scale * p.amp)
            q_up = self.up(q)   # also keeps the PD's last tilt current for the landing
            return self.level(q) if p.level else q_up
        if p.push and p.brake:
            # Bend over the bed's way up: from the bottom to its rest level.
            v = float(w.data.qvel[w.bed_v])
            self.bottom = min(getattr(self, "bottom", 0.0), float(w.data.qpos[w.bed_q])) if v <= 0 or self.progress == 0 else self.bottom
            if v > 0 and self.bottom < 0:
                risen = float(w.data.qpos[w.bed_q]) - self.bottom
                self.progress = max(self.progress, float(np.clip(risen / -self.bottom, 0.0, 1.0)))
            depth = p.mid - self.scale * p.amp * (1 - 2 * self.progress)
        elif p.push:
            depth_below_rest = -float(w.data.qpos[w.bed_q])
            v = float(w.data.qvel[w.bed_v])
            if v < 0:
                ahead = depth_below_rest - v * p.push_lead
                self.progress = max(self.progress, float(np.clip(ahead / (p.push_travel / 1000), 0.0, 1.0)))
            depth = p.mid + self.scale * p.amp * (1 - 2 * self.progress)
        else:
            depth = p.mid - self.scale * p.amp * np.sin(self.phase() + np.radians(p.lead))
        if not self.balance:
            self.up(self.pose(depth))   # keeps the PD's last tilt current
            return self.pose(depth)
        return self.hips(self.up(self.pose(depth)))

    def hips(self, q):
        """Hip pitch feedback on trunk pitch. A higher left hip target (right mirrored)
        tips the trunk nose-up when the feet are on the bed."""
        w, p = self.w, self.p
        if not (p.hip_kp or p.hip_kd):
            return q
        _, pitch = tilt_angles(w.data.qpos[w.root + 3:w.root + 7])
        rate = float(w.data.qvel[w.root + 4])   # gyro y (body frame): pitch rate, nose down > 0
        u = p.hip_kp * pitch + p.hip_kd * rate
        q = q.copy()
        q[w.act["left_hip_pitch"]] += u
        q[w.act["right_hip_pitch"]] -= u
        return q


def run(p: PumpParams, seconds=8.0, film=None, tail_s=3.0, **world_kw):
    w = TrampolineWorld(**world_kw)
    w.place(w.stand, drop_mm=p.drop)
    ctrl = Pump(w, p)
    if film is not None:
        w.on_substep = film(w)
    xy = []
    for _ in range(int(round(seconds / w.control_dt))):
        w.step(ctrl())
        xy.append(w.data.subtree_com[w.trunk][:2].copy())
    r = bounce_report(w.log, t0=p.start_s)
    # How far the CoM wanders from the bed centre before any fall. The bed
    # box ends 130 mm from the centre.
    alive = np.array(xy)[:max(1, int((p.start_s + r["survived_s"]) / w.control_dt))]
    r["max_drift_mm"] = round(float(np.linalg.norm(alive, axis=1).max() * 1000), 1)
    a = np.array(w.log)
    r["max_torque_Nm"] = round(float(a[:, COL["max_torque"]].max()), 3)
    # The bed hits its hard stop (like a real bed touching the floor).
    r["bottomed_out"] = int((a[:, COL["bed_z"]] <= w.frame.range[0] + 0.001).sum())
    # How high the soles get above the bed's rest level, per flight (what the eye sees).
    for f in r["flights"]:
        seg = a[(a[:, 0] >= f["t"]) & (a[:, 0] <= f["t"] + f["flight_ms"] / 1000)]
        f["sole_over_rest_mm"] = round(float((seg[:, COL["sole_gap"]] + seg[:, COL["bed_z"]]).max() * 1000), 1)
        f["robot_heights"] = round(f["sole_over_rest_mm"] / 1000 / ROBOT_HEIGHT, 2)
    # Steady state: the flights in the last `tail_s` seconds.
    last = [f for f in r["flights"] if f["t"] >= seconds - tail_s]
    r["tail"] = dict(seconds=tail_s,
        bounces=len(last),
        flight_ms=round(float(np.mean([f["flight_ms"] for f in last])), 1) if last else 0.0,
        com_rise_mm=round(float(np.mean([f["com_rise_mm"] for f in last])), 1) if last else 0.0,
        sole_over_rest_mm=round(float(np.mean([f["sole_over_rest_mm"] for f in last])), 1) if last else 0.0,
        robot_heights=round(float(np.mean([f["robot_heights"] for f in last])), 2) if last else 0.0,
    )
    r["params"] = asdict(p)
    return r, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(PumpParams()).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--video", action="store_true", help="film the run (needs MUJOCO_GL=egl)")
    ap.add_argument("--slow", type=float, nargs=2, default=None, help="slow-motion window (s)")
    ap.add_argument("--name", default=None, help="clip name (default from the parameters)")
    ap.add_argument("--world", default="{}", help='JSON world settings, e.g. {"frame": "big", "sag_mm": 80}')
    a = ap.parse_args()
    p = PumpParams(**{k: getattr(a, k) for k in asdict(PumpParams())})

    film = None
    if a.video:
        from microduck_lab.tasks.trampoline.video import BedFilm
        film = lambda w: BedFilm(w, slow_window=a.slow)   # noqa: E731
    r, w = run(p, seconds=a.seconds, film=film, **json.loads(a.world))
    flights = r.pop("flights")
    print(json.dumps(r))
    print("flights (t, ms, CoM rise mm, soles over bed rest mm):",
          [(f["t"], f["flight_ms"], f["com_rise_mm"], f["sole_over_rest_mm"]) for f in flights])
    if a.video:
        name = a.name or f"pump_m{p.mid:g}_a{p.amp:g}_l{p.lead:g}"
        print("videos ->", *w.on_substep.write(name))


if __name__ == "__main__":
    main()
