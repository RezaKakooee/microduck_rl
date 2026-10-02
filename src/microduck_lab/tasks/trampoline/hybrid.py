"""Hybrid flip: the scripted pump bounces the duck up, then the RL flip policy takes over.

The user's idea (2026-10-02). Every RL attempt to start standing, bounce and then
flip in one policy failed (runs 3-7): from standing, doing nothing was the safe
choice. The RL flip policy (run 2) flips 97-99% of the time, but only from a
drop: upright, in the air, with energy. The scripted pump (pump.py, push law,
big-bed settings from flip.py) bounces the duck up from standing and keeps its
trunk upright and its soles level. So:

1. pump: the scripted pump runs from standing;
2. prepare: in the first flight whose predicted top (height + vz^2 / 2g) is
   above `switch_h` m (trunk above its standing height on the resting bed),
   the pump moves the legs to the standing pose on the way up. Measured: run 2
   flips from a 30 cm drop in the standing pose, but NOT from the same drop in
   the pump's bent-leg air pose (speed and the bed's motion did not matter);
3. hand over: at the top of that flight (vertical speed turns down) the RL
   policy takes over and keeps control;
4. the RL policy lands, rebounds and flips, as from a drop.

The RL policy runs from its exported ONNX (observation normalizer baked in),
with the 61-D observation built as in training: trunk gyro (body frame),
gravity in the body frame, joint positions minus the default pose, joint
speeds, its last action, and zero commands. At the hand-over its "last action"
is the pump's current target, so the action history does not jump.

This also checks the policy across simulators: trained in MuJoCo Warp, run
here in CPU MuJoCo with the BAM motors, on the same bed.

    uv run python -m microduck_lab.tasks.trampoline.hybrid --switch-h 0.15
    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.trampoline.hybrid --switch-h 0.15 --video
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.trampoline.flip import BED, PUMP
from microduck_lab.tasks.trampoline.pump import Pump, PumpParams
from microduck_lab.tasks.trampoline.world import ROBOT_HEIGHT, TrampolineWorld

RUN2_ONNX = str(paths.REPO / "logs/rsl_rl/microduck_trampoline_flip/2026-10-01_03-04-52_run2_beddr"
                / "2026-10-01_03-04-52_run2_beddr.onnx")
STAND_ROOT_Z = 0.125      # trunk above the soles, standing (as in the RL env)
COMMAND_DIMS = 3 + 4 + 6  # twist, head, body: zeros


class OnnxPolicy:
    def __init__(self, path, world):
        import onnx
        import onnxruntime as ort
        meta = {p.key: p.value for p in onnx.load(path).metadata_props}
        names = meta["joint_names"].split(",")
        if list(names) != list(world.names):
            raise ValueError(f"joint order differs: {names} vs {world.names}")
        self.default = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
        self.scale = float(meta.get("action_scale", 1.0))
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1      # a small MLP; many runs go in parallel
        self.sess = ort.InferenceSession(path, opt, providers=["CPUExecutionProvider"])
        self.last = np.zeros(len(names))
        self.w = world

    def observation(self):
        w, d = self.w, self.w.data
        gyro = d.qvel[w.root + 3:w.root + 6]                     # free joint: body frame
        R = d.xmat[w.trunk].reshape(3, 3)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        q = d.qpos[w.qidx] - self.default
        qd = d.qvel[w.vidx]
        return np.concatenate([gyro, gravity, q, qd, self.last, np.zeros(COMMAND_DIMS)]).astype(np.float32)

    def __call__(self):
        a = self.sess.run(None, {"obs": self.observation()[None]})[0][0].astype(float)
        self.last = a
        return self.default + self.scale * a


class Hybrid:
    def __init__(self, world, pump: PumpParams, onnx_path=RUN2_ONNX, switch_h=0.15, last_action="pump",
                 bounce_onnx=None, prepare="stand", blend_rate=1.0):
        self.w = world
        # The bouncer: the scripted pump, or an RL bounce policy (ONNX) when bounce_onnx is given.
        self.pump = OnnxPolicy(bounce_onnx, world) if bounce_onnx else Pump(world, pump)
        self.rl = OnnxPolicy(onnx_path, world)
        self.switch_h = switch_h
        self.last_action = last_action   # "pump": the pump's target as the RL's last action; "zero": as at an RL reset
        # "stand": standing pose on the way up, in the air; "stance": wait for the next landing and take
        # the standing pose on the bed (the feet hold the trunk), then hand over at the top of the next
        # flight; "none": keep the bouncer's pose.
        # "blend": as "stance", but reach the standing pose over 1 / blend_rate s of time on the bed.
        self.prepare, self.blend_rate = prepare, blend_rate
        self.mode = "pump"
        self.switched_at = None
        self.switch_height = None
        self.film = None

    def height(self):
        return float(self.w.data.xpos[self.w.trunk][2] - (self.w.frame.top + STAND_ROOT_Z))

    def __call__(self):
        w = self.w
        if self.mode == "pump":
            q = self.pump()
            feet, other = w.touching()
            air = not feet and not other
            vz = float(w.data.qvel[w.root + 2])
            if air and vz > 0 and self.height() + vz * vz / (2 * 9.81) >= self.switch_h:
                self.mode = {"stance": "await_land", "blend": "blend"}.get(self.prepare, "prepare")
                self.alpha = 0.0
                if self.film is not None and self.mode == "prepare":
                    self.film.tag = "scripted: standing pose"
            if self.mode in ("pump", "await_land"):
                return q
        if self.mode == "blend":
            # Move from the bouncer's targets to the standing pose slowly, and only while on the bed
            # (in the air, any pose change turns the trunk). Then as "stance".
            q = self.pump()
            feet, other = w.touching()
            if feet or other:
                self.alpha = min(1.0, self.alpha + self.blend_rate * w.control_dt)
            if self.alpha < 1.0:
                return (1 - self.alpha) * q + self.alpha * self.w.stand
            self.mode, self.risen = "prepare", False
            if self.film is not None:
                self.film.tag = "scripted: standing pose on the bed"
        if self.mode == "await_land":
            q = self.pump()
            feet, other = w.touching()
            if not (feet or other):
                return q
            self.mode, self.risen = "prepare", False
            if self.film is not None:
                self.film.tag = "scripted: standing pose on the bed"
        if self.mode == "prepare":
            q = self.pump()      # keeps its state current
            if self.prepare in ("stand", "stance", "blend"):
                q = self.w.stand.copy()
            elif self.prepare == "legs":     # legs to standing, neck and head (servos 5-8) as the bouncer has them
                q = np.concatenate([self.w.stand[:5], q[5:9], self.w.stand[9:]])
            feet, other = w.touching()
            vz = float(w.data.qvel[w.root + 2])
            if self.prepare in ("stance", "blend"):
                self.risen = self.risen or (not (feet or other) and vz > 0)
                if feet or other or not self.risen or vz > 0:
                    return q                # on the bed, or still rising: keep the standing pose
            elif feet or other:
                self.mode = "pump"          # landed before the hand-over: back to pumping
                return q
            if vz <= 0:
                self.mode, self.switched_at, self.switch_height = "rl", float(w.data.time), self.height()
                self.rl.last = ((q - self.rl.default) / self.rl.scale if self.last_action == "pump"
                                else np.zeros_like(self.rl.last))
                if self.film is not None:
                    self.film.tag = "RL flip policy"
                return self.rl()
            return q
        return self.rl()


def run(switch_h=0.15, seconds=14.0, pump=None, film=None, onnx_path=RUN2_ONNX, last_action="pump",
        bounce_onnx=None, prepare="stand", blend_rate=1.0, **world_kw):
    w = TrampolineWorld(**dict(BED, **world_kw))
    p = pump or PumpParams(**{k: getattr(PUMP, k) for k in PUMP.__dataclass_fields__})
    w.place(w.stand, drop_mm=p.drop)
    ctrl = Hybrid(w, p, onnx_path, switch_h, last_action, bounce_onnx, prepare, blend_rate)
    if film is not None:
        ctrl.film = film(w)
        ctrl.film.tag = "RL bounce policy" if bounce_onnx else "scripted pump"
        w.on_substep = ctrl.film

    flights, rot, air_t, fell_at, fall_reason = [], 0.0, 0.0, None, None
    landed_after_flip, apex = None, -1.0
    for _ in range(int(round(seconds / w.control_dt))):
        w.step(ctrl())
        d = w.data
        feet, other = w.touching()
        air = not feet and not other
        if air:
            rot += float(d.qvel[w.root + 4]) * w.control_dt      # pitch rate, nose down > 0
            air_t += w.control_dt
            apex = max(apex, ctrl.height())
        else:
            if air_t >= 0.08:
                flights.append(dict(t_land=round(float(d.time), 3), flight_s=round(air_t, 3),
                                    rotation_deg=round(math.degrees(rot), 1), apex_m=round(apex, 3),
                                    by=("rl" if ctrl.switched_at is not None and d.time - air_t >= ctrl.switched_at - 1e-6 else "pump"),
                                    landed_on_feet=bool(feet) and not other))
                if flights[-1]["rotation_deg"] >= 300 and landed_after_flip is None:
                    landed_after_flip = flights[-1]
            rot, air_t, apex = 0.0, 0.0, -1.0
        if other and fell_at is None:
            fell_at, fall_reason = float(d.time), "body contact"
        if w.tilt_deg() > 45 and not air and fell_at is None:
            fell_at, fall_reason = float(d.time), "tilt > 45 deg on the bed"
        if fell_at is not None:
            break
    flip = landed_after_flip
    out = dict(
        switch_h=switch_h, switched_at_s=ctrl.switched_at, switch_height_m=ctrl.switch_height,
        flip=flip is not None, flip_flight=flip,
        fell_at_s=fell_at, fall_reason=fall_reason,
        up_after_flip_s=(round((fell_at if fell_at else float(w.data.time)) - flip["t_land"], 2) if flip else None),
        bounces_before_switch=sum(f["by"] == "pump" for f in flights),
        max_pump_apex_m=max([f["apex_m"] for f in flights if f["by"] == "pump"], default=None),
        flights=flights,
    )
    return out, w, ctrl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--switch-h", type=float, default=0.15, help="hand over at the top of a flight above this (m)")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--onnx", default=RUN2_ONNX)
    ap.add_argument("--last-action", default="pump", choices=("pump", "zero"))
    ap.add_argument("--bounce-onnx", default=None, help="an RL bounce policy instead of the scripted pump")
    ap.add_argument("--prepare", default="stand", choices=("stand", "legs", "stance", "blend", "none"), help="pose before the hand-over")
    ap.add_argument("--blend-rate", type=float, default=1.0, help="prepare blend: 1 / seconds on the bed")
    ap.add_argument("--world", default="{}", help='JSON bed changes, e.g. {"sag_mm": 60}')
    ap.add_argument("--video", action="store_true", help="film it (needs MUJOCO_GL=egl)")
    ap.add_argument("--slow", type=float, nargs=2, default=None, help="slow-motion window (s); default around the hand-over")
    ap.add_argument("--name", default="hybrid")
    a = ap.parse_args()
    film = None
    if a.video:
        from microduck_lab.tasks.trampoline.video import BedFilm
        film = lambda w: BedFilm(w, slow_window=a.slow, slow_dt=0.01, camera="single")   # noqa: E731
    out, w, ctrl = run(a.switch_h, a.seconds, film=film, onnx_path=a.onnx, last_action=a.last_action,
                       bounce_onnx=a.bounce_onnx, prepare=a.prepare, blend_rate=a.blend_rate, **json.loads(a.world))
    flights = out.pop("flights")
    print(json.dumps(out))
    print("flights (land s, flight s, rotation deg, apex m, by, on feet):",
          [(f["t_land"], f["flight_s"], f["rotation_deg"], f["apex_m"], f["by"], f["landed_on_feet"]) for f in flights])
    if a.video:
        print("videos ->", *ctrl.film.write(a.name))


if __name__ == "__main__":
    main()
