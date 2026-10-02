"""Hang upside down by both leg hooks and inch sideways along the rope.

Physics only: the robot starts still, already on the rope, and after that
only motor targets move it. No welds, no support forces, no rope state in
the controller. Each filmed trial gets a new number and is never overwritten.

    # CPU, no video (quick check):
    python -m microduck_lab.tasks.rope_traverse.traverse --cycles 2 --no-film
    # film on a GPU node:
    sbatch -M cluster local_storage/hb_dev/scripted_policy/rope_traverse/video.sbatch \
        -m microduck_lab.tasks.rope_traverse.traverse --cycles 12
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import zipfile

import mujoco
import numpy as np

from microduck_lab.tasks.rope_traverse.scene import Scene, build
from microduck_lab.tasks.rope_traverse.arch import KP, Loads, hang_pose, place, rope_arc
from microduck_lab.tasks.rope_traverse.inchworm import Gait, Inchworm

ROOT = Path(__file__).resolve().parent
OUTPUT = Path('videos/rope_traverse')
SETTLE_S = .3            # hold the start pose before the gait begins
HOLD_S = 2.              # a hold passes after this long on the legs alone
TRAVEL_M = .15           # a traverse passes after this much travel along the rope


def reserve():
    """Next free rope_try number (shared with the earlier mouth-grip trials)."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for n in range(1, 100000):
        p = OUTPUT / f'rope_try{n}'
        if any(p.with_suffix(ext).exists() for ext in ('.mp4', '.json', '.npz')):
            continue
        try:
            with p.with_suffix('.lock').open('x'):
                pass
            return p
        except FileExistsError:
            continue
    raise RuntimeError('No unused trial number')


class Trial:
    """One run: build, place, settle, then play the gait."""

    def __init__(self, gait=Gait(), cycles=1, direction=1, scene=Scene(), kp=KP, x=0.):
        self.w = w = build(scene)
        self.duck = k = w.ducks['she']
        k.motor.model.actuator.kp = kp
        self.q0 = hang_pose()
        self.gap = place(w, self.q0, x=x)
        self.gait = Inchworm(gait, cycles, direction)
        self.loads = Loads(w)
        self.weight = k.mass() * 9.81
        self.rows = []
        self.t_gait = None

    def start(self):
        self.w.start()
        while self.w.t < SETTLE_S - 1e-9:
            self.duck.hold(self.q0)
            self.w.step()
            self.record('settle')
        self.t_gait = self.w.t
        self.arc0 = rope_arc(self.w, self.duck.com())

    def tick(self):
        t = self.w.t - self.t_gait
        self.duck.hold(self.gait(t))
        self.w.step()
        self.record(self.gait.phase(t))

    def record(self, phase):
        w, k = self.w, self.duck
        mujoco.mj_forward(w.model, w.data)
        L = self.loads()
        self.rows.append(dict(t=w.t, phase=phase, com=k.com().tolist(), arc=rope_arc(w, k.com()),
                              left=L['left'], right=L['right'], head=L['head'], other=L['other'],
                              floor=L['floor'], post=L['post'], low=float(k.points(k.solid)[:, 2].min()),
                              torque=np.abs(k.torque()).tolist()))

    def run(self, on_frame=None):
        self.start()
        while self.w.t - self.t_gait < self.gait.duration - 1e-9:
            self.tick()
            if on_frame: on_frame(self)
        return judge(self.rows, self.weight, self.t_gait, self.gait.direction)


def judge(rows, weight, t_gait, direction=1):
    """Pass/fail from the recorded trace.

    Hold: 2 s in a row carried by the two leg hooks alone (floor and posts
    < 0.5 N, no other body part on the rope, legs carry at least half the weight).
    Traverse: centre of mass moves 15 cm along the rope, the hold rule is
    never broken after the start, and both hooks carry load at the end."""
    t = np.array([r['t'] for r in rows])
    legs = np.array([r['left'] + r['right'] for r in rows])
    bad = np.array([r['floor'] > .5 or r['other'] > .5 or r['head'] > .5 or r.get('post', 0.) > .5 for r in rows])
    held = ~bad & (legs > .5 * weight)
    longest = cur = 0
    for h in held:
        cur = cur + 1 if h else 0; longest = max(longest, cur)
    gait = t >= t_gait - 1e-9
    arc = np.array([r['arc'] for r in rows])
    travel = direction * float(arc[-1] - arc[gait][0]) if gait.any() else 0.
    last = rows[-1]
    end_both = last['left'] > .5 and last['right'] > .5
    peak_torque = float(np.max([r['torque'] for r in rows]))
    return dict(held_s=round(longest * .02, 2), hold_pass=bool(longest * .02 >= HOLD_S),
                travel_m=round(travel, 4), speed_mm_s=round(1000 * travel / max(t[-1] - t_gait, 1e-9), 3),
                broken_hold_ticks=int((~held[gait]).sum()), floor_contact=bool(any(r['floor'] > .5 for r in rows)),
                other_rope_contact=bool(any(r['other'] > .5 or r['head'] > .5 for r in rows)),
                post_contact=bool(any(r.get('post', 0.) > .5 for r in rows)),
                end_com_x=round(rows[-1]['com'][0], 4) if 'com' in rows[-1] else None,
                both_hooks_at_end=bool(end_both), peak_joint_torque_nm=round(peak_torque, 3),
                min_leg_support=round(float(legs[gait].min() / weight), 3) if gait.any() else 0.,
                traverse_pass=bool(travel >= TRAVEL_M and not (~held[gait]).any() and end_both))


def film(trial, stem, size=(640, 480), wide=False):
    """Render every second tick, two views side by side, with the loads written on top.
    `wide`: the left view is fixed and shows the whole rope; the right one follows the robot."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    m = trial.w.model
    renderer = mujoco.Renderer(m, height=size[1], width=size[0])
    writer = imageio.get_writer(str(stem.with_suffix('.mp4')), fps=25, codec='libx264', macro_block_size=2, quality=7)
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    views = ((90, -8, .6), (25, -18, .6))
    count = [0]

    def frame(tr):
        count[0] += 1
        if count[0] % 2:
            return
        r = tr.rows[-1]
        look = np.array(r['com']); look[2] = .41
        ims = []
        for i, (az, el, dist) in enumerate(views):
            cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = az, el, dist
            cam.lookat[:] = (0., 0., .36) if wide and i == 0 else look
            if wide and i == 0:
                cam.distance = 1.25
            renderer.update_scene(tr.w.data, camera=cam); ims.append(renderer.render())
        im = Image.fromarray(np.hstack(ims)); draw = ImageDraw.Draw(im)
        travel = 1000 * tr.gait.direction * (r['arc'] - tr.arc0)
        lines = [f'Rope inchworm | legs only, contact physics | t {r["t"]:5.2f}s | {r["phase"]}',
                 f'rope load: left hook {r["left"]:.1f} N  right hook {r["right"]:.1f} N  other {r["other"] + r["head"]:.1f} N'
                 f'  floor {r["floor"]:.1f} N  post {r["post"]:.1f} N | travel {travel:+.0f} mm']
        for i, s in enumerate(lines):
            draw.text((8, 6 + 18 * i), s, font=font, fill='black')
        writer.append_data(np.asarray(im))

    def close():
        writer.close(); renderer.close()
    return frame, close


def run(gait=Gait(), cycles=12, direction=1, do_film=True, scene=Scene(), kp=KP, note='', x=0., wide=False):
    if do_film and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('MUJOCO_GL') != 'egl'):
        raise RuntimeError('Film through video.sbatch on a GPU node, or pass --no-film')
    stem = reserve()
    trial = Trial(gait, cycles, direction, scene, kp, x)
    sources = {p: p.read_bytes() for p in ROOT.glob('*.py')}
    for cls in (type(trial.w), type(trial.duck)):
        p = Path(inspect.getfile(cls)); sources[p] = p.read_bytes()
    warnings = []; old = mujoco.get_mju_user_warning(); mujoco.set_mju_user_warning(warnings.append)
    frame = close = None
    if do_film:
        frame, close = film(trial, stem, wide=wide)
    error = None; result = {}
    try:
        result = trial.run(on_frame=frame)
    except Exception as e:
        error = repr(e); raise
    finally:
        mujoco.set_mju_user_warning(old)
        if close: close()
        rows = trial.rows
        np.savez_compressed(stem.with_suffix('.npz'), t=[r['t'] for r in rows], arc=[r['arc'] for r in rows],
                            left=[r['left'] for r in rows], right=[r['right'] for r in rows],
                            other=[r['other'] + r['head'] for r in rows], floor=[r['floor'] for r in rows],
                            com=[r['com'] for r in rows], torque=[r['torque'] for r in rows])
        out = dict(method='leg-hook inchworm (legs only, upside down)', note=note, gait=asdict(gait),
                   cycles=cycles, direction=direction, start_x=x, kp_fw=kp, scene=asdict(scene), error=error,
                   start_gap_m=trial.gap, warning_messages=warnings,
                   physics_valid=error is None and not warnings,
                   video=str(stem.with_suffix('.mp4')) if do_film else None,
                   sources={str(p): hashlib.sha256(b).hexdigest() for p, b in sources.items()})
        out.update(result)
        if not out['physics_valid']:
            out.update(hold_pass=False, traverse_pass=False)
        with zipfile.ZipFile(stem.with_suffix('.sources.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
            for p, b in sources.items():
                z.writestr(str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else p.name, b)
        stem.with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
        lock = stem.with_suffix('.lock')
        if lock.exists(): lock.unlink()
        print(json.dumps({k: v for k, v in out.items() if k not in ('sources', 'gait', 'scene')}), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cycles', type=int, default=12)
    p.add_argument('--direction', type=int, default=1, choices=(1, -1))
    p.add_argument('--gait', type=Path, help='JSON file with Gait fields (default: built-in)')
    p.add_argument('--kp', type=float, default=KP)
    p.add_argument('--no-film', action='store_true')
    p.add_argument('--note', default='')
    p.add_argument('--x', type=float, default=0., help='start position along the rope (m, 0 = centre)')
    p.add_argument('--wide', action='store_true', help='left view shows the whole rope')
    a = p.parse_args()
    gait = Gait.from_dict(json.loads(a.gait.read_text())) if a.gait else Gait()
    run(gait, a.cycles, a.direction, not a.no_film, kp=a.kp, note=a.note, x=a.x, wide=a.wide)


if __name__ == '__main__':
    main()
