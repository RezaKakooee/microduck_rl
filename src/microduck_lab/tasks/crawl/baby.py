"""Synchronized tuck/drive crawl, inspired by the swing's face-down landing.

Search: 8 CEM rounds x 40 candidates x 8 seconds. Both legs always receive
mirrored targets from the SAME pose. Uses the original Crawl physics unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.crawl.crawl import Crawl, DT
from microduck_lab.tasks.crawl.expert import _label

POSE_JOINTS = ('hip_pitch', 'knee', 'ankle', 'neck_pitch', 'head_pitch')
PARAM_NAMES = [f'{pose}_{joint}' for pose in ('tuck', 'drive')
               for joint in POSE_JOINTS] + ['tuck_seconds', 'drive_seconds']
REFERENCE = np.array([-.35, 1.30, 0., -.30, -.30,
                      -1.50, -1.45, 0., -.20, -.10, .70, .90])
# Keep the recognizable fold/extend shape while searching timing and contact.
LOW = np.array([-.90, .35, -.70, -.80, -.80,
                -1.56, -1.55, -.70, -.80, -.80, .18, .18])
HIGH = np.array([.30, 1.55, .70, .30, .30,
                 -.70, -.30, .70, .30, .30, .90, 1.10])
PARAMS = paths.REPO / 'crawl_baby_gait.npy'


def gait(c, p, t):
    """A piecewise constant, in-phase two-pose cycle (right = -left)."""
    phase = 0 if t % (p[-2] + p[-1]) < p[-2] else 1
    q = c.base.copy()
    for joint, value in zip(POSE_JOINTS, p[phase * 5:phase * 5 + 5]):
        if joint in ('neck_pitch', 'head_pitch'):
            q[c.ai[joint]] = value
        else:
            q[c.ai['left_' + joint]] = value
            q[c.ai['right_' + joint]] = -value
    return q


def evaluate(p, seconds=20., video=None, trace=None, voltage=7.4):
    """One rollout path for search, validation, and rendering; net displacement.

    A motor-update observer checks posture at every 2 ms physics step, without
    changing controls or state. Reject upright/rolled postures as well as height.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != REFERENCE.shape or not np.isfinite(p).all() or np.any(p[-2:] <= 0):
        raise ValueError('Expected 12 finite parameters with positive dwell times')
    steps = round(seconds / DT)
    if steps < 1 or abs(steps * DT - seconds) > 1e-8:
        raise ValueError('Duration must be a positive multiple of 0.02 seconds')
    c = Crawl(voltage=voltage)
    # Continue the existing 25 cm scenery on both sides of the start line.
    # Collision masks stay zero; physics is identical to the original scene.
    for k in range(11, 22):
        c.model.geom_pos[c.id('GEOM', f'mark_{k}'), 0] = (k - 10) * .25
    max_height = float(c.start[2])
    max_upright = 0.
    failure = None
    original_update = c.motor.update

    def observe():
        nonlocal max_height, max_upright
        z = float(c.data.xpos[c.trunk, 2])
        upright = abs(float(c.data.xmat[c.trunk, 8]))
        max_height = max(max_height, z)
        max_upright = max(max_upright, upright)
        if z > .200:
            raise RuntimeError('Trunk exceeded 200 mm')
        if upright > .70:
            raise RuntimeError('Trunk became upright/inverted instead of crawling')
        # Local lateral axis must remain horizontal: no rolling onto the side.
        if abs(float(c.data.xmat[c.trunk, 7])) > .60:
            raise RuntimeError('Rolled onto its side instead of crawling')

    def monitored_update():
        observe()
        original_update()

    c.motor.update = monitored_update
    rows = [[0., *c.start, c.data.xmat[c.trunk, 8]]]
    joint_rows = [c.data.qpos[c.qidx].copy()]
    writer = renderer = None
    try:
        if video:
            import imageio.v2 as imageio
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(c.model, height=720, width=1280)
            writer = imageio.get_writer(str(video), fps=25, codec='libx264',
                                       quality=None, macro_block_size=8,
                                       output_params=['-crf', '24', '-preset', 'medium',
                                                      '-pix_fmt', 'yuv420p',
                                                      '-movflags', '+faststart'])
        for i in range(steps):
            c.step(gait(c, p, c.t))
            observe()
            rows.append([c.t, *c.data.xpos[c.trunk], c.data.xmat[c.trunk, 8]])
            joint_rows.append(c.data.qpos[c.qidx].copy())
            if renderer is not None and i % 2 == 1:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = c.data.xpos[c.trunk] + np.array([.06, 0., .025])
                cam.distance = .72
                cam.azimuth = 105
                cam.elevation = -13
                renderer.update_scene(c.data, camera=cam)
                mm = 1000 * c.travelled()[0]
                label = f'Baby crawl | {c.t:4.1f} s | {mm:.0f} mm | {mm / c.t:.0f} mm/s'
                writer.append_data(_label(renderer.render(), label + '\nStripes: 25 cm | real time'))
    except RuntimeError as exc:
        failure = str(exc)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    a = np.asarray(rows)
    far, dx, dy = c.travelled()
    elapsed = c.t
    half = a[np.searchsorted(a[:, 0], elapsed / 2), 1:3]
    second_half = float(np.linalg.norm(a[-1, 1:3] - half))
    report = dict(
        success=failure is None and far > .05 and second_half > .005 * elapsed / 2,
        failure=failure,
        seconds=elapsed, requested_seconds=seconds,
        travelled_mm=1000 * far, delta_x_mm=1000 * dx, delta_y_mm=1000 * dy,
        speed_mm_s=1000 * far / elapsed if elapsed else 0.,
        second_half_speed_mm_s=1000 * second_half / (elapsed / 2) if elapsed else 0.,
        trunk_height_mm=dict(mean=1000 * float(a[:, 3].mean()), max=1000 * max_height),
        maximum_abs_trunk_upright_cosine=max_upright,
        maximum_servo_torque_nm=c.max_torque, servo_torque_limit_nm=c.torque_limit,
        net_servo_work_j=c.work, gait=dict(zip(PARAM_NAMES, p.tolist())),
        controller='in-phase mirrored tuck/drive, BAM XL330 M6, 50 Hz',
        voltage=voltage, settle_seconds=1., physics_timestep=.002,
        posture_check_hz=500, distance_metric='net horizontal trunk displacement')
    if trace:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace, trajectory=a, joint_positions=joint_rows,
                            joint_names=c.names, params=p)
    return report


def _candidate(p):
    r = evaluate(p, seconds=8.)
    return (-1. if r['failure'] else r['travelled_mm'] / 1000), r


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def search(workers=4, seed=7, output=PARAMS):
    rng = np.random.default_rng(seed)
    mu = REFERENCE.copy()
    sd = (HIGH - LOW) / 3
    history, all_candidates = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for iteration in range(8):
            candidates = np.clip(rng.normal(mu, sd, (40, len(mu))), LOW, HIGH)
            if iteration == 0:
                candidates[0] = REFERENCE
            results = list(pool.map(_candidate, candidates))
            scores = np.array([s for s, _ in results])
            valid = np.flatnonzero(scores >= 0)
            if not len(valid):
                raise RuntimeError('No crawling candidates in this round')
            elite_ids = valid[np.argsort(scores[valid])[-8:]]
            elite = candidates[elite_ids]
            mu = elite.mean(axis=0)
            sd = np.maximum(elite.std(axis=0), (HIGH - LOW) * .035)
            for p, (s, r) in zip(candidates, results):
                all_candidates.append(dict(params=p.tolist(), score_m=s, report=r))
            summary = dict(round=iteration + 1, best_mm=1000 * float(scores.max()),
                           valid=len(valid), elite_mean_mm=1000 * float(scores[elite_ids].mean()))
            history.append(summary)
            print(json.dumps(summary), flush=True)
            save_json(paths.video('crawl', 'crawl_baby_search.json'),
                      dict(seed=seed, rounds=8, candidates_per_round=40, rollout_seconds=8,
                           bounds=dict(low=LOW.tolist(), high=HIGH.tolist()),
                           history=history, candidates=all_candidates))
        # Validate the strongest individual candidates at the FULL video duration.
        # Never assume the mean parameter vector is itself a successful gait.
        ranked = sorted(all_candidates, key=lambda a: a['score_m'], reverse=True)[:12]
        full = list(pool.map(_full_candidate, [x['params'] for x in ranked]))
    accepted = [(r['travelled_mm'], k) for k, r in enumerate(full)
                if r['success'] and r['second_half_speed_mm_s'] > 5.]
    if not accepted:
        raise RuntimeError('No candidate sustained crawling over the full 20 seconds')
    _, best = max(accepted)
    selected = np.array(ranked[best]['params'])
    np.save(output, selected)
    save_json(paths.video('crawl', 'crawl_baby_selection.json'),
              dict(full_duration_seconds=20, candidates=full, selected=best))
    print('Selected full-duration result:', json.dumps(full[best]), flush=True)
    return selected


def _full_candidate(p):
    return evaluate(p, seconds=20.)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--search', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--params', type=Path, default=PARAMS)
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--seconds', type=float, default=20.)
    ap.add_argument('--video')
    ap.add_argument('--trace')
    ap.add_argument('--report', default=paths.video('crawl', 'crawl_baby_report.json'))
    a = ap.parse_args()
    if a.search:
        search(a.workers, a.seed, a.params)
        return
    p = REFERENCE if a.reference else np.load(a.params)
    r = evaluate(p, seconds=a.seconds, video=a.video, trace=a.trace)
    save_json(a.report, r)
    print(json.dumps(r, indent=2), flush=True)
    if r['failure']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
