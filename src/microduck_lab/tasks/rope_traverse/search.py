"""Headless CPU tools for the inchworm (no video).

    check:  run one gait under 16 changed conditions and print a table.
    search: cross-entropy search over the Gait fields. Each sample is scored
            in several conditions; the worst score counts.

    sbatch -M cluster -p h200 --cpus-per-task=16 --mem=24G <cpu sbatch> \
        -m microduck_lab.tasks.rope_traverse.search check \
        --gait src/microduck_lab/tasks/rope_traverse/fast_inchworm.json

Score = speed (mm/s) minus 2 for every 20 ms tick in which the legs carry less
than half the weight (the judge's hold rule), minus 10 for floor contact,
another body part on the rope, or an error.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import time

import numpy as np

from microduck_lab.tasks.rope_traverse.inchworm import Gait
from microduck_lab.tasks.rope_traverse.scene import Scene
from microduck_lab.tasks.rope_traverse.traverse import Trial
from microduck_lab.tasks.rope_traverse.arch import KP

CONDITIONS = {
    'nominal': {},
    'friction x0.7': dict(friction=.7),
    'friction x1.3': dict(friction=1.3),
    'mass x1.05': dict(mass=1.05),
    'mass x0.95': dict(mass=.95),
    'kp 400': dict(kp=400.),
    'kp 200': dict(kp=200.),
    'battery 7.0 V': dict(vin=7.0),
    'rope sag 20 mm': dict(sag=.02),
    'rope sag 6 mm': dict(sag=.006),
    'rope 10 mm thick': dict(radius=.005),
    'rope 15 mm thick': dict(radius=.0075),
    'start 10 cm left of centre': dict(x=-.10),
    'start 10 cm right of centre': dict(x=.10),
    'moving the other way': dict(direction=-1),
    'rope 60 cm high': dict(height=.60),
}
SEARCH_CONDITIONS = ('nominal', 'friction x0.7 mass x1.05', 'kp 400')
# Search bounds for each Gait field.
BOUNDS = dict(lead_in=(-.384, 0.), lead_out=(0., .384), trail_out=(-.384, 0.), trail_in=(0., .384),
              clamp_knee=(1.0, 1.57), clamp_ankle=(-1.0, 0.), free_knee=(-.2, .9), free_ankle=(-.5, .6),
              carry_knee=(-.2, 1.2), carry_ankle=(-.5, .6), grip_trail=(.3, 2.), slide_lead=(.3, 2.),
              grip_lead=(.3, 2.), slide_trail=(.3, 2.), relax=(.3, 2.), carry=(.3, 2.))


def condition(name):
    """A named condition; names may combine, e.g. 'friction x0.7 mass x1.05'."""
    if name in CONDITIONS:
        return dict(CONDITIONS[name])
    out = {}
    for part in ('friction x0.7', 'friction x1.3', 'mass x1.05', 'mass x0.95', 'kp 400', 'kp 200'):
        if part in name:
            out.update(CONDITIONS[part])
    return out


def evaluate(gait_dict, name, cycles):
    """Run one gait in one condition. Returns the judge result plus the travel per cycle."""
    v = condition(name)
    scene = Scene()
    for key in ('sag', 'radius', 'height'):
        if key in v:
            scene = replace(scene, **{key: v[key]})
    t0 = time.time()
    try:
        trial = Trial(Gait.from_dict(gait_dict), cycles=cycles, direction=v.get('direction', 1),
                      scene=scene, kp=v.get('kp', KP), x=v.get('x', 0.))
        m, k = trial.w.model, trial.duck
        if 'friction' in v:
            m.geom_friction[:, 0] *= v['friction']
        if 'mass' in v:
            for b in k.bodies:
                m.body_mass[b] *= v['mass']
        if 'vin' in v:
            k.motor.model.actuator.vin = v['vin']
        result = trial.run()
        arc = np.array([r['arc'] for r in trial.rows])
        t = np.array([r['t'] for r in trial.rows]) - trial.t_gait
        marks = [arc[min(np.searchsorted(t, c - 1e-6), len(arc) - 1)] for c in trial.gait.cycle_starts()]
        result['travel_per_cycle_mm'] = (np.diff(marks) * 1000 * trial.gait.direction).round(1).tolist()
        result['error'] = None
    except Exception as e:
        result = dict(error=repr(e), speed_mm_s=0., broken_hold_ticks=0)
    result.update(condition=name, wall_s=round(time.time() - t0))
    return result


def score(r):
    if r['error'] or r.get('floor_contact') or r.get('other_rope_contact'):
        return -10. + .1 * r['speed_mm_s']
    return r['speed_mm_s'] - 2. * r['broken_hold_ticks']


def _evaluate(args):
    return evaluate(*args)


def check(gait_dict, cycles, workers):
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_evaluate, [(gait_dict, name, cycles) for name in CONDITIONS]))
    for r in results:
        note = '' if not r.get('broken_hold_ticks') else '  dip: %d ticks, %d%% of weight' % (
            r['broken_hold_ticks'], round(100 * r['min_leg_support']))
        if r['error']: note = '  ' + r['error']
        elif r['floor_contact'] or r['other_rope_contact']: note += '  CONTACT'
        print('%-28s travel %6.1f mm  %5.2f mm/s%s' % (r['condition'], 1000 * r.get('travel_m', 0.),
                                                     r['speed_mm_s'], note), flush=True)
    return results


def search(start, cycles, workers, generations, population, elite, spread, seed, out):
    keys = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in keys]); hi = np.array([BOUNDS[k][1] for k in keys])
    mean = np.array([start[k] for k in keys]); sd = (hi - lo) * spread
    rng = np.random.default_rng(seed)
    history = []
    with ProcessPoolExecutor(workers) as pool:
        for gen in range(generations):
            X = np.clip(mean + sd * rng.standard_normal((population, len(keys))), lo, hi)
            X[0] = np.clip(mean, lo, hi)
            samples = [dict(start, **dict(zip(keys, x.tolist()))) for x in X]
            jobs = [(s, c, cycles) for s in samples for c in SEARCH_CONDITIONS]
            results = list(pool.map(_evaluate, jobs))
            n = len(SEARCH_CONDITIONS)
            scores = np.array([min(score(r) for r in results[i * n:(i + 1) * n]) for i in range(population)])
            order = np.argsort(-scores)
            mean = X[order[:elite]].mean(0); sd = np.maximum(X[order[:elite]].std(0), (hi - lo) * .02)
            best = int(order[0])
            history.append(dict(generation=gen, best_score=float(scores[best]), mean_score=float(scores.mean()),
                                best=samples[best], results=results[best * n:(best + 1) * n]))
            print(json.dumps(dict(generation=gen, best=round(float(scores[best]), 2), mean=round(float(scores.mean()), 2))), flush=True)
            Path(out).write_text(json.dumps(history, indent=1))
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('check', 'search'))
    p.add_argument('--gait', type=Path, help='start gait (JSON); default: built-in reference')
    p.add_argument('--cycles', type=int, default=5)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--generations', type=int, default=10)
    p.add_argument('--population', type=int, default=20)
    p.add_argument('--elite', type=int, default=5)
    p.add_argument('--spread', type=float, default=.1, help='first sample spread, as a fraction of each bound')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('local_storage/hb_dev/scripted_policy/rope_traverse/search.json'))
    a = p.parse_args()
    start = asdict(Gait.from_dict(json.loads(a.gait.read_text()))) if a.gait else asdict(Gait())
    if a.command == 'check':
        results = check(start, a.cycles, a.workers)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.with_name(a.out.stem + '_check.json').write_text(json.dumps(dict(gait=start, results=results), indent=1))
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        search(start, a.cycles, a.workers, a.generations, a.population, a.elite, a.spread, a.seed, a.out)


if __name__ == '__main__':
    main()
