"""Headless CPU search: launch keyframes plus the closed-loop catch.

Every earlier hop landed with the COM behind the heel. This search scores the
landing directly: COM offset over the soles at touchdown, time on the feet
before any body contact, then the unchanged clean-landing judge. It uses the
same `Episode` physics as the filmed runner, so a result replays exactly
through `batch.py`. No video here: film the winners with `batch.py`.

    rescore: score earlier qualifying hops (with the catch) or a --cases list.
    search:  cross-entropy search from the best seeds; --perturb N scores each
             sample on N noisy copies, --sync-pose K also searches the pitch sync.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path

import numpy as np

from microduck_lab.tasks.salmon_jump.catch import Catch
from microduck_lab.tasks.salmon_jump.run import Candidate, Episode, summarize, CONTROL_DT

FOLDER = Path('local_storage/hb_dev/scripted_policy/salmon_jump/catch_search')
# Searched catch parameters and their bounds.
CATCH = {'com_offset': (-.01, .03), 'neck': (-1.5, 1.0), 'head': (-1.0, 1.4),
         'air_neck': (-1.5, 1.0), 'pitch_gain': (0., 1.), 'com_gain': (0., 20.),
         'crouch_s': (.1, 1.), 'rise_s': (.4, 2.), 'pitch_rate_limit': (.2, 1.5),
         'lookahead_s': (0., .15)}


def evaluate(candidate, seconds=9.):
    """Run one episode headless. Stops early once the outcome is decided."""
    episode = Episode(candidate)
    duck, extra = episode.duck, []
    takeoff = None
    for _ in range(round(seconds/CONTROL_DT)):
        episode.tick()
        row = episode.rows[-1]
        sole = duck.points(duck.sole_geoms)
        R = duck.R()
        extra.append((duck.com()[0]-sole[:, 0].mean(), np.arctan2(-R[2, 0], R[2, 2]),
                      episode.w.data.qvel[duck.dof+4]))
        t = row[0]
        if takeoff is None and episode.recovery and t > 3.:
            takeoff = t
        if takeoff is not None and row[9] > .5:
            break  # body contact after the catch started: the clean landing failed
        if takeoff is None and t > 3.+sum(candidate.durations)+candidate.hold+1.:
            break  # no takeoff
    rows = np.asarray(episode.rows)
    extra = np.asarray(extra)
    result = summarize(rows, candidate.kind) if len(rows) > 155 else dict(valid_back_start=False)
    result.update(candidate=asdict(candidate), seconds_run=float(rows[-1, 0]))
    if result.get('takeoff_time_s') is not None:
        # Time on the feet, from the judge's own takeoff (not the catch trigger).
        post = rows[rows[:, 0] >= result['takeoff_time_s']]
        hit = np.flatnonzero(post[:, 9] > .5)
        result['body_contact_after_takeoff_s'] = float((post[hit[0], 0] if len(hit) else rows[-1, 0])
                                                       - result['takeoff_time_s'])
        i = int(np.argmin(np.abs(rows[:, 0]-result['takeoff_time_s'])))
        after = np.flatnonzero(rows[i:, 8]+rows[i:, 9] > .05)
        if len(after):
            j = i+after[0]
            result.update(land_offset_m=float(extra[j, 0]), land_pitch=float(extra[j, 1]),
                          land_rate=float(extra[j, 2]), takeoff_offset_m=float(extra[i, 0]),
                          takeoff_rate=float(extra[i, 2]))
    # Arrival on the feet: after the first feet-only support, how far forward
    # does the COM get over the soles before any body contact (0.5 s window)?
    active = np.flatnonzero((rows[:, 0] > 3.) & (rows[:, 8] > 1.) & (rows[:, 9] < .5) & (rows[:, 4] > .5))
    if len(active):
        k = active[0]
        window = np.arange(k, min(len(rows), k+25))
        body = np.flatnonzero(rows[window, 9] > .5)
        window = window[:body[0]] if len(body) else window
        result.update(arrival_time_s=float(rows[k, 0]), arrival_rate=float(extra[k, 2]),
                      arrival_best_offset_m=float(extra[window, 0].max()))
    # Launch shaping when there is no qualifying takeoff (as takeoff_search.py).
    a = rows[(rows[:, 0] > 3.) & (rows[:, 4] < .8) & (rows[:, 13] > .05)]
    if len(a):
        unload = np.clip(1-(a[:, 8]+a[:, 9])/(duck.mass()*9.81), 0, 1)
        result['launch_momentum'] = float(np.max(a[:, 13]*unload))
    return result


def upright_flight(r):
    """A flight counts only if the robot is near upright in the air and at touchdown.
    (A bounce on the back also passes the judge's flight test.)"""
    return (r.get('takeoff_time_s') is not None and r.get('land_offset_m') is not None
            and (r.get('max_up_while_airborne') or 0.) > .7 and abs(r['land_pitch']) < 1.)


def score(r):
    if not r.get('valid_back_start'):
        return -100.
    if not upright_flight(r):
        return float(10*r.get('launch_momentum', 0.) + 50*min(r.get('max_clearance_m', 0.), .02))
    survive = r['body_contact_after_takeoff_s']
    return float(20 + 100*min(r['qualifying_airborne_s'], .1) + 300*min(r['max_clearance_m'], .02)
                 - 300*abs(r['land_offset_m']-.005) + 10*min(survive, 3.)
                 + 5*min(r['final_standing_s'], 2.) + 50*r['clean_landing'] + 100*r['success'])


def score_arrival(r):
    """Stage scores: COM over the feet after arrival, then flight, then landing."""
    if not r.get('valid_back_start'):
        return -100.
    s = 0.
    if r.get('arrival_best_offset_m') is not None:
        s += 100*np.clip(r['arrival_best_offset_m']+.06, 0., .065)  # up to 6.5 at +5 mm
    s += 5*r.get('launch_momentum', 0.) + 50*min(max(r.get('max_clearance_m', 0.), 0.), .01)
    if upright_flight(r):
        survive = r['body_contact_after_takeoff_s']
        s += (10 + 100*min(r['qualifying_airborne_s'], .1) + 300*min(r['max_clearance_m'], .02)
              - 200*abs(r['land_offset_m']-.005) + 10*min(survive, 3.))
    return float(s + 5*min(r.get('final_standing_s', 0.), 2.)
                 + 50*r.get('clean_landing', False) + 100*r.get('success', False))


def score_height(r):
    """Among successes, prefer longer flight and more clearance."""
    s = score_arrival(r)
    if r.get('success'):
        s += 200*min(r['qualifying_airborne_s'], .15) + 1000*min(r['max_clearance_m'], .03)
    return s


SCORES = {'landing': score, 'arrival': score_arrival, 'height': score_height}


def job(payload):
    candidate, seconds, objective = payload
    try:
        r = evaluate(Candidate(**candidate), seconds)
    except Exception as exc:  # a bad candidate must not stop the search
        r = dict(candidate=candidate, error=repr(exc), valid_back_start=False)
    r['score'] = SCORES[objective](r)
    return r


SYNC = (-1.3, -.1)  # bounds of the searched sync pitch (rad)


def pack(c, sync=False):
    x = [np.ravel(c['poses']), c['durations'], [c['hold']]]
    catch = c.get('catch') or {}
    x.append([catch.get(k, getattr(Catch(), k)) for k in CATCH])
    if sync:
        x.append([c.get('sync_pitch', -.86)])
    return np.concatenate(x)


def unpack(x, template, fixed=None, sync_pose=-1):
    n = len(template['poses'])
    c = dict(template)
    c['poses'] = x[:5*n].reshape(n, 5).tolist()
    c['durations'] = x[5*n:6*n].tolist()
    c['hold'] = float(x[6*n])
    c['recovery'] = 'catch'
    c['catch'] = dict(zip(CATCH, map(float, x[6*n+1:6*n+1+len(CATCH)])))
    c['catch'].update(fixed or {})  # e.g. hand over to the standing policy after the crouch
    if sync_pose >= 0:
        c['sync_pose'], c['sync_pitch'] = sync_pose, float(x[-1])
    return c


def bounds(n, sync=False):
    low = np.r_[np.tile([-1.5]*5, n), np.full(n, .04), .1, [v[0] for v in CATCH.values()]]
    high = np.r_[np.tile([1.5, 1.5, 1.5, 1., 1.5], n), np.full(n, .8), .8, [v[1] for v in CATCH.values()]]
    if sync:
        low, high = np.r_[low, SYNC[0]], np.r_[high, SYNC[1]]
    return low, high


def output(name):
    FOLDER.mkdir(parents=True, exist_ok=True)
    for n in range(1, 10000):
        path = FOLDER/f'{name}_{n}.json'
        try:
            with path.open('x') as f:
                f.write('{}\n')
            return path
        except FileExistsError:
            continue


def rescore(args, pool):
    if args.cases:  # a JSON list of candidates, scored as given
        cases = json.loads(Path(args.cases).read_text())
    else:  # earlier qualifying hops, re-run with the catch
        cases = []
        for t in json.loads(Path(args.review).read_text())['trials']:
            if t.get('takeoff_com_vz') is None:
                continue
            c = json.loads(Path(t['video']).with_suffix('.json').read_text())['candidate']
            c.update(recovery='catch', catch=dict(args.catch))
            cases.append(c)
    path = output('rescore')
    results = []
    for r in pool.map(job, [(c, args.seconds, args.objective) for c in cases]):
        results.append(r)
        print('RESCORE', round(r['score'], 2), r.get('land_offset_m'), r.get('body_contact_after_takeoff_s'), r.get('success'), flush=True)
        results.sort(key=lambda r: -r['score'])
        path.write_text(json.dumps(dict(catch=args.catch, results=results), indent=1)+'\n')
    print('RESCORE_RESULT', path, flush=True)


def perturb(c, noise):
    """A copy with keyframe angle, duration and start-pitch noise (robustness)."""
    angles, durations, pitch = noise
    c = dict(c)
    c['poses'] = (np.asarray(c['poses'])+angles).tolist()
    c['durations'] = np.maximum(.04, np.asarray(c['durations'])+durations).tolist()
    c['start_pitch'] = float(c['start_pitch']+pitch)
    return c


def search(args, pool):
    rng = np.random.default_rng(args.seed)
    unique = {}
    for s in json.loads(Path(args.seeds).read_text())['results']:
        unique.setdefault(json.dumps(s['candidate'], sort_keys=True), s)  # replays repeat
    template = next(iter(unique.values()))['candidate']
    n = len(template['poses'])
    seeds = [s for s in unique.values() if len(s['candidate']['poses']) == n][:args.num_seeds]
    sync = args.sync_pose >= 0
    low, high = bounds(n, sync)
    X = np.array([pack(s['candidate'], sync) for s in seeds])
    mean, std = X[0], (high-low)*args.spread
    best, history = None, []
    path = output('search')
    for generation in range(args.generations):
        samples = np.clip(mean+rng.normal(size=(args.population, len(low)))*std, low, high)
        if generation == 0:
            samples[:len(X)] = X
        elif best is not None:
            samples[0] = pack(best['candidate'], sync)
        candidates = [unpack(x, template, args.fix, args.sync_pose) for x in samples]
        if args.perturb > 1:
            # The same noise for every sample in a generation; copy 0 is unperturbed.
            draw = np.random.default_rng(1000*args.seed+generation)
            noise = [(np.zeros((n, 5)), np.zeros(n), 0.)] + [
                (draw.normal(0, .02, (n, 5)), draw.normal(0, .003, n), draw.uniform(-.02, .02))
                for _ in range(args.perturb-1)]
            runs = list(pool.map(job, [(perturb(c, e), args.seconds, args.objective)
                                       for c in candidates for e in noise]))
            results = []
            for i, c in enumerate(candidates):
                group = runs[i*len(noise):(i+1)*len(noise)]
                r = dict(group[0])  # the unperturbed run, with the group's mean score
                r['score'] = float(np.mean([g['score'] for g in group]))
                r['success_rate'] = float(np.mean([bool(g.get('success')) for g in group]))
                results.append(r)
        else:
            results = list(pool.map(job, [(c, args.seconds, args.objective) for c in candidates]))
        scores = np.array([r['score'] for r in results])
        for r in results:
            history.append({k: r.get(k) for k in ('score', 'success', 'clean_landing', 'land_offset_m',
                                                  'arrival_best_offset_m', 'arrival_rate',
                                                  'body_contact_after_takeoff_s', 'qualifying_airborne_s',
                                                  'max_clearance_m', 'final_standing_s', 'success_rate',
                                                  'candidate')})
            if best is None or r['score'] > best['score']:
                best = r
        elite = samples[np.argsort(scores)[-max(4, args.population//8):]]
        mean = .3*mean+.7*elite.mean(axis=0)
        std = np.maximum((high-low)*.01, .3*std+.7*elite.std(axis=0))
        path.write_text(json.dumps(dict(generation=generation, best=best, history=history), indent=1)+'\n')
        print('GENERATION', generation, 'best', round(best['score'], 2), 'success', best.get('success'),
              'land_offset', best.get('land_offset_m'), 'arrival_offset', best.get('arrival_best_offset_m'),
              'flight', best.get('qualifying_airborne_s'), 'mean', round(float(scores.mean()), 2),
              'successes', int(sum(r.get('success', False) for r in results)),
              'best_success_rate', best.get('success_rate'), flush=True)
    print('SEARCH_RESULT', path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('rescore', 'search'))
    p.add_argument('--review', default='local_storage/hb_dev/scripted_policy/salmon_jump/review_2.json')
    p.add_argument('--cases', help='rescore this JSON list of candidates instead')
    p.add_argument('--catch', type=json.loads, default={})
    p.add_argument('--seeds')
    p.add_argument('--num-seeds', type=int, default=8)
    p.add_argument('--fix', type=json.loads, default={}, help='catch parameters kept fixed in the search')
    p.add_argument('--perturb', type=int, default=1, help='score each sample on this many noisy copies')
    p.add_argument('--sync-pose', type=int, default=-1, help='keyframe that waits for the IMU pitch (searched)')
    p.add_argument('--generations', type=int, default=30)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--spread', type=float, default=.08)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seconds', type=float, default=9.)
    p.add_argument('--objective', choices=tuple(SCORES), default='arrival')
    args = p.parse_args()
    # Fresh workers every few episodes: memory grows with each built world.
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=10) as pool:
        (rescore if args.mode == 'rescore' else search)(args, pool)


if __name__ == '__main__':
    main()
