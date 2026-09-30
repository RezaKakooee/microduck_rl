"""Small filmed parameter search; no learned policy or physics changes."""
import argparse
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
from microduck_lab.rl.scripted_policy.salmon_jump.run import Candidate, run

LOW=np.r_[np.tile([-1.5,-1.5,-1.5,-1.5,-1.5],3),[.08,.08,.08,.1]]
HIGH=np.r_[np.tile([1.5,1.5,1.5,1.,1.5],3),[.65,.65,.65,.8]]


def unpack(x, template=None):
    n = (len(x)-1)//6
    extra = {} if template is None else {k:v for k,v in template.items()
                                         if k not in ('poses','durations','hold')}
    return Candidate(poses=x[:5*n].reshape(n,5).tolist(),
                     durations=x[5*n:6*n].tolist(),hold=float(x[-1]),**extra)


def evaluate(payload):
    candidate, seconds = payload
    return run(candidate, seconds=seconds, width=640)


def score(r):
    if not r['valid_back_start'] or r['error']:
        return -100.
    return (4*r['max_up'] + 12*r['com_rise_m'] + 8*r['longest_airborne_s']
            + 2*r['final_up'] + 4*min(2.,r['final_standing_s']) + 30*r['success'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generations',type=int,default=4)
    p.add_argument('--population',type=int,default=32)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--resume',type=Path)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--spread',type=float,default=.2)
    a=p.parse_args();rng=np.random.default_rng(a.seed)
    low,high=LOW.copy(),HIGH.copy()
    template=None
    folder=Path('local_storage/hb_dev/scripted_policy/salmon_jump');folder.mkdir(parents=True,exist_ok=True)
    for n in range(1,10000):
        path=folder/f'search_{n}.json'
        try:
            with path.open('x') as f: f.write('{}\n')
            break
        except FileExistsError: continue
    mean=(low+high)/2;std=(high-low)*.35
    if a.resume:
        old=json.loads(a.resume.read_text());seed_result=old.get('best',old);c=seed_result['candidate'];template=c
        n=len(c['poses'])
        low=np.r_[np.tile(LOW[:5],n),np.full(n,.04),.1]
        high=np.r_[np.tile(HIGH[:5],n),np.full(n,.8),.8]
        mean=np.r_[np.ravel(c['poses']),c['durations'],c['hold']];std=(high-low)*a.spread
        # Always re-evaluate the seed under this objective and current judge.
        best=None
    else: best=None
    history=[]
    pool_context = (ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'))
                    if a.workers > 1 else nullcontext(None))
    with pool_context as pool:
        optimize(a,rng,path,low,high,mean,std,best,history,template,pool)


def optimize(a,rng,path,low,high,mean,std,best,history,template,pool):
    for generation in range(a.generations):
        if generation==0 and a.resume is None:
            samples=rng.uniform(low,high,size=(a.population,len(low)))
        else:
            samples=np.clip(mean+rng.normal(size=(a.population,len(low)))*std,low,high)
        if generation==0 and a.resume is not None:
            samples[0]=mean
        if best is not None:
            c=best['candidate'];samples[0]=np.r_[np.ravel(c['poses']),c['durations'],c['hold']]
        scores=[]
        candidates=[unpack(x,template) for x in samples]
        payload=[(c,a.seconds) for c in candidates]
        results=pool.map(evaluate,payload) if pool else map(evaluate,payload)
        for c,r in zip(candidates,results):
            value=score(r);scores.append(value)
            row=dict(score=value,candidate=asdict(c),video=r['video'],success=r['success'],
                     airborne=r['longest_airborne_s'],rise=r['com_rise_m'],
                     max_up=r['max_up'],final_up=r['final_up'],hold=r['final_standing_s'])
            history.append(row)
            if best is None or value>best['score']: best=row
            path.write_text(json.dumps(dict(generation=generation,best=best,history=history),indent=2)+'\n')
        elite=samples[np.argsort(scores)[-max(4,a.population//5):]]
        mean=.2*mean+.8*elite.mean(axis=0)
        std=np.maximum((high-low)*.025,.2*std+.8*elite.std(axis=0))
        print('GENERATION',generation,'BEST',json.dumps(best),flush=True)
    print('SEARCH_RESULT',str(path),flush=True)


if __name__=='__main__':main()
