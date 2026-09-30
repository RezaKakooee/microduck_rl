"""Optimize upward unloading; unsupported falling receives no flight reward."""
from pathlib import Path
import numpy as np
from microduck_lab.rl.scripted_policy.salmon_jump import search


def takeoff_score(r):
    if r['error'] or not r['valid_back_start']:
        return -100.
    a=np.load(Path(r['video']).with_suffix('.npz'))['trace']
    if r.get('takeoff_com_vz') is not None:
        post=a[a[:,0]>=r['takeoff_time_s']]
        stumble=min(8.,.1*np.count_nonzero(post[:,9]>.5))
        return float(10+50*r['max_clearance_m']+10*r['takeoff_com_vz']
                     +5*r['qualifying_airborne_s']+3*min(2.,r['final_standing_s'])
                     -stumble+20*r.get('clean_landing',False)+50*r['success'])
    a=a[(a[:,0]>3.) & (a[:,12]<.5) & (a[:,4]<.8) & (a[:,13]>.05)]
    if not len(a):
        return -10.
    unload=np.clip(1-(a[:,8]+a[:,9])/(r['robot_mass_kg']*9.81),0,1)
    # Simultaneous upward speed and unloading, rather than peaks from different times.
    momentum=np.max(a[:,13]*unload)
    clearance=np.max(np.clip(a[:,10],-.005,.05))
    rising_air=(a[:,8]+a[:,9]<.05) & (a[:,10]>.002)
    return float(12*momentum+80*clearance+2*np.max(a[:,13])
                 +4*np.count_nonzero(rising_air)
                 +(5*min(2.,r['final_standing_s']) if r['takeoff_com_vz'] is not None else 0.)
                 +50*r['success'])


if __name__=='__main__':
    search.score=takeoff_score
    search.main()
