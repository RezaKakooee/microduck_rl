"""Search for unloading and flight; grounded stand-up earns no landing bonus."""
from pathlib import Path
import numpy as np
from microduck_lab.tasks.salmon_jump import search


def flight_score(r):
    if not r['valid_back_start'] or r['error']:
        return -100.
    a=np.load(Path(r['video']).with_suffix('.npz'))['trace']
    active=a[(a[:,0]>3.) & (a[:,0]<6.5)]
    launch=active[(active[:,13]>.1) & (active[:,4]<.8)]
    if len(launch):
        unload=np.clip(1.-np.min(launch[:,8]+launch[:,9])/(r['robot_mass_kg']*9.81),0.,1.)
        clearance=np.clip(launch[:,10].max(),-.005,.03)
    else:
        unload,clearance=0.,-.005
    flight=r['longest_airborne_s']
    return float(6*unload+100*clearance+2*max(0.,r['peak_vertical_speed'])
                 +10*r['com_rise_m']+.5*r['max_up']+50*flight
                 +(4*min(2.,r['final_standing_s']) if flight>=.04 else 0.)
                 +50*r['success'])


if __name__=='__main__':
    search.score=flight_score
    print('OBJECTIVE flight_score: unload floor during upward motion before upright; no grounded landing bonus',flush=True)
    search.main()
