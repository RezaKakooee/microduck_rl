"""Prioritize measured upward impulse and genuine flight before landing."""
from microduck_lab.tasks.salmon_jump import search


def launch_score(r):
    if not r['valid_back_start'] or r['error']:
        return -100.
    return (30*r['com_rise_m'] + 3*max(0.,r['peak_vertical_speed'])
            + 30*r['longest_airborne_s'] + 2*r['max_up']
            + 2*r['final_up'] + 4*min(2.,r['final_standing_s']) + 40*r['success'])


if __name__ == '__main__':
    search.score = launch_score
    print('OBJECTIVE launch_score: upward COM velocity, COM rise, flight, then recovery',flush=True)
    search.main()
