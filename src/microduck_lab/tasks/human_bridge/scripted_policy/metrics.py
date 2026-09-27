"""The requested numerical checks, separate from landing preferences."""
import numpy as np

from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.scripted_policy.story import DROP_Z


def nonfoot_support(duck):
    world, model = duck.world, duck.model
    feet = set(model.site_bodyid[list(duck.foot_sites)])
    nonfeet = {g for g in duck.solid if model.geom_bodyid[g] not in feet}
    other = set(range(model.ngeom)) - set(duck.geoms)
    return sum(c[2] for c in world.contacts(nonfeet, other))


def audit(trace, *, drop_z=None, far_edge=None):
    """Check the task limits and the uninterrupted final clearance time."""
    trace = np.asarray(trace)
    drop_z = DROP_Z if drop_z is None else drop_z
    far_edge = L.far_edge if far_edge is None else far_edge
    if len(trace) == 0:
        return dict(task_limits=False, final_clearance_seconds=0.)
    crossing = trace[:, 0] >= 5.02
    safe = ((trace[:, 4] > .5) & (trace[:, 3] >= drop_z)
            & (~crossing | (np.abs(trace[:, 6]) < .35)))
    clear = trace[:, 7] > far_edge + .005
    good = safe & clear
    bad = np.flatnonzero(~good)
    first = bad[-1]+1 if len(bad) else 0
    duration = float(trace[-1, 0]-trace[first, 0]) if first < len(trace) else 0.
    return dict(task_limits=bool(safe.all() and clear[-1]),
                final_clearance_seconds=duration)
