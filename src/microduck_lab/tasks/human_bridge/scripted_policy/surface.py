"""Vertical intersections with convex mesh hulls, as used for collisions."""
import numpy as np
from scipy.spatial import ConvexHull

from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L


class Surface:
    def __init__(self, duck):
        self.duck = duck
        m = duck.model
        self.hulls = {}
        for g in duck.solid:
            mid = int(m.geom_dataid[g])
            if mid not in self.hulls:
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                self.hulls[mid] = ConvexHull(m.mesh_vert[a:a+n]).equations

    def height(self, x, y):
        h = 0.
        for box in L.design().boxes:
            if box.lo[0] <= x <= box.hi[0] and box.lo[1] <= y <= box.hi[1]:
                h = max(h, box.hi[2])
        m, d = self.duck.model, self.duck.data
        for g in self.duck.solid:
            if np.linalg.norm(d.geom_xpos[g, :2] - [x, y]) > m.geom_rbound[g]:
                continue
            eq = self.hulls[int(m.geom_dataid[g])]
            normal = eq[:, :3] @ d.geom_xmat[g].reshape(3, 3).T
            offset = eq[:, 3] - normal @ d.geom_xpos[g]
            rhs = -offset - normal[:, 0]*x - normal[:, 1]*y
            nz = normal[:, 2]
            if np.any(rhs[np.abs(nz) < 1e-9] < -1e-8):
                continue
            upper = np.min(rhs[nz > 1e-9] / nz[nz > 1e-9], initial=np.inf)
            lower = np.max(rhs[nz < -1e-9] / nz[nz < -1e-9], initial=-np.inf)
            if lower <= upper + 1e-8:
                h = max(h, upper)
        return float(h)
