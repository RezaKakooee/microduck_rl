"""Procedural meshes for the duck film: ring, egg shells, hearts.

MJCF can carry a mesh inline as `vertex="..." face="..."`, so nothing here needs
an STL on disk. Every shape is visual only (contype/conaffinity 0) -- the story
is animated kinematically, so none of it is ever asked to collide.
"""

import numpy as np


def _fan(ring_idx, apex_idx):
    """Triangles closing a loop of vertices onto one apex vertex."""
    n = len(ring_idx)
    return [(apex_idx, ring_idx[i], ring_idx[(i + 1) % n]) for i in range(n)]


def _quad_strip(a, b):
    """Triangles between two equal-length vertex loops."""
    n = len(a)
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(a[i], b[i], b[j]), (a[i], b[j], a[j])]
    return f


def torus(radius=0.008, tube=0.0016, nu=40, nv=12):
    """The engagement ring."""
    verts, faces = [], []
    for i in range(nu):
        u = 2 * np.pi * i / nu
        cu, su = np.cos(u), np.sin(u)
        for j in range(nv):
            v = 2 * np.pi * j / nv
            r = radius + tube * np.cos(v)
            verts.append((r * cu, r * su, tube * np.sin(v)))
    idx = lambda i, j: (i % nu) * nv + (j % nv)
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [(a, b, c), (a, c, d)]
    return np.array(verts), np.array(faces)


def egg_profile(t, a, c):
    """Radius of an egg at height parameter t in [-1, 1]: an ellipse, fattened
    at the bottom and drawn in at the top, which is what makes it read as an egg
    rather than a pill."""
    r = a * np.sqrt(np.maximum(0.0, 1.0 - t * t))
    return r * (1.0 - 0.16 * t)


def egg_shell(a=0.030, c=0.040, split=0.18, teeth=7, jag=0.10, nu=48, nv=20, top=True):
    """One half of a cracked egg shell, split along a zigzag.

    `split` is where the crack sits (in t = z/c), `teeth` how many zigzag teeth
    run around it, `jag` their amplitude. Top and bottom halves are generated
    from the same crack curve, so they interlock exactly when closed.
    """
    crack = lambda u: split + jag * np.sin(teeth * u)

    verts, faces = [], []
    rim = []
    for i in range(nu):
        u = 2 * np.pi * i / nu
        t0 = crack(u)
        ts = np.linspace(t0, 1.0 if top else -1.0, nv)
        col = []
        for t in ts:
            r = egg_profile(t, a, c)
            col.append(len(verts))
            verts.append((r * np.cos(u), r * np.sin(u), c * t))
        rim.append(col)

    for i in range(nu):
        col_a, col_b = rim[i], rim[(i + 1) % nu]
        for j in range(nv - 1):
            faces += [(col_a[j], col_b[j], col_b[j + 1]),
                      (col_a[j], col_b[j + 1], col_a[j + 1])]

    # Close the pole (the last ring collapses to a point) and cap the crack rim
    # with a fan, so each half is a solid piece rather than an open surface.
    pole = len(verts)
    verts.append((0.0, 0.0, c * (1.0 if top else -1.0)))
    faces += _fan([col[-1] for col in rim], pole)

    hub = len(verts)
    verts.append((0.0, 0.0, c * split))
    ring0 = [col[0] for col in rim]
    faces += _fan(ring0, hub) if top else _fan(ring0[::-1], hub)
    return np.array(verts), np.array(faces)


def heart(size=0.02, thick=0.25, n=48):
    """A heart, extruded from the classic parametric curve."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    z = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x, z = x / 17.0 * size, z / 17.0 * size
    d = size * thick / 2

    verts = [(xi, -d, zi) for xi, zi in zip(x, z)]
    verts += [(xi, d, zi) for xi, zi in zip(x, z)]
    front = list(range(n))
    back = list(range(n, 2 * n))
    cf, cb = 2 * n, 2 * n + 1
    verts += [(0.0, -d, 0.0), (0.0, d, 0.0)]
    faces = _fan(front[::-1], cf) + _fan(back, cb) + _quad_strip(front, back)
    return np.array(verts), np.array(faces)


def mesh_xml(name, verts, faces, scale=1.0):
    v = " ".join("%.5f" % x for x in (np.asarray(verts).ravel() * scale))
    f = " ".join(str(int(i)) for i in np.asarray(faces).ravel())
    return f'<mesh name="{name}" vertex="{v}" face="{f}"/>'
