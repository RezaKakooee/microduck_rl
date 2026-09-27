"""Local baked surface for the optional bridge observations."""
import json
from pathlib import Path
import mujoco
from microduck_lab import paths
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
POSE_FILE = Path(__file__).with_name("bridge_pose.json")
START_X = -.25

def add_bridge(spec: mujoco.MjSpec) -> None:
    """Ledges and his baked body, as static geoms on the terrain body."""
    terrain = spec.body("terrain")
    for box in L.design().boxes:
        c = [(a + b) / 2 for a, b in zip(box.lo, box.hi)]
        h = [(b - a) / 2 for a, b in zip(box.lo, box.hi)]
        terrain.add_geom(name=f"bridge_{box.name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=c, size=h, rgba=box.rgba)
    pose = json.loads(Path(POSE_FILE).read_text())
    added = set()
    for i, g in enumerate(pose["geoms"]):
        mesh = f"brother_{g['mesh']}"
        if mesh not in added:
            spec.add_mesh(name=mesh, file=str(paths.REPO / g["file"]), scale=g["scale"])
            added.add(mesh)
        terrain.add_geom(name=f"brother_{i}_{g['mesh']}", type=mujoco.mjtGeom.mjGEOM_MESH,
                         meshname=mesh, pos=g["pos"], quat=g["quat"],
                         friction=g["friction"], rgba=(0.30, 0.53, 0.93, 1))

def surface_table(step=0.005):
    """Highest point under her stance at each trunk x, on the baked set.

    Collision uses each mesh's convex hull, whose top is the highest mesh
    vertex, so this reads vertices, not rays: rays pass straight through his
    hollow shell meshes and put her spawn inside his trunk. Per x column:
    the highest vertex or box top across her sideways sole (|y| < 25 mm), then
    the max over her stance (feet +-42 mm, soles +-21 mm)."""
    import numpy as np
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    add_bridge(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    P = np.vstack(pts)
    P = P[np.abs(P[:, 1]) < 0.025]
    boxes = [b for b in L.design().boxes if b.lo[1] <= 0.0 <= b.hi[1]]
    xs = np.arange(START_X - 0.10, L.far_edge + 0.30, step)
    top = np.zeros(len(xs))
    for i, x in enumerate(xs):
        col = P[np.abs(P[:, 0] - x) <= step / 2, 2]
        h = [b.hi[2] for b in boxes if b.lo[0] <= x <= b.hi[0]]
        top[i] = max([*h, *(col.tolist() or [0.0])])
    k = int(round(0.063 / step))
    stance = np.array([top[max(0, i - k):i + k + 1].max() for i in range(len(top))])
    return tuple(round(float(v), 4) for v in xs), tuple(round(float(v), 4) for v in stance)
