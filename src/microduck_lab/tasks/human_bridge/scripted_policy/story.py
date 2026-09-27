"""Local camera and judging helpers, copied for this controller."""
import math
import mujoco
import numpy as np
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
HEADING = -math.pi / 2
DROP_Z = min(L.near_z, L.far_z) + .05

def bridge_state(duck, table, mid_x=0.05, stand_height=0.122):
    """The 6 numbers the bridge policy reads in its body-pose slots; the
    same formulas as `rl.mdp_bridge.bridge_state` (training)."""
    sx, sz = table
    h = lambda xq: float(np.interp(xq, sx, sz))
    x, y, z = duck.pos()
    err = math.atan2(math.sin(duck.yaw() - HEADING), math.cos(duck.yaw() - HEADING))
    here = h(x)
    return np.array([(x - mid_x) / 0.3, y / 0.05, err / 0.3,
                     (h(x + 0.05) - here) / 0.03, (h(x + 0.10) - here) / 0.03,
                     (here + stand_height - z) / 0.03], dtype=np.float32)

def rearmost(duck):
    """x of the back-most point of her solid geoms (bounding spheres)."""
    m, d = duck.model, duck.data
    g = np.asarray(duck.solid)
    return float(np.min(d.geom_xpos[g, 0] - m.geom_rbound[g]))

class Judge:
    """The strict rules for "she WALKED across him". The story and the test
    benches (local_storage/hb_dev) both use this class, so every attempt is
    scored the same way. Call `tick` every 50 Hz tick while she crosses and
    stands on the far ledge, and `finish` once all of her is past its edge.
    A bench without him passes `he=None` and the geoms that stand in for his
    back as `bridge` (for example a beam).

    * Her trunk stays within ~25 deg of upright (up >= MIN_UP).
    * Only her feet touch anything: the soles, and the foot shells whose
      bottom is 2 mm above them. Knees, hips, body or head = fail.
    * Her feet touch only him, her ledge (near_top) and the far ledge
      (far_top). The side ledges, the notch floors and the yard fail: she may
      not walk round him or step down beside him.
    * Her trunk stays above DROP_Z, and he stays lying (|up| <= 0.35).
    * She really steps. By `finish`, each foot must have landed on him at
      least MIN_STEPS times, each time after at least MIN_AIR_TICKS in the
      air and a lift of at least MIN_LIFT. A foot slid across him fails.

    The old rules (fall = tilt past 60 deg, across = all of her past the far
    edge) passed `scripted_try96`: she tipped over onto him and fell across.
    Until 2026-09-26 her thighs and neck could not collide at all (see
    `world.arm`), so "only her soles touched" was not checked for them.
    """

    MIN_UP = 0.9
    MIN_LIFT = 0.005
    MIN_AIR_TICKS = 2
    MIN_STEPS = 3
    LEDGES = ("near_top", "far_top")

    def __init__(self, world, she, he=None, bridge=None):
        m = world.model
        self.m, self.d, self.she, self.he = m, world.data, she, he
        mesh = lambda g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or ""
                          if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else "")
        shells = [{g for g in she.solid if mesh(g).endswith("_foot_" + side)} for side in ("left", "right")]
        assert all(len(s) == 1 for s in shells), shells
        self.foot = [shells[k] | {she.sole_geoms[k]} for k in (0, 1)]
        self.shells = shells[0] | shells[1]
        self.rest = set(she.solid) - self.foot[0] - self.foot[1]
        self.him = set(he.solid) if bridge is None else set(bridge)
        self.set = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        self.allowed = self.him | {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.LEDGES}
        self.air = [0, 0]
        self.z0 = [0.0, 0.0]
        self.peak = [0.0, 0.0]
        self.steps = [0, 0]           # landings on him after a real swing
        self.shell_ticks = 0          # ticks a foot shell (not a sole) touched something

    def name(self, g):
        m = self.m
        if m.geom_bodyid[g] == 0:
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])

    def tick(self):
        """The first broken rule this tick, as text, or None."""
        she, he, d = self.she, self.he, self.d
        x = she.pos()[0]
        if she.up() < self.MIN_UP:
            return f"she tipped past 25 deg at x={x:.3f} m (not walking)"
        touching = [set(), set()]
        shell = False
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.rest and (b in self.him or b in self.set):
                    return f"her {self.name(a)} touched {self.name(b)} at x={x:.3f} m (not walking)"
                for k in (0, 1):
                    if a in self.foot[k]:
                        if b not in self.allowed:
                            side = ("left", "right")[k]
                            return (f"her {side} foot touched {self.name(b)} at x={x:.3f} m "
                                    "(only his back and the two ledges count)")
                        touching[k].add(b)
                        shell |= a in self.shells
        self.shell_ticks += shell
        if she.pos()[2] < DROP_Z:
            return f"she stepped off him at x={x:.3f}, y={she.pos()[1]:.3f} m"
        if he is not None and abs(he.up()) > 0.35:
            return "he was knocked loose"
        for k in (0, 1):
            z = float(d.site_xpos[she.foot_sites[k], 2])
            if not touching[k]:
                if self.air[k] == 0:
                    self.z0[k] = self.peak[k] = z
                self.air[k] += 1
                self.peak[k] = max(self.peak[k], z)
            else:
                if (self.air[k] >= self.MIN_AIR_TICKS and self.peak[k] - self.z0[k] >= self.MIN_LIFT
                        and touching[k] & self.him):
                    self.steps[k] += 1
                self.air[k] = 0
        return None

    def finish(self):
        """Call once all of her is past the far edge: did she step on him?"""
        for k, side in enumerate(("left", "right")):
            if self.steps[k] < self.MIN_STEPS:
                return (f"she did not step across: her {side} foot landed on him {self.steps[k]} "
                        f"times after a real swing (need {self.MIN_STEPS})")
        return None

    def report(self):
        return dict(steps_on_him=list(self.steps), foot_shell_ticks=int(self.shell_ticks))

class Camera:
    """In front of her (she faces -y), following her across, drifting a little."""

    def __init__(self, width, height):
        import mujoco
        self.mujoco = mujoco
        self.width, self.height = width, height
        self.renderer = None

    def frame(self, story):
        mj = self.mujoco
        if self.renderer is None:
            self.renderer = mj.Renderer(story.world.model, self.height, self.width)
        t = story.world.t
        c = mj.MjvCamera()
        x = float(np.clip(story.she.pos()[0], -0.2, 0.35))
        c.lookat[:] = (0.05 + 0.6 * (x - 0.05), 0.0, L.near_z)
        c.distance = 1.0
        c.azimuth = 90.0 - 25.0 * np.cos(t / 12.0)
        c.elevation = -22.0
        self.renderer.update_scene(story.world.data, c)
        return self.renderer.render()
