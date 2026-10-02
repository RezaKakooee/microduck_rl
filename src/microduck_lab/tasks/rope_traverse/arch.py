"""Hang the duck upside down from the rope by its two legs.

Each leg is folded into a hook ("arch"): the thigh on one side of the rope,
the shin across the top, the foot on the other side. The robot hangs below
the rope with its head down. Folding the knee further closes the hook into
a ring around the rope. No friction is needed to hang: the rope sits under
the shin, between two walls.

The rope runs along the robot's lateral (y) axis, so the robot moves
sideways along it. The mouth is not used: the 92 mm head does not fit
between the two hooks on the rope.

`place` is setup only. After `World.start()` only motor targets move the robot.
"""
import numpy as np
import mujoco

from microduck_lab.tasks.rope_traverse.fit import HOME, I, balance_about_rope

# Leg-arch joint targets, left-leg sign (the right leg is mirrored).
# hip pitch: thigh parallel to the trunk axis; knee: shin at 90 deg;
# ankle: foot parallel to the thigh. Found from the collision hulls.
ARCH = dict(hip_pitch=-1.0195, knee=.6055, ankle=.0556, hip_roll=0.)
# XL330 firmware P gain for this skill. 200 is the walking-policy value
# (0.55 Nm/rad); 800 (2.2 Nm/rad) lets a hip roll push a leg along the rope.
# The runtime already raises the gain in standby (~730). Stall torque is unchanged.
KP = 800.
# Body parts used to sort rope contacts.
LEGS = {'left': ('hip_l', 'upper_leg_left', 'leg', 'ankle_left', 'yaw2roll'),
        'right': ('hip_l_2', 'upper_leg_right', 'leg_2', 'ankle_right', 'bearing_roll')}
HEAD = ('jaw_soft', 'mouth_jaw', 'yaw_roll_motion', 'neck_pitch', 'neck')
# A trunk-frame point inside both leg hooks (between thigh and foot wall).
POCKET_TRUNK = np.array([.034, 0., -.038])


def leg_targets(q, side, **joints):
    """Set left-sign joint values on one leg of a 14-joint target."""
    q = np.array(q, float)
    for name, value in joints.items():
        q[I[f'{side}_{name}']] = value if side == 'left' else -value
    return q


def hang_pose():
    q = HOME.copy()
    for side in ('left', 'right'):
        q = leg_targets(q, side, **ARCH)
    return q


def place(w, q, x=0., gap=.0008):
    """Setup only: robot upside down, lateral axis along the rope (world x),
    the rope inside both leg hooks, `gap` metres above the rope, then turned
    about the rope so the centre of mass hangs below it."""
    if w.running:
        raise RuntimeError('place() is setup only')
    m, d, k = w.model, w.data, w.ducks['she']
    c = w.rope_config
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])     # robot y -> world x, robot z -> down
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, R.ravel())
    d.qpos[k.adr + 3:k.adr + 7] = quat
    d.qpos[k.adr:k.adr + 3] = [x, 0, .8]
    d.qpos[k.qidx] = q; k.hold(q)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    pocket = d.xpos[k.trunk] + d.xmat[k.trunk].reshape(3, 3) @ POCKET_TRUNK
    rope_z = c.height - c.sag * (1 - (2 * x / c.span) ** 2)
    d.qpos[k.adr + 1] -= pocket[1]
    d.qpos[k.adr + 2] += rope_z - pocket[2]
    mujoco.mj_forward(m, d)
    for _ in range(30):
        g = rope_gap(w)
        if abs(g - gap) < 1e-5:
            break
        d.qpos[k.adr + 2] += gap - g
        mujoco.mj_forward(m, d)
    for _ in range(4):
        balance_about_rope(w)
    # balance_about_rope turns about the rope's middle height; off centre the rope
    # sits higher, so set the gap again.
    for _ in range(30):
        g = rope_gap(w)
        if abs(g - gap) < 1e-5:
            break
        d.qpos[k.adr + 2] += gap - g
        mujoco.mj_forward(m, d)
    return rope_gap(w)


def rope_gap(w):
    """Smallest distance between any robot collision geom and the rope."""
    m, d, k = w.model, w.data, w.ducks['she']
    return min(mujoco.mj_geomDistance(m, d, g, r, .05, None) for g in k.solid for r in w.rope_geoms)


class Loads:
    """Normal contact force on the rope from each body group, and from the floor and posts."""

    def __init__(self, w):
        m = w.model
        self.w = w
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_' + n)
        self.part = {bid(n): side for side, names in LEGS.items() for n in names}
        self.part.update({bid(n): 'head' for n in HEAD})
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.posts = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('post_')}
        self.solid = set(w.ducks['she'].solid)
        self.f = np.zeros(6)

    def __call__(self):
        m, d, w = self.w.model, self.w.data, self.w
        out = dict(left=0., right=0., head=0., other=0., floor=0., post=0.)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in w.rope_geoms and c.geom2 in self.solid: g = c.geom2
            elif c.geom2 in w.rope_geoms and c.geom1 in self.solid: g = c.geom1
            elif self.floor in (c.geom1, c.geom2) and (c.geom1 in self.solid or c.geom2 in self.solid):
                mujoco.mj_contactForce(m, d, i, self.f); out['floor'] += self.f[0]; continue
            elif (c.geom1 in self.posts or c.geom2 in self.posts) and (c.geom1 in self.solid or c.geom2 in self.solid):
                mujoco.mj_contactForce(m, d, i, self.f); out['post'] += self.f[0]; continue
            else:
                continue
            mujoco.mj_contactForce(m, d, i, self.f)
            out[self.part.get(int(m.geom_bodyid[g]), 'other')] += self.f[0]
        return out


def rope_arc(w, point):
    """Arc length from the first rope anchor to the rope point nearest `point`.
    For judging only; the controller never reads the rope state."""
    m, d = w.model, w.data
    geoms = sorted(w.rope_geoms)
    ends = [d.geom_xpos[g] - d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1] for g in geoms]
    ends.append(d.geom_xpos[geoms[-1]] + d.geom_xmat[geoms[-1]].reshape(3, 3)[:, 2] * m.geom_size[geoms[-1]][1])
    pts = np.array(ends); seg = np.diff(pts, axis=0); L = np.linalg.norm(seg, axis=1)
    u = np.clip(np.einsum('ij,ij->i', point - pts[:-1], seg) / L ** 2, 0, 1)
    dist = np.linalg.norm(pts[:-1] + u[:, None] * seg - point, axis=1)
    i = int(np.argmin(dist))
    return float(np.r_[0, np.cumsum(L)][i] + u[i] * L[i])
