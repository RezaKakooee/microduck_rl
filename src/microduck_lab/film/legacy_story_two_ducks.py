#!/usr/bin/env python3
"""Two microducks: a proposal, a kiss, two eggs, two ducklings.

A rendered short film, six scenes:

    1. meet     the blue duck walks up to the pink one, a ring in his beak
    2. propose  he kneels, offers the ring, she says yes and takes it
    3. kiss     he stands, they lean in, beaks touching, hearts
    4. eggs     she sits in the nest and lays two eggs; he kisses her head
    5. brood    she sits on the eggs, then they swap and he sits on them
    6. hatch    both eggs crack open and a blue and a pink duckling come out

THIS IS ANIMATION, NOT SIMULATION. Nothing here is a policy, a controller or a
physics rollout: `mj_step` is never called and no reward, servo torque or
contact force takes part. Every frame writes `qpos` directly and calls
`mj_kinematics`, the way a keyframe animator poses a rig. The result says
nothing about what the robot can do -- it is a picture of a story, made from
the real MJCF so that the ducks are the actual robot and not a cartoon.

What IS taken from the real model, because it costs nothing to be honest about
the geometry:

* the ducks are `robot_allcollisions_mouth.xml`, so the beak really opens
  (the 15th servo, the one no policy drives) and the ring hangs off the real
  `mouth_tip` site;
* poses are built from the STAND and SIT keyframes in `scene.xml`, in the
  joint order the whole repo uses (left leg, neck/head, right leg), with the
  left/right sign mirror the model has;
* every pose is GROUNDED by forward kinematics -- the trunk height is solved
  each frame so the lowest point of the duck rests on the floor -- and the
  kiss distance, the head-kiss reach and the ring's seat on the beak are all
  solved from FK too, never eyeballed;
* the ducklings are the same MJCF scaled to a third, so a duckling is a
  microduck.

Rendering needs a GPU (EGL); the geometry solving does not:

        bash src/microduck_lab/render/render.sh story

    # or directly
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \\
        src/microduck_lab/film/legacy_story_two_ducks.py --out videos/love_story/story

    src/microduck_lab/film/legacy_story_two_ducks.py --check          # FK checks, no GL needed
    src/microduck_lab/film/legacy_story_two_ducks.py --frames 3.0 12.5 --out videos/love_story/story  # stills
"""

import argparse
import math
import os
import sys

import numpy as np

import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUCK_XML = os.path.join(
    REPO, "src/microduck_lab/models/robot_allcollisions_mouth.xml"
)

# ---------------------------------------------------------------------------
# The cast
# ---------------------------------------------------------------------------

PINK = np.array([1.00, 0.42, 0.62])      # she
BLUE = np.array([0.28, 0.55, 0.95])      # he
BEAK = np.array([0.98, 0.71, 0.05])      # beak and feet, as the model has them
DUCKLING_SCALE = 0.33

# Materials that keep their own colour whatever duck they belong to: the beak
# and feet (already orange in the MJCF) and everything dark (servos, lens,
# eyes). Anything else is a shell, and a shell takes the duck's colour.
KEEP_ORANGE = 0.55      # min r-b spread to read as "this part is already orange"
KEEP_DARK = 0.34        # max luma to read as "this part is a servo, leave it"

# ---------------------------------------------------------------------------
# Procedural meshes: the eggs, the ring and the hearts
# ---------------------------------------------------------------------------


def egg_profile(u, length, width, point):
    """Radius and height of an egg surface of revolution.

    `u` runs 0 (bottom pole) to 1 (top pole). `point` > 0 narrows the top,
    which is what makes an egg an egg rather than an ellipsoid.
    """
    th = (u - 0.5) * math.pi
    z = 0.5 * length * math.sin(th)
    r = width * math.cos(th) * (1.0 - point * math.sin(th))
    return max(r, 0.0), z


def shell_piece(u_lo, u_hi, length, width, point, thickness,
                n_u=18, n_phi=48):
    """A closed thin shell between two boundary curves of an egg.

    `u_lo(phi)` and `u_hi(phi)` give the piece's lower and upper edge as a
    function of the angle, so a ragged crack line is just a ragged `u_lo`.
    The piece is built as an outer surface, an inner surface shrunk towards
    the egg's axis, and a rim joining them, which makes it watertight -- a
    hollow half-shell a duckling can sit inside, rather than a solid lump.

    A boundary that touches a pole (u = 0 or 1) collapses to a point, so no
    rim is emitted there and the quads become triangles on an apex vertex.
    """
    k = 1.0 - thickness / width          # inner surface, as a shrunk profile
    verts, faces = [], []

    def add(p):
        verts.append(p)
        return len(verts) - 1

    def point_at(u, phi, inner):
        r, z = egg_profile(u, length, width, point)
        if inner:
            r, z = r * k, z * k
        return [r * math.cos(phi), r * math.sin(phi), z]

    phis = [2.0 * math.pi * j / n_phi for j in range(n_phi)]
    lo = [u_lo(p) for p in phis]
    hi = [u_hi(p) for p in phis]
    pole_lo = max(lo) <= 1e-6
    pole_hi = min(hi) >= 1.0 - 1e-6

    # Vertex grid: [surface][i][j], surface 0 outer, 1 inner.
    grid = [[[0] * n_phi for _ in range(n_u + 1)] for _ in range(2)]
    for s in (0, 1):
        for i in range(n_u + 1):
            f = i / n_u
            for j in range(n_phi):
                u = lo[j] + f * (hi[j] - lo[j])
                grid[s][i][j] = add(point_at(u, phis[j], s == 1))
    apex_lo = add(point_at(0.0, 0.0, False)) if pole_lo else None
    apex_hi = add(point_at(1.0, 0.0, False)) if pole_hi else None

    def quad(a, b, c, d):
        faces.append([a, b, c])
        faces.append([a, c, d])

    for i in range(n_u):
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            if i == 0 and pole_lo:
                faces.append([apex_lo, grid[0][1][jn], grid[0][1][j]])
                faces.append([apex_lo, grid[1][1][j], grid[1][1][jn]])
                continue
            if i == n_u - 1 and pole_hi:
                faces.append([apex_hi, grid[0][n_u - 1][j], grid[0][n_u - 1][jn]])
                faces.append([apex_hi, grid[1][n_u - 1][jn], grid[1][n_u - 1][j]])
                continue
            quad(grid[0][i][j], grid[0][i][jn], grid[0][i + 1][jn], grid[0][i + 1][j])
            quad(grid[1][i][j], grid[1][i + 1][j], grid[1][i + 1][jn], grid[1][i][jn])
    if not pole_lo:
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            quad(grid[0][0][j], grid[1][0][j], grid[1][0][jn], grid[0][0][jn])
    if not pole_hi:
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            quad(grid[0][n_u][jn], grid[1][n_u][jn], grid[1][n_u][j], grid[0][n_u][j])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def heart_mesh(size=1.0, puff=0.34, n_out=64, n_ring=8):
    """A puffy 3D heart, standing in the x-z plane and bulging along y.

    The outline is the classic sin-cubed heart curve; the surface is that
    outline shrunk towards its centre in rings, lifted out of the plane on a
    quarter-sine, mirrored front and back. Flat hearts vanish when the camera
    swings round, and these do not.
    """
    outline = []
    for i in range(n_out):
        t = 2.0 * math.pi * i / n_out
        x = 16.0 * math.sin(t) ** 3
        z = (13.0 * math.cos(t) - 5.0 * math.cos(2 * t)
             - 2.0 * math.cos(3 * t) - math.cos(4 * t))
        outline.append((x / 17.0 * size, z / 17.0 * size))
    cx = sum(p[0] for p in outline) / n_out
    cz = sum(p[1] for p in outline) / n_out

    verts, faces = [], []
    for s in (1.0, -1.0):
        base = len(verts)
        for k in range(n_ring):
            a = 0.5 * math.pi * k / n_ring
            shrink, y = math.cos(a), s * puff * size * math.sin(a)
            for (x, z) in outline:
                verts.append([cx + (x - cx) * shrink, y, cz + (z - cz) * shrink])
        tip = len(verts)
        verts.append([cx, s * puff * size, cz])
        for k in range(n_ring - 1):
            for j in range(n_out):
                jn = (j + 1) % n_out
                a = base + k * n_out + j
                b = base + k * n_out + jn
                c = base + (k + 1) * n_out + jn
                d = base + (k + 1) * n_out + j
                if s > 0:
                    faces += [[a, b, c], [a, c, d]]
                else:
                    faces += [[a, c, b], [a, d, c]]
        last = base + (n_ring - 1) * n_out
        for j in range(n_out):
            jn = (j + 1) % n_out
            faces.append([tip, last + j, last + jn] if s > 0
                         else [tip, last + jn, last + j])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

WORLD_XML = """
<mujoco model="two_duck_story">
  <compiler angle="radian"/>
  <statistic extent="1.1" center="0 0 0.18"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.38 0.38 0.40" specular="0.12 0.12 0.12"/>
    <rgba haze="0.86 0.90 0.95 1"/>
    <global azimuth="270" elevation="-12" offwidth="1600" offheight="1200"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.005" zfar="40" shadowclip="2" shadowscale="1.2"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.66 0.92" rgb2="0.98 0.88 0.86"
             width="512" height="3072"/>
    <texture type="2d" name="lawn" builtin="checker" mark="edge"
             rgb1="0.44 0.63 0.36" rgb2="0.38 0.57 0.32" markrgb="0.52 0.71 0.44"
             width="400" height="400"/>
    <material name="lawn" texture="lawn" texuniform="true" texrepeat="9 9" reflectance="0.03"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0.6 -0.8 2.2" dir="-0.25 0.35 -1" directional="true"
           diffuse="0.55 0.54 0.5" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-1.2 -1.4 0.9" dir="0.6 0.7 -0.4" directional="true"
           diffuse="0.22 0.22 0.28" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="lawn"/>
  </worldbody>
</mujoco>
"""

EGG_LENGTH, EGG_WIDTH, EGG_POINT = 0.046, 0.0175, 0.24
EGG_SHELL = 0.0016
CRACK_BASE, CRACK_TEETH, CRACK_AMP = 0.52, 7, 0.075
N_HEARTS = 18
NEST = np.array([0.0, 0.0])          # the nest sits at the origin
STAGE = np.array([0.0, -0.52])       # the proposal happens in front of it


def crack(phi):
    """The ragged line where an egg comes apart -- a triangle wave in phi."""
    saw = (phi / (2.0 * math.pi) * CRACK_TEETH) % 1.0
    return CRACK_BASE + CRACK_AMP * (2.0 * abs(saw - 0.5) - 0.5)


def scale_spec(spec, s):
    """Scale a whole model about its root: lengths by s, mass by s^3.

    A duckling is the microduck at a third of the size, which means every body
    offset, joint anchor, geom size and mesh scale, not just the meshes -- a
    mesh-only scale explodes the robot into a cloud of shrunk parts.
    """
    for b in spec.bodies:
        b.pos = np.asarray(b.pos) * s
        b.ipos = np.asarray(b.ipos) * s
        b.mass = float(b.mass) * s ** 3
        b.inertia = np.asarray(b.inertia) * s ** 5
        for g in b.geoms:
            g.pos = np.asarray(g.pos) * s
            g.size = np.asarray(g.size) * s
            ft = np.asarray(g.fromto)
            if np.all(np.isfinite(ft)) and np.max(np.abs(ft)) < 1e6:
                g.fromto = ft * s
        for si in b.sites:
            si.pos = np.asarray(si.pos) * s
            si.size = np.asarray(si.size) * s
        for j in b.joints:
            j.pos = np.asarray(j.pos) * s
    for me in spec.meshes:
        me.scale = np.asarray(me.scale) * s
    return spec


def _prop(world, name, pos=(0, 0, -1.0)):
    """A body that the animation moves by hand: free joint, no dynamics."""
    b = world.worldbody.add_body(name=name, pos=list(pos))
    b.add_freejoint(name=name + "_free")
    b.explicitinertial = True
    b.mass = 0.01
    b.inertia = [1e-6, 1e-6, 1e-6]
    return b


def _visual(body, **kw):
    """A geom that is only ever looked at: no collisions, no contact class."""
    kw.setdefault("contype", 0)
    kw.setdefault("conaffinity", 0)
    kw.setdefault("group", 2)
    kw.setdefault("mass", 0.0)
    return body.add_geom(**kw)


def build_world(duckling_scale=DUCKLING_SCALE):
    """Assemble the scene: two ducks, two ducklings, a nest, a ring, eggs."""
    world = mujoco.MjSpec.from_string(WORLD_XML)

    # --- meshes ---------------------------------------------------------
    # A ring 32 mm across: big enough to read as a ring at duck scale, small
    # enough to sit in a beak that is 30 mm wide.
    ring = world.add_mesh(name="ring")
    ring.make_supertorus(48, 0.176, 1.0, 1.0)
    ring.scale = np.array([0.0136, 0.0136, 0.0136])

    nest = world.add_mesh(name="nest")
    nest.make_supertorus(64, 0.34, 1.0, 0.75)
    nest.scale = np.array([0.105, 0.105, 0.072])

    for nm, (lo, hi) in {
        "egg_bot": (lambda p: 0.0, crack),
        "egg_top": (crack, lambda p: 1.0),
    }.items():
        v, f = shell_piece(lo, hi, EGG_LENGTH, EGG_WIDTH, EGG_POINT, EGG_SHELL)
        world.add_mesh(name=nm, uservert=v.flatten(), userface=f.flatten())

    hv, hf = heart_mesh(size=0.030)
    world.add_mesh(name="heart", uservert=hv.flatten(), userface=hf.flatten())

    # --- the nest, the only prop that never moves -----------------------
    static = world.worldbody.add_body(name="nest", pos=[NEST[0], NEST[1], 0.0])
    _visual(static, name="nest_rim", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="nest", rgba=[0.46, 0.33, 0.18, 1])
    rng = np.random.default_rng(7)
    for i in range(46):                       # straw, thrown on at angles
        a = 2 * math.pi * i / 46 + rng.uniform(-0.14, 0.14)
        r = 0.100 + rng.uniform(-0.022, 0.026)
        ln = rng.uniform(0.035, 0.075)
        tilt = rng.uniform(-0.6, 0.6)
        c, s = math.cos(a), math.sin(a)
        p0 = np.array([r * c, r * s, 0.018 + rng.uniform(-0.010, 0.026)])
        d = np.array([-s * math.cos(tilt), c * math.cos(tilt), math.sin(tilt) * 0.45])
        p1 = p0 + d * ln
        tan = rng.uniform(0.0, 0.18)
        _visual(static, name=f"straw{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=list(p0) + list(p1), size=[0.0026, 0, 0],
                rgba=[0.62 + tan, 0.46 + tan, 0.22 + tan * 0.6, 1])

    # --- the ring, the eggs, the hearts ---------------------------------
    b = _prop(world, "ring")
    _visual(b, name="ring_band", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="ring",
            rgba=[1.0, 0.80, 0.22, 1])
    # The stone sits on the band at local +y, so a quarter turn about x
    # stands the ring up with the stone on top and its face to the camera.
    _visual(b, name="ring_gem", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.0042],
            pos=[0, 0.0136, 0], rgba=[0.82, 0.95, 1.0, 0.9])

    for e in (0, 1):
        for half, mesh in (("bot", "egg_bot"), ("top", "egg_top")):
            b = _prop(world, f"egg{e}_{half}")
            _visual(b, name=f"egg{e}_{half}_g", type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh, rgba=[0.98, 0.94, 0.88, 1])

    for i in range(N_HEARTS):
        b = _prop(world, f"heart{i}")
        _visual(b, name=f"heart{i}_g", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="heart", rgba=[1.0, 0.30, 0.45, 0.0])

    # --- the ducks ------------------------------------------------------
    for prefix, scale in (("blue_", 1.0), ("pink_", 1.0),
                          ("bblue_", duckling_scale), ("bpink_", duckling_scale)):
        spec = mujoco.MjSpec.from_file(DUCK_XML)
        if scale != 1.0:
            scale_spec(spec, scale)
        frame = world.worldbody.add_frame(pos=[0, 0, -2.0])
        world.attach(spec, prefix=prefix, frame=frame)

    model = world.compile()
    for prefix, colour in (("blue_", BLUE), ("pink_", PINK),
                           ("bblue_", BLUE), ("bpink_", PINK)):
        paint(model, prefix, colour)
    return model


def paint(model, prefix, colour):
    """Give one duck its colour, part by part.

    The MJCF is a real CAD export, so each part already carries a material:
    white shells, orange beak and feet, dark servos. Shells take the duck's
    colour scaled by how light they were, which keeps the shading the export
    has; the beak, the feet and everything dark are left as they are.
    """
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
    mine = set()
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == root:
                mine.add(b)
                break
            p = model.body_parentid[p]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in mine:
            continue
        mat = model.geom_matid[g]
        base = model.mat_rgba[mat][:3] if mat >= 0 else model.geom_rgba[g][:3]
        luma = float(base @ np.array([0.30, 0.59, 0.11]))
        if float(base[0] - base[2]) > KEEP_ORANGE:      # beak, feet
            rgb = BEAK
        elif luma < KEEP_DARK:                          # servos, lens, eyes
            rgb = base
        else:
            rgb = np.clip(colour * (0.42 + 0.62 * luma), 0, 1)
        model.geom_rgba[g] = [rgb[0], rgb[1], rgb[2], 1.0]


# ---------------------------------------------------------------------------
# Posing: the rig, and the ground under it
# ---------------------------------------------------------------------------

LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
NECK = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def quat(yaw=0.0, pitch=0.0, roll=0.0):
    """Quaternion for yaw about z, then pitch about y, then roll about x."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def pose(left, neck, mouth=0.0, pitch=0.0, roll=0.0, right=None, lift=0.0):
    """One posture of a duck.

    `left` and `right` are both written in the LEFT leg's sign convention --
    (hip_yaw, hip_roll, hip_pitch, knee, ankle) -- and the mirror the model
    needs is applied when the pose is written, so an asymmetric pose reads the
    way you would describe it out loud. `pitch` and `roll` tilt the trunk;
    `lift` raises the whole duck off the ground it would otherwise rest on.
    """
    left = np.asarray(left, dtype=float)
    return dict(left=left, right=left.copy() if right is None else np.asarray(right, float),
                neck=np.asarray(neck, float), mouth=float(mouth),
                pitch=float(pitch), roll=float(roll), lift=float(lift))


def blend(a, b, u):
    """Linear blend of two poses; `u` is already eased by the caller."""
    u = float(np.clip(u, 0.0, 1.0))
    out = {}
    for k in a:
        out[k] = a[k] * (1 - u) + b[k] * u if isinstance(a[k], np.ndarray) else \
            a[k] * (1 - u) + b[k] * u
    return out


# STAND is scene.xml's STAND keyframe, the pose the standing policy holds.
STAND = pose((0, -0.0873, -0.4579, -0.0049, 0.4530), (0.3491, 0.3491, 0, 0))
# SIT is scene.xml's SIT keyframe with a friendlier head.
SIT = pose((0, 0, -0.5236, 1.0472, 0), (0.62, 0.30, 0, 0), pitch=-0.10)
# LAY: the same sit with the head bowed over the nest. Not decoration -- sitting
# with her head up puts her crown 16 mm BEHIND her own trunk and 245 mm high,
# which no standing duck can reach past her own body. Bowed, the crown comes
# 113 mm forward and drops to 211 mm, and he can stand clear and reach it.
LAY = pose((0, 0, -0.5236, 1.0472, 0), (-0.34, 1.18, 0, 0), pitch=-0.05)
# Kneeling: the left leg folds under -- thigh back, shin down, foot flat
# behind -- while the right stays planted ahead. Bird legs kneel on the hock.
KNEEL = pose((0, -0.05, 0.62, -1.45, 0.78), (0.10, -0.30, 0, 0),
             right=(0, -0.12, -0.95, 0.62, 0.42), pitch=-0.14)


class Duck:
    """One duck in the world, posed by hand rather than driven.

    Everything is written straight into `qpos`; `mj_kinematics` then puts the
    bodies where they belong. `ground()` is the only thing standing in for
    physics: it drops the duck until the lowest corner of its visible geometry
    touches z = 0, which is what keeps a kneel, a sit and a walk cycle all
    resting on the same floor without a single hand-tuned trunk height.
    """

    def __init__(self, model, data, prefix):
        self.model, self.data, self.prefix = model, data, prefix
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                prefix + "trunk_base_freejoint")
        self.root = int(model.jnt_qposadr[jid])
        self.j = {}
        for side in ("left", "right"):
            for n in LEG:
                nm = f"{side}_{n}"
                self.j[nm] = int(model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + nm)])
        for n in NECK + ("mouth",):
            self.j[n] = int(model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)])
        self.tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "mouth_tip")
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
        self.head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      prefix + "yaw_roll_motion")

        # Geoms that are actually drawn, and the eight corners of each one's
        # own bounding box -- enough to find the lowest visible point of any
        # posture with one matrix multiply per frame.
        ids, corners = [], []
        for g in range(model.ngeom):
            if model.geom_group[g] != 2:
                continue
            b = model.geom_bodyid[g]
            p = b
            while p > 0 and p != self.body:
                p = model.body_parentid[p]
            if p != self.body:
                continue
            ids.append(g)
            corners.append(_geom_corners(model, g))
        self.gids = np.array(ids)
        self.corners = np.array(corners)                       # (n, 8, 3)
        self.pose = STAND
        self.xy = np.zeros(2)
        self.yaw = 0.0

    def write(self, xy, yaw, p, z=None):
        """Put the duck at `xy` facing `yaw` in pose `p`, resting on the floor."""
        self.xy, self.yaw, self.pose = np.asarray(xy, float), float(yaw), p
        q = self.data.qpos
        q[self.root:self.root + 2] = self.xy
        q[self.root + 2] = 0.30 if z is None else z
        q[self.root + 3:self.root + 7] = quat(yaw, p["pitch"], p["roll"])
        for i, n in enumerate(LEG):
            q[self.j["left_" + n]] = p["left"][i]
            q[self.j["right_" + n]] = -p["right"][i]     # the model's mirror
        for i, n in enumerate(NECK):
            q[self.j[n]] = p["neck"][i]
        q[self.j["mouth"]] = p["mouth"]
        mujoco.mj_kinematics(self.model, self.data)
        if z is None:
            q[self.root + 2] += p["lift"] - self.low()
            mujoco.mj_kinematics(self.model, self.data)

    def low(self):
        """World height of the lowest visible corner of this duck."""
        xpos = self.data.geom_xpos[self.gids]
        zrow = self.data.geom_xmat[self.gids][:, 6:9]           # (n, 3)
        z = xpos[:, 2:3] + np.einsum("nij,nj->ni", self.corners, zrow)
        return float(z.min())

    def low_of(self, body_names):
        """Lowest visible corner of a few named bodies -- one shin, one sole."""
        want = set()
        for nm in body_names:
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + nm)
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == b and self.model.geom_group[g] == 2:
                    want.add(g)
        sel = np.array([i for i, g in enumerate(self.gids) if g in want])
        xpos = self.data.geom_xpos[self.gids[sel]]
        zrow = self.data.geom_xmat[self.gids[sel]][:, 6:9]
        z = xpos[:, 2:3] + np.einsum("nij,nj->ni", self.corners[sel], zrow)
        return float(z.min())

    def fade(self, a):
        """Alpha on every visible geom -- a duckling arriving out of a shell."""
        if not hasattr(self, "_rgba0"):
            self._rgba0 = self.model.geom_rgba[self.gids].copy()
        self.model.geom_rgba[self.gids, 3] = float(np.clip(a, 0, 1))

    def beak(self):
        return self.data.site_xpos[self.tip].copy()

    def head_top(self):
        """The crown of the head: the middle of the head, at its highest point.

        The corner of the head's bounding box is 46 mm off to one side, so a
        kiss aimed at it lands on an ear rather than a crown -- this returns
        the centre in xy and the top in z.
        """
        pts = []
        for i, g in enumerate(self.gids):
            p = self.model.geom_bodyid[g]
            while p > 0 and p != self.head:
                p = self.model.body_parentid[p]
            if p != self.head:
                continue
            pts.append(self.data.geom_xpos[g]
                       + self.corners[i] @ self.data.geom_xmat[g].reshape(3, 3).T)
        pts = np.concatenate(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2]])


def _geom_corners(model, g):
    """Eight corners of a geom's bounding box, in the geom's own frame."""
    t = model.geom_type[g]
    s = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        a, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
        v = model.mesh_vert[a:a + n]
        lo, hi = v.min(axis=0), v.max(axis=0)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        hi = np.array([s[0], s[0], s[0]]); lo = -hi
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        hi = np.array([s[0], s[0], s[1] + s[0]]); lo = -hi
    elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX,
               mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        hi = np.array([s[0], s[1] if t != mujoco.mjtGeom.mjGEOM_CYLINDER else s[0], s[2]
                       if t != mujoco.mjtGeom.mjGEOM_CYLINDER else s[1]]); lo = -hi
    else:
        hi = np.zeros(3); lo = np.zeros(3)
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


class Prop:
    """A ring, an eggshell or a heart: a free body the animation carries."""

    def __init__(self, model, data, name):
        self.model, self.data, self.name = model, data, name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_free")
        self.q = int(model.jnt_qposadr[jid])
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        self.alpha0 = model.geom_rgba[self.gids, 3].copy()

    def write(self, pos, q=(1, 0, 0, 0)):
        self.data.qpos[self.q:self.q + 3] = pos
        self.data.qpos[self.q + 3:self.q + 7] = q

    def fade(self, a):
        self.model.geom_rgba[self.gids, 3] = np.clip(a, 0, 1)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def ramp(t, t0, t1):
    return float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))


def ease(t, t0, t1):
    """Smoothstep from 0 at t0 to 1 at t1 -- every move in the film uses it."""
    u = ramp(t, t0, t1)
    return u * u * (3.0 - 2.0 * u)


def bump(t, t0, t1):
    """0 -> 1 -> 0 across the window: a nod, a breath, a shrug."""
    u = ramp(t, t0, t1)
    return math.sin(math.pi * u) ** 2


def lerp(a, b, u):
    return np.asarray(a) * (1.0 - u) + np.asarray(b) * u


# ---------------------------------------------------------------------------
# Poses the story needs beyond STAND / SIT / KNEEL
# ---------------------------------------------------------------------------

# Kneeling with the head lifted and the beak open: the offer.
OFFER = pose((0, -0.05, 0.10, -0.823, 0.95), (-0.10, -0.62, 0, 0),
             right=(0, -0.10, -0.50, 0.92, -0.42), pitch=0.06, mouth=0.34)
KNEEL = pose((0, -0.05, 0.10, -0.823, 0.95), (0.16, -0.20, 0, 0),
             right=(0, -0.10, -0.50, 0.92, -0.42), pitch=0.10)
# She rocks back on her heels, beak open: "me?"
SURPRISE = pose((0, -0.10, -0.30, 0.10, 0.24), (0.62, -0.55, 0, 0),
                pitch=-0.16, mouth=0.40)
# Leaning in, neck stretched forward, head level: the kiss.
LEAN = pose((0, -0.09, -0.62, 0.16, 0.50), (-0.42, 0.16, 0, 0), pitch=0.16)
# Reaching down to take something, or to touch a duckling.
REACH = pose((0, -0.09, -0.70, 0.30, 0.44), (-0.30, 0.72, 0, 0),
             pitch=0.22, mouth=0.30)
# Sitting on the nest, wings-down, settled.
BROOD = pose((0, 0, -0.5236, 1.0472, 0), (0.70, 0.34, 0, 0), pitch=-0.06)
# Standing tall and proud, head up.
PROUD = pose((0, -0.0873, -0.4579, -0.0049, 0.4530), (0.10, -0.30, 0, 0))
# Neck up, beak down: the highest a duck can put its beak (223 mm) while
# still pointing it downwards. This is what a kiss on a sitting duck's head
# costs -- her crown sits at 230 mm even sitting, so he has nothing to spare.
BOW = pose((0, -0.0873, -0.4579, -0.0049, 0.4530), (0.60, 0.60, 0, 0),
           pitch=0.12, mouth=0.10)
# Down to duckling height: knees folded, trunk tipped forward, beak at 98 mm,
# which is where a duckling's head is (92 mm). Found by searching the pose
# rather than guessed -- the standing bow cannot get anywhere near it.
STOOP = pose((0, -0.09, -0.900, 0.811, 0.089), (-0.75, 0.30, 0, 0),
             pitch=0.157, mouth=0.16)
# The highest his beak goes: a kiss on the head has to aim at or below this.
BEAK_CEILING = 0.216


def gait(base, phase, amp=1.0):
    """A walk cycle laid on top of a standing pose.

    The legs are the same motion half a period apart, written in the left
    leg's sign convention like every other pose here. The ankle takes back
    what the hip and knee did, which is what keeps a sole flat on the floor --
    the same relation the STAND keyframe has (ankle = -(hip + knee)).
    The bob and the sway are not authored: `Duck.ground()` produces them,
    because the trunk has to ride on whichever foot is lowest.
    """
    def delta(p):
        a = 2.0 * math.pi * p
        swing, lift = math.sin(a), max(0.0, math.sin(a))
        hip = -0.30 * amp * swing
        knee = 0.52 * amp * lift
        return np.array([0.0, 0.0, hip, knee, -(hip + knee)])

    out = dict(base)
    out["left"] = base["left"] + delta(phase)
    out["right"] = base["right"] + delta(phase + 0.5)
    out["roll"] = base["roll"] + 0.06 * amp * math.sin(2.0 * math.pi * phase)
    out["neck"] = base["neck"] + np.array(
        [0.05 * amp * math.sin(4.0 * math.pi * phase), 0.0, 0.0, 0.0])
    return out


STEP = 0.055        # metres of ground per gait cycle, so the feet do not skate


# ---------------------------------------------------------------------------
# The film
# ---------------------------------------------------------------------------

Y_STAGE = STAGE[1]
KISS_GAP = 0.004


class Story:
    """Everyone in the film, and the six scenes they play."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.he = Duck(model, data, "blue_")
        self.she = Duck(model, data, "pink_")
        self.son = Duck(model, data, "bblue_")
        self.daughter = Duck(model, data, "bpink_")
        self.ring = Prop(model, data, "ring")
        self.egg = [[Prop(model, data, f"egg{e}_{h}") for h in ("bot", "top")]
                    for e in (0, 1)]
        self.hearts = [Prop(model, data, f"heart{i}") for i in range(N_HEARTS)]
        rng = np.random.default_rng(11)
        self.heart_rand = rng.uniform(-1.0, 1.0, size=(N_HEARTS, 5))

        # Where the two of them stand so that, in the kiss, the beaks meet.
        # Measured off the model rather than guessed: pose each duck at the
        # origin, read the beak's own offset, and split the difference.
        self.he.write((0, 0), 0.0, LEAN)
        reach_he = float(self.he.beak()[0])
        self.she.write((0, 0), math.pi, LEAN)
        reach_she = float(-self.she.beak()[0])
        mid = STAGE[0]
        self.he_x = mid - (reach_he + KISS_GAP / 2)
        self.she_x = mid + (reach_she + KISS_GAP / 2)
        self.nest_z = 0.5 * EGG_LENGTH - 0.004        # eggs resting in the nest
        # Apart along x, because the camera looks down y: two eggs side by
        # side across the lens are one egg with a shadow.
        self.egg_home = [np.array([0.018, 0.020]), np.array([0.064, -0.018])]

    # -- helpers ---------------------------------------------------------

    def park(self):
        """Off-camera by default: hearts dark, ducklings under the floor."""
        for h in self.hearts:
            h.fade(0.0)
            h.write((0, 0, -1.0))
        for d in (self.son, self.daughter):
            d.fade(0.0)
            d.write((0.0, 2.0), 0.0, STAND)

    def hold_ring(self, duck, out=0.014, tilt=0.0):
        """Put the ring in a beak, standing up, its face to the camera.

        Along the beak's OWN axis, not the duck's heading: when she bows to
        take it her beak points down, and a ring offset horizontally
        disappears inside her head.
        """
        tip = duck.beak()
        fwd = tip - self.data.xpos[duck.head]
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-6)
        self.ring.write(tip + fwd * out, quat(tilt, 0.0, math.pi / 2))

    def wear_ring(self, duck):
        """The ring worn at the throat, lying flat, stone to the camera.

        The one place a 27 mm ring goes round this robot: the 22 mm stalk
        just under the head. Lower down, the neck is 40 mm across and the
        ring would hang through it.
        """
        stalk = self.data.xpos[duck.head].copy()
        self.ring.write(stalk + np.array([0, 0, -0.006]), quat(math.pi))

    def heart_stream(self, origin, t, t0, t1, first, count, rate=2.6,
                     life=2.4, rise=0.16, spread=0.075, size=1.0):
        """Hearts leaving `origin`, one every 1/rate seconds, drifting up."""
        for k in range(count):
            h = self.hearts[(first + k) % N_HEARTS]
            born = t0 + k / rate
            age = t - born
            if age < 0.0 or age > life or t > t1 + life:
                continue
            u = age / life
            r = self.heart_rand[(first + k) % N_HEARTS]
            pos = np.asarray(origin) + np.array([
                spread * r[0] * u + 0.02 * math.sin(4.0 * age + r[1]),
                spread * 0.5 * r[1] * u,
                rise * u + 0.02 * u * u,
            ])
            h.write(pos, quat(0.6 * r[2] + 0.7 * math.sin(1.5 * age + r[3]),
                              0.0, 0.35 * r[4]))
            h.fade(min(1.0, 6.0 * u) * (1.0 - max(0.0, (u - 0.55) / 0.45)) * 0.95)

    @staticmethod
    def path(p0, p1, via=None):
        """A straight line, or a quadratic bend through `via`.

        Walking a straight line from one side of the nest to the other means
        walking through the nest, so the swaps bend around the back.
        """
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        c = None if via is None else np.asarray(via, float)

        def at(u):
            u = float(np.clip(u, 0.0, 1.0))
            if c is None:
                return p0 + (p1 - p0) * u
            return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * c + u * u * p1

        def heading(u):
            d = at(min(1.0, u + 0.02)) - at(max(0.0, u - 0.02))
            return math.atan2(d[1], d[0])

        def length(u):
            pts = np.array([at(k / 24.0 * u) for k in range(25)])
            return float(np.abs(np.diff(pts, axis=0)).sum(axis=1).sum())

        return at, heading, length

    def walk(self, duck, at, heading, length, u, base=None, amp=1.0):
        """Carry a duck along a path with its feet keeping up."""
        base = STAND if base is None else base
        w = min(1.0, 6.0 * min(u, 1.0 - u) + 0.12) if 0.0 < u < 1.0 else 0.0
        duck.write(at(u), heading(u), gait(base, length(u) / STEP, amp=amp * w))

    def kiss_head(self, duck, target, base, yaw, gain=9.0, iters=5):
        """Stand a duck so its beak lands on `target` -- a kiss on the head.

        Two unknowns, solved rather than tuned: the trunk's xy comes from the
        beak's own offset in the pose (one subtraction), and the height from
        bending the neck, whose beak travels about 69 mm per radian. Four
        passes put the beak inside a millimetre of wherever the other duck is
        holding her head, however she happens to be sitting.
        """
        target = np.asarray(target, float)
        p = base
        for _ in range(iters):
            duck.write((0.0, 0.0), yaw, p)
            off = duck.beak()[:2].copy()
            duck.write(target[:2] - off, yaw, p)
            dz = float(target[2] - duck.beak()[2])
            if abs(dz) < 0.0015:
                break
            p = dict(p)
            p["neck"] = p["neck"] + np.array([np.clip(gain * dz, -0.5, 0.5), 0, 0, 0])
            p["neck"][0] = float(np.clip(p["neck"][0], -1.45, 1.00))
        return p

    def travel(self, duck, p0, p1, yaw0, yaw1, t, t0, t1, base=None, amp=1.0,
               via=None):
        """Turn, walk, turn: a duck going somewhere and arriving facing right.

        Walking sideways looks like a robot on a conveyor, so the heading is
        the direction of travel for the middle of the move and only swings to
        the pose's own facing at each end.
        """
        at, heading, length = self.path(p0, p1, via)
        h0, h1 = heading(0.0), heading(1.0)
        yaw0, yaw1 = _unwrap(yaw0, h0), _unwrap(yaw1, h1)
        span = t1 - t0
        a, b = t0 + 0.16 * span, t1 - 0.16 * span
        base = STAND if base is None else base
        if t < a:                                   # turn on the spot
            u = ease(t, t0, a)
            duck.write(at(0.0), yaw0 + (h0 - yaw0) * u,
                       gait(base, 0.55 * u, amp=0.45 * u))
        elif t < b:
            self.walk(duck, at, heading, length, ease(t, a, b), base=base, amp=amp)
        else:
            u = ease(t, b, t1)
            duck.write(at(1.0), h1 + (yaw1 - h1) * u,
                       gait(base, 0.55 * u, amp=0.45 * (1 - u)))


def _unwrap(a, near):
    """The version of angle `a` that is closest to `near`."""
    while a - near > math.pi:
        a -= 2 * math.pi
    while near - a > math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------------------
# The six scenes
# ---------------------------------------------------------------------------

# The camera looks down the y axis, so everyone who has to be seen at once is
# spread along x. The nest is 105 mm across with a 135 mm outer rim: a parent
# standing at |x| > 0.20 is beside the nest, not in it.
NEST_SIT = np.array([-0.035, 0.0])       # where she sits to lay
BROOD_SIT = np.array([0.041, 0.001])     # where a parent sits to brood, on the eggs
ASIDE_HER = np.array([-0.245, 0.030])    # her spot beside the nest
ASIDE_HIM = np.array([0.250, 0.035])     # his, on the other side
BOW_SPOT = np.array([-0.235, 0.015])     # where he stands to bow over her head
BEHIND = np.array([0.0, 0.30])           # a waypoint round the back of the nest
CHICK_MEET = (np.array([-0.165, -0.010]), np.array([0.185, -0.020]))
EGG_Z = 0.019 + 0.5 * EGG_LENGTH - 0.023  # eggs nestled a little into the straw


def s1_meet(S, t):
    """He comes up the lawn with a ring in his beak; she notices him."""
    S.park()
    look = dict(STAND)
    look["neck"] = STAND["neck"] + np.array(
        [-0.10 * ease(t, 3.4, 4.6), -0.10 * ease(t, 3.4, 4.6),
         0.0, 0.30 * bump(t, 4.8, 6.6)])
    S.she.write((S.she_x, Y_STAGE), math.pi, look)

    if t < 4.5:
        S.travel(S.he, (S.he_x - 0.60, Y_STAGE), (S.he_x, Y_STAGE),
                 0.0, 0.0, t, 0.15, 4.5, base=PROUD)
    else:
        p = dict(PROUD)
        p["neck"] = PROUD["neck"] + np.array([0.06 * bump(t, 4.8, 6.2), 0, 0, 0])
        S.he.write((S.he_x, Y_STAGE), 0.0, p)
    S.hold_ring(S.he)


def s2_propose(S, t):
    """He kneels and offers it. She is startled, then says yes, then takes it."""
    S.park()
    if t < 2.5:
        p = blend(PROUD, KNEEL, ease(t, 0.4, 2.5))
    elif t < 3.7:
        p = blend(KNEEL, OFFER, ease(t, 2.5, 3.7))
    else:
        p = dict(OFFER)
        p["neck"] = OFFER["neck"] + np.array([0.05 * bump(t, 4.0, 5.4), 0, 0, 0])
    S.he.write((S.he_x, Y_STAGE), 0.0, p)

    if t < 3.3:
        q = dict(STAND)
        q["neck"] = STAND["neck"] + np.array([0, 0.14 * ease(t, 1.2, 2.6), 0, 0.2])
    elif t < 6.8:
        q = blend(STAND, SURPRISE, ease(t, 3.3, 4.5))
        nod = bump(t, 5.0, 5.8) + bump(t, 5.9, 6.7)      # yes, yes
        q = dict(q)
        q["neck"] = q["neck"] + np.array([0.30 * nod, 0.55 * nod, 0, 0])
    elif t < 8.6:
        q = blend(SURPRISE, REACH, ease(t, 6.8, 8.2))
    else:
        q = blend(REACH, PROUD, ease(t, 8.8, 10.4))
    S.she.write((S.she_x, Y_STAGE), math.pi, q)

    if t < 8.0:                                   # the ring changes beaks
        S.hold_ring(S.he)
    elif t < 8.6:
        S.hold_ring(S.he)
        a = S.ring.data.qpos[S.ring.q:S.ring.q + 3].copy()
        S.hold_ring(S.she)
        b = S.ring.data.qpos[S.ring.q:S.ring.q + 3].copy()
        u = ease(t, 8.0, 8.6)
        S.ring.write(lerp(a, b, u) + np.array([0, 0, 0.012 * bump(t, 8.0, 8.6)]),
                     quat(0, 0, math.pi / 2))
    else:
        S.hold_ring(S.she)
    S.heart_stream((S.he_x + 0.10, Y_STAGE, 0.22), t, 4.6, 10.5, 0, 7, rate=1.6)


def s3_kiss(S, t):
    """He stands, the ring goes round her neck, and they hold a long kiss."""
    S.park()
    breath = 0.0022 * math.sin(2.0 * math.pi * 0.45 * (t - 3.8))
    tilt = 0.22 * ease(t, 3.0, 4.0)

    if t < 2.3:
        p = blend(OFFER, PROUD, ease(t, 0.2, 2.1))
    elif t < 7.2:
        p = blend(PROUD, LEAN, ease(t, 2.5, 3.8))
    else:
        p = blend(LEAN, PROUD, ease(t, 7.2, 8.4))
    p = dict(p)
    p["neck"] = p["neck"] + np.array([0, 0, 0, tilt])
    p["mouth"] = 0.10 * ease(t, 3.4, 4.0)
    S.he.write((S.he_x - breath, Y_STAGE), 0.0, p)

    if t < 2.5:
        q = blend(PROUD, PROUD, 0.0)
    elif t < 7.2:
        q = blend(PROUD, LEAN, ease(t, 2.5, 3.8))
    else:
        q = blend(LEAN, PROUD, ease(t, 7.2, 8.4))
    q = dict(q)
    q["neck"] = q["neck"] + np.array([0, 0, 0, tilt])
    q["mouth"] = 0.10 * ease(t, 3.4, 4.0)
    S.she.write((S.she_x + breath, Y_STAGE), math.pi, q)

    if t < 0.7:                                   # beak -> around her neck
        S.hold_ring(S.she)
    elif t < 1.7:
        S.hold_ring(S.she)
        a = S.ring.data.qpos[S.ring.q:S.ring.q + 3].copy()
        S.wear_ring(S.she)
        b = S.ring.data.qpos[S.ring.q:S.ring.q + 3].copy()
        S.ring.write(lerp(a, b, ease(t, 0.7, 1.7)), quat(math.pi))
    else:
        S.wear_ring(S.she)
    S.heart_stream((STAGE[0], Y_STAGE, 0.27), t, 3.9, 7.4, 2, 12,
                   rate=3.4, life=2.6, rise=0.26, spread=0.15)


def s4_eggs(S, t):
    """They move to the nest; she sits and lays two eggs, he kisses her head."""
    S.park()

    # --- her: over to the nest, then down, then two eggs -----------------
    if t < 4.4:
        S.travel(S.she, (S.she_x, Y_STAGE), NEST_SIT, math.pi, math.pi,
                 t, 0.2, 4.4, via=(0.20, -0.22))
    else:
        q = dict(blend(blend(STAND, SIT, ease(t, 4.5, 5.8)), LAY, ease(t, 5.8, 6.8)))
        push = bump(t, 6.8, 8.0) + bump(t, 9.8, 11.0)      # the effort of laying
        q["neck"] = q["neck"] + np.array([-0.10 * push, 0.16 * push, 0, 0])
        q["pitch"] = q["pitch"] - 0.06 * push
        S.she.write(NEST_SIT, math.pi, q)

    # --- him: round to her far side, then bowing to her crown ------------
    if t < 4.4:
        S.travel(S.he, (S.he_x, Y_STAGE), BOW_SPOT, 0.0, 0.0, t, 0.6, 4.4,
                 via=(-0.34, -0.26))
    else:
        bow = ease(t, 4.6, 6.2) * (1.0 - ease(t, 11.8, 12.8))
        base = dict(blend(STAND, BOW, bow))
        if bow > 0.05:
            crown = S.she.head_top()
            crown[2] = min(crown[2] - 0.008, BEAK_CEILING) + 0.004 * bump(t, 6.8, 9.8)
            S.kiss_head(S.he, crown, base, 0.0)
        else:
            S.he.write(BOW_SPOT, 0.0, base)
    S.wear_ring(S.she)

    # --- the eggs, out from under her and into the straw -----------------
    for e, (t0, t1) in enumerate(((6.8, 8.4), (9.8, 11.4))):
        home = np.array([S.egg_home[e][0], S.egg_home[e][1], EGG_Z])
        birth = np.array([NEST_SIT[0] + 0.052, NEST_SIT[1] + 0.014 - 0.028 * e, 0.032])
        u = ease(t, t0, t1)
        pos = lerp(birth, home, u)
        pos[2] += 0.008 * math.sin(math.pi * u)
        for half in S.egg[e]:
            half.write(pos, quat(0.5 + 1.2 * u + 0.4 * e, 0.10 * math.sin(3 * u), 0))
            half.fade(ease(t, t0 + 0.1, t0 + 0.7))
    S.heart_stream((-0.16, 0.02, 0.30), t, 6.4, 12.5, 4, 9, rate=1.3, life=2.8)


def s5_brood(S, t):
    """She turns round and sits on them; then they swap and he takes a turn."""
    S.park()
    for e in (0, 1):
        for half in S.egg[e]:
            half.fade(1.0)
            half.write((S.egg_home[e][0], S.egg_home[e][1], EGG_Z),
                       quat(1.7 + 0.4 * e, 0.10, 0))

    brood = dict(BROOD)
    brood["lift"] = 0.021

    # --- her turn: up, round, and down onto the eggs ---------------------
    if t < 1.4:
        S.she.write(NEST_SIT, math.pi, blend(LAY, STAND, ease(t, 0.1, 1.3)))
    elif t < 3.8:
        S.travel(S.she, NEST_SIT, BROOD_SIT, math.pi, 0.0, t, 1.4, 3.6,
                 via=(-0.02, -0.16))
    elif t < 7.8:
        settle = dict(brood)
        settle["roll"] = 0.035 * math.sin(2.0 * math.pi * 0.32 * t)
        settle["neck"] = brood["neck"] + np.array(
            [0, 0.14 * bump(t, 5.2, 6.2), 0.28 * bump(t, 6.4, 7.6), 0])
        S.she.write(BROOD_SIT, 0.0, blend(STAND, settle, ease(t, 3.7, 4.8)))
    elif t < 8.9:
        S.she.write(BROOD_SIT, 0.0, blend(brood, STAND, ease(t, 7.8, 8.8)))
    else:
        S.travel(S.she, BROOD_SIT, ASIDE_HER, 0.0, 0.12, t, 8.9, 10.8,
                 via=(-0.10, -0.16))

    # --- his turn -------------------------------------------------------
    if t < 1.0:
        S.he.write(BOW_SPOT, 0.0, STAND)
    elif t < 4.2:
        S.travel(S.he, BOW_SPOT, ASIDE_HIM, 0.0, math.pi - 0.12, t, 1.0, 4.0,
                 via=BEHIND)
    elif t < 9.4:
        p = dict(STAND)
        p["neck"] = STAND["neck"] + np.array(
            [0.10 * bump(t, 5.0, 6.4), 0.24, 0, 0.20 * bump(t, 6.6, 8.0)])
        S.he.write(ASIDE_HIM, math.pi - 0.12, p)
    elif t < 11.6:
        S.travel(S.he, ASIDE_HIM, BROOD_SIT, math.pi - 0.12, math.pi,
                 t, 9.4, 11.4, via=(0.16, -0.14))
    else:
        settle = dict(brood)
        settle["roll"] = 0.030 * math.sin(2.0 * math.pi * 0.30 * t)
        settle["neck"] = brood["neck"] + np.array([0, 0.12 * bump(t, 12.4, 13.4), 0, 0])
        S.he.write(BROOD_SIT, math.pi, blend(STAND, settle, ease(t, 11.5, 12.6)))
    S.wear_ring(S.she)
    S.heart_stream((0.0, 0.06, 0.30), t, 5.4, 12.0, 6, 6, rate=0.8, life=3.2)


def hop(base, phase, height=0.014):
    """A duckling's hop: both legs tuck, and the body leaves the ground."""
    air = max(0.0, math.sin(2.0 * math.pi * phase))
    p = dict(base)
    tuck = np.array([0.0, 0.0, -0.35 * air, 0.75 * air, -0.40 * air])
    p["left"] = base["left"] + tuck
    p["right"] = base["right"] + tuck
    p["neck"] = base["neck"] + np.array([-0.12 * air, -0.10 * air, 0, 0])
    return p, height * air


def s6_hatch(S, t):
    """Both eggs rock, crack open, and a blue and a pink duckling come out."""
    S.park()

    # --- the parents get out of the way and lean in ----------------------
    if t < 1.3:
        S.he.write(BROOD_SIT, math.pi, blend(BROOD, STAND, ease(t, 0.1, 1.2)))
    elif t < 3.6:
        S.travel(S.he, BROOD_SIT, ASIDE_HIM, math.pi, math.pi - 0.12,
                 t, 1.3, 3.4, via=(0.16, -0.14))
    watch_him = dict(STAND)
    watch_him["neck"] = STAND["neck"] + np.array(
        [-0.16 * ease(t, 3.6, 4.8), 0.36 * ease(t, 3.6, 4.8), 0,
         0.18 * bump(t, 5.2, 6.6)])
    watch_him["pitch"] = 0.10 * ease(t, 3.6, 4.8)
    if t >= 3.6:
        S.he.write(ASIDE_HIM, math.pi - 0.12, watch_him)

    watch_her = dict(STAND)
    watch_her["neck"] = STAND["neck"] + np.array(
        [-0.15 * ease(t, 1.4, 2.8), 0.34 * ease(t, 1.4, 2.8), 0,
         0.20 * bump(t, 3.2, 4.6)])
    watch_her["pitch"] = 0.10 * ease(t, 1.4, 2.8)
    S.she.write(ASIDE_HER, 0.12, watch_her)
    S.wear_ring(S.she)

    # --- the eggs: rocking, then a lid coming off ------------------------
    OPEN = (4.7, 6.5)
    LAND = (np.array([-0.088, 0.062, 0.008]), np.array([0.128, -0.058, 0.008]))
    for e in (0, 1):
        home = np.array([S.egg_home[e][0], S.egg_home[e][1], EGG_Z])
        shake = ease(t, 1.9, 4.4) * (1.0 - ease(t, OPEN[e], OPEN[e] + 0.25))
        w = 2.0 * math.pi * (1.5 + 1.4 * shake)
        rock = shake * (0.30 * math.sin(w * t + 2.1 * e)
                        + 0.12 * math.sin(2.7 * w * t + 1.0 * e))
        base = home + np.array([0, 0, 0.006 * shake * abs(math.sin(w * t))])
        S.egg[e][0].fade(1.0)
        S.egg[e][0].write(base, quat(1.7 + 0.4 * e, rock, 0.55 * rock))

        u = ease(t, OPEN[e], OPEN[e] + 0.85)
        lid = lerp(base + np.array([0, 0, 0.004]), LAND[e], u)
        lid[2] += 0.055 * math.sin(math.pi * u)        # it pops, then falls
        S.egg[e][1].fade(1.0)
        S.egg[e][1].write(lid, quat(1.7 + 0.4 * e + 2.2 * u,
                                    rock * (1 - u) + 2.4 * u, 0.55 * rock + 0.8 * u))

    # --- the ducklings: up out of the shell, then over the rim -----------
    for e, (duck, born, face) in enumerate(((S.son, 5.1, 2.6), (S.daughter, 6.9, 0.5))):
        if t < born:
            continue
        rise = ease(t, born, born + 2.2)
        out = ease(t, born + 2.6, born + 4.4)
        home = np.array(S.egg_home[e])
        here = lerp(home, CHICK_MEET[e], out)
        duck.write(here, face, STAND)                 # grounded, to read its height
        z_stand = float(S.data.qpos[duck.root + 2])
        look = dict(STAND)
        look["neck"] = STAND["neck"] + np.array(
            [-0.20 * rise, -0.10 * rise, 0.55 * math.sin(1.9 * (t - born)), 0])
        look["mouth"] = 0.30 * bump(t, born + 1.4, born + 2.2)
        p, lifted = hop(look, max(0.0, t - born - 2.6) * 1.25)
        yaw = face + (math.atan2(*(CHICK_MEET[e] - home)[::-1]) - face) * out
        duck.write(here + np.array([0.008 * math.sin(1.4 * (t - born)), 0.0]),
                   yaw + 0.35 * math.sin(0.9 * (t - born)) * (1.0 - out),
                   p, z=lerp(0.010, z_stand, rise) + lifted
                   + 0.030 * math.sin(math.pi * out))     # over the nest rim
        duck.fade(min(1.0, ease(t, born, born + 0.6) + 0.05))

    # --- the parents come down to meet them ------------------------------
    nuz = ease(t, 11.2, 12.6) * (1.0 - ease(t, 13.8, 14.6))
    if nuz > 0.05:
        S.kiss_head(S.she, S.son.head_top() + np.array([0, 0, 0.004]),
                    blend(watch_her, STOOP, nuz), 0.12)
        S.kiss_head(S.he, S.daughter.head_top() + np.array([0, 0, 0.004]),
                    blend(watch_him, STOOP, nuz), math.pi - 0.12)
    S.heart_stream((0.0, 0.02, 0.20), t, 5.4, 14.2, 0, 16, rate=1.5, life=3.2,
                   rise=0.32, spread=0.20)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# (time, lookat, distance, azimuth, elevation). The camera sits on the -y side
# of the lawn, so +x runs to the right of frame: he comes in from the left.

SCENES = [
    ("meet", 7.0, s1_meet, [
        (0.0, (0.02, -0.42, 0.15), 1.30, 99, -11),
        (4.6, (0.00, -0.50, 0.15), 1.00, 92, -8),
        (7.0, (0.00, -0.52, 0.15), 0.92, 88, -7)]),
    ("propose", 10.5, s2_propose, [
        (0.0, (0.00, -0.52, 0.15), 0.92, 88, -7),
        (3.2, (-0.01, -0.52, 0.14), 0.78, 82, -4),
        (7.0, (0.01, -0.52, 0.15), 0.70, 92, -3),
        (10.5, (0.00, -0.52, 0.16), 0.74, 99, -5)]),
    ("kiss", 8.5, s3_kiss, [
        (0.0, (0.00, -0.52, 0.16), 0.74, 99, -5),
        (3.6, (0.00, -0.52, 0.19), 0.56, 90, -1),
        (7.0, (0.00, -0.52, 0.19), 0.54, 82, -2),
        (8.5, (0.00, -0.50, 0.18), 0.66, 76, -6)]),
    ("eggs", 13.0, s4_eggs, [
        (0.0, (0.00, -0.42, 0.16), 1.00, 76, -9),
        (4.4, (-0.09, -0.03, 0.14), 0.92, 88, -11),
        (8.0, (-0.07, -0.01, 0.13), 0.80, 94, -10),
        (13.0, (-0.05, 0.00, 0.12), 0.78, 101, -11)]),
    ("brood", 13.0, s5_brood, [
        (0.0, (-0.05, 0.00, 0.12), 0.80, 101, -11),
        (5.0, (0.02, 0.00, 0.10), 0.72, 92, -13),
        (9.5, (0.04, 0.00, 0.11), 0.78, 82, -11),
        (13.0, (0.03, 0.00, 0.10), 0.76, 88, -12)]),
    ("hatch", 15.0, s6_hatch, [
        (0.0, (0.03, 0.00, 0.10), 0.76, 88, -12),
        (3.8, (0.02, 0.00, 0.07), 0.60, 94, -10),
        (7.5, (0.03, 0.00, 0.07), 0.56, 86, -8),
        (11.5, (0.01, -0.01, 0.09), 0.80, 92, -9),
        (15.0, (0.00, -0.02, 0.12), 1.05, 98, -11)]),
]


def aim(cam, track, t):
    for i in range(len(track) - 1):
        if t <= track[i + 1][0] or i == len(track) - 2:
            k0, k1 = track[i], track[i + 1]
            u = ease(t, k0[0], k1[0])
            cam.lookat[:] = lerp(k0[1], k1[1], u)
            cam.distance = k0[2] * (1 - u) + k1[2] * u
            cam.azimuth = k0[3] * (1 - u) + k1[3] * u
            cam.elevation = k0[4] * (1 - u) + k1[4] * u
            return


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def check(S):
    """Everything the geometry has to get right, measured and printed."""
    print(f"stage      he x {S.he_x*1000:+.0f} mm, she x {S.she_x*1000:+.0f} mm")
    S.he.write((S.he_x, Y_STAGE), 0.0, LEAN)
    a = S.he.beak()
    S.she.write((S.she_x, Y_STAGE), math.pi, LEAN)
    b = S.she.beak()
    print(f"kiss       beaks {np.linalg.norm(a - b)*1000:.1f} mm apart, "
          f"height {a[2]*1000:.0f} vs {b[2]*1000:.0f} mm")
    S.he.write((0, 0), 0.0, KNEEL)
    print(f"kneel      trunk {S.data.qpos[S.he.root+2]*1000:.0f} mm, "
          f"folded leg {S.he.low_of(['leg', 'ankle_left'])*1000:+.1f} mm, "
          f"planted foot {S.he.low_of(['ankle_right'])*1000:+.1f} mm")
    S.she.write(NEST_SIT, math.pi, LAY)
    crown = S.she.head_top()
    want = crown.copy()
    want[2] = min(crown[2] - 0.008, BEAK_CEILING)
    S.kiss_head(S.he, want, BOW, 0.0)
    print(f"head kiss  beak lands {np.linalg.norm(S.he.beak()-want)*1000:.1f} mm "
          f"from the aim point (her crown {crown[2]*1000:.0f} mm), "
          f"he stands at x {S.he.xy[0]*1000:+.0f} mm")
    S.son.write(CHICK_MEET[0], 0.0, STAND)
    chick = S.son.head_top() + np.array([0, 0, 0.004])
    S.kiss_head(S.she, chick, STOOP, 0.12)
    print(f"nuzzle     beak lands {np.linalg.norm(S.she.beak()-chick)*1000:.1f} mm "
          f"from a duckling crown at {chick[2]*1000:.0f} mm, "
          f"she stands at x {S.she.xy[0]*1000:+.0f} mm")
    S.son.write((0, 0), 0.0, STAND)
    duck_z = float(S.data.qpos[S.son.root + 2])
    S.he.write((0, 0), 0.0, STAND)
    print(f"duckling   trunk {duck_z*1000:.0f} mm vs a parent's "
          f"{S.data.qpos[S.he.root+2]*1000:.0f} mm")
    print(f"egg        {EGG_LENGTH*1000:.0f} x {2*EGG_WIDTH*1000:.0f} mm, "
          f"resting centre {EGG_Z*1000:.0f} mm, homes "
          f"{[list((h*1000).round(0)) for h in S.egg_home]}")
    print(f"film       {len(SCENES)} scenes, {sum(s[1] for s in SCENES):.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="videos/love_story/story", help="directory for the clips")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--scene", default=None, help="render one scene by name")
    ap.add_argument("--frames", type=float, nargs="*", default=None,
                    help="write PNG stills at these times in the film, then stop")
    ap.add_argument("--check", action="store_true",
                    help="print what the FK solves gave, render nothing")
    args = ap.parse_args()

    model = build_world()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    story = Story(model, data)

    if args.check:
        check(story)
        return

    import imageio.v2 as iio
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def draw(scene, t):
        scene[2](story, t)
        mujoco.mj_kinematics(model, data)
        aim(cam, scene[3], t)
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    if args.frames is not None:
        starts, acc = [], 0.0
        for sc in SCENES:
            starts.append(acc)
            acc += sc[1]
        for want in args.frames:
            i = max(0, sum(1 for st in starts if st <= want) - 1)
            path = os.path.join(out, f"still_{want:06.2f}.png")
            iio.imwrite(path, draw(SCENES[i], want - starts[i]))
            print(f"{path}  ({SCENES[i][0]} + {want - starts[i]:.2f}s)")
        return

    want = [sc for sc in SCENES if args.scene in (None, sc[0])]
    if not want:
        sys.exit(f"no scene called {args.scene!r}")
    whole = None
    if args.scene is None:
        whole = iio.get_writer(os.path.join(out, "story.mp4"), fps=args.fps,
                               quality=8, macro_block_size=None)
    for i, scene in enumerate(SCENES):
        if scene not in want:
            continue
        path = os.path.join(out, f"{i+1}_{scene[0]}.mp4")
        w = iio.get_writer(path, fps=args.fps, quality=8, macro_block_size=None)
        n = int(round(scene[1] * args.fps))
        for k in range(n):
            img = draw(scene, k / args.fps)
            w.append_data(img)
            if whole is not None:
                whole.append_data(img)
        w.close()
        print(f"{path}  {n} frames  {scene[1]:.1f}s")
    if whole is not None:
        whole.close()
        print(f"{os.path.join(out, 'story.mp4')}  "
              f"{sum(sc[1] for sc in SCENES):.1f}s total")


if __name__ == "__main__":
    main()
