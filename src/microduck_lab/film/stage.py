"""The set, the cast and the rig for the duck film.

Everything here is geometry: how the world is built, how a duck is coloured,
and how one duck is posed and dropped onto the floor. The story itself lives in
`film.py`.

The ducks are the real MJCF (`robot_allcollisions_mouth.xml`, the variant with
the 15th servo that opens the beak), attached four times into one world with
`MjSpec.attach` -- twice at full size for the parents and twice at a third
scale for the ducklings, so a duckling is a real microduck and not a cartoon.
"""

import numpy as np

import mujoco

from microduck_lab import paths
from microduck_lab.film import geometry as geo

REPO = str(paths.REPO)
DUCK_XML = paths.model("robot_allcollisions_mouth.xml")

# ---------------------------------------------------------------------------
# Cast and colours
# ---------------------------------------------------------------------------

PINK = np.array([0.97, 0.45, 0.66])
PINK_SOFT = np.array([1.00, 0.82, 0.89])
BLUE = np.array([0.30, 0.53, 0.93])
BLUE_SOFT = np.array([0.74, 0.87, 1.00])
BEAK = np.array([0.99, 0.66, 0.10])
EYE = np.array([0.05, 0.05, 0.08])
METAL = np.array([0.24, 0.25, 0.29])
SHELL = (0.98, 0.94, 0.84)

# prefix -> (main colour, soft colour, scale)
CAST = {
    "she": (PINK, PINK_SOFT, 1.0),
    "he": (BLUE, BLUE_SOFT, 1.0),
    "kidb": (BLUE, BLUE_SOFT, 0.20),
    "kidp": (PINK, PINK_SOFT, 0.20),
}

# Mesh name (after the duck's prefix) -> which colour it takes. Anything not
# listed is a servo, a bearing or a PCB and stays dark, which is what it is.
ROLE = {}
for _n in ("top_head_shell", "bottom_head_shell", "left_shell", "right_shell",
           "trunk_base", "neck", "neck_pitch", "yaw_roll_motion", "yaw2roll",
           "hip_l", "upper_leg_left", "upper_leg_right", "leg", "ankle_left",
           "ankle_right", "upper_leg_rigidity_plate", "motor_support"):
    ROLE[_n] = "main"
for _n in ("jaw", "jaw_soft", "soft_mouth_top", "sole_left", "sole_right",
           "foot_left", "foot_right"):
    ROLE[_n] = "beak"
for _n in ("lens", "noenoeil", "m12_lens_holder"):
    ROLE[_n] = "eye"
ROLE["face_part"] = "soft"

# ---------------------------------------------------------------------------
# Stage layout, in metres. The camera watches from -Y, so the couple face each
# other along X, and the nest sits between them and the lens.
# ---------------------------------------------------------------------------

STAGE_Y = 0.60           # where the two of them meet
NEST = (0.0, 0.0)        # where the family ends up
N_HEARTS = 16

# The nest is a flat woven mat, not a bowl. A raised rim would be something a
# duck's foot lands on, and the film would have to lift the whole duck 25 mm on
# one leg to keep it out of the twigs. At 15 mm the lift is invisible.
NEST_LINING_R = 0.100    # flat brown lining
NEST_LINING_Z = 0.006    # a 6 mm mat: a dragged foot catches on a 12 mm step
NEST_RING_R = 0.125      # the twigs sit on this circle -- outside where she stands to lay
NEST_RING_W = 0.020      # and are solid this far either side of it
NEST_RING_Z = 0.010      # top of a twig, 4 mm above the mat

# Egg size is MEASURED, not chosen twice over. It has to clear two things: a
# curled duckling has to fit inside it (33 x 37 x 48 mm at a quarter scale), and
# it has to fit in the cavity under a sitting duck, whose belly is 69 mm up over
# the middle and whose folded feet are 64 mm out to each side.
EGG_SEMI = (0.017, 0.023)                     # 34 x 46 mm
# Where they lie UNDER HER, in her own frame -- rotated into the world by the
# yaw she broods at. Both numbers are measured, not chosen.
#
# The size: a curled duckling has to have fitted inside (25 x 28 x 37 mm at a
# fifth scale, against a 32 mm opening), and two of them have to fit in the
# cavity under a sitting duck. At 34 mm she settles onto them with no lift at
# all; at 48 mm she is jacked 25 mm up and looks perched on them.
#
# The place: BEHIND her tail. When the sit-stand policy really sits her, the
# lowest point of her body is 19 mm off the floor -- there is no cavity under
# a sitting duck, and an egg under her props her up; the policy, trained from
# a flat sit, then leaps forward standing up and she falls. Her collision
# hull ends 45 mm behind her trunk, so the eggs sit 80 mm behind it: tucked
# against her tail, hidden from a camera in front of her, and never under her.
# (Also why she backs onto the nest: a walking duck treads on anything in
# front of it.)
EGG_LOCAL = ((-0.080, 0.019), (-0.080, -0.019))
BROOD_YAW = -1.28        # the one facing for the whole nest sequence; she faces the camera


def egg_spots():
    """World xy of the two eggs: EGG_LOCAL rotated out of her brooding frame."""
    c, s_ = np.cos(BROOD_YAW), np.sin(BROOD_YAW)
    R = np.array([[c, -s_], [s_, c]])
    return [np.array(NEST) + R @ np.array(p) for p in EGG_LOCAL]


CRADLE_R = 0.024         # posts of straw round each egg: equator 17 mm + 3 mm gap + post radius 4 mm
CRADLE_H = 0.040         # post height above the lining; the egg's equator is at 23 mm, its top at 46
SPLIT = 0.18             # crack height as a fraction of the egg's half-height


def _nest_xml(cx, cy, radius=NEST_RING_R, n=26, seed=7):
    """A woven ring of twigs. Seeded, so two renders match frame for frame.

    Everything here compiles COLLIDABLE (contype 1) and gets its real bits in
    `arm_collisions`: MuJoCo builds the world body's broadphase tree at compile
    time from the geoms that can collide then, and a geom switched on later is
    not in the tree and is never touched. (Planes are tested outside the tree,
    which is why the floor worked all along.)"""
    rng = np.random.default_rng(seed)
    out = [f'<geom name="nest_lining" type="cylinder" pos="{cx} {cy} {NEST_LINING_Z/2:.4f}" '
           f'size="{NEST_LINING_R} {NEST_LINING_Z/2:.4f}" '
           f'rgba="0.44 0.31 0.17 1" contype="1" conaffinity="1"/>']
    # a few loose bits of straw across the lining, so it is a nest and not a
    # brown disc
    for k in range(16):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.0, NEST_LINING_R * 0.92)
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        b = rng.uniform(0, np.pi)
        h = rng.uniform(0.016, 0.030)
        sh = rng.uniform(0.8, 1.2)
        out.append(f'<geom name="straw{k}" type="capsule" size="0.0028" fromto="{x-h*np.cos(b):.4f} '
                   f'{y-h*np.sin(b):.4f} {NEST_LINING_Z+0.001:.4f} {x+h*np.cos(b):.4f} '
                   f'{y+h*np.sin(b):.4f} {NEST_LINING_Z+0.003:.4f}" '
                   f'rgba="{0.62*sh:.3f} {0.47*sh:.3f} {0.26*sh:.3f} 1" '
                   f'contype="1" conaffinity="1"/>')
    for i in range(n):
        a = 2 * np.pi * i / n + rng.uniform(-0.09, 0.09)
        r = radius + rng.uniform(-0.012, 0.012)
        z = 0.0035 + rng.uniform(-0.001, 0.001)   # half buried; the rim is a 4 mm step
        half = rng.uniform(0.026, 0.040)
        dx, dy = -np.sin(a) * half, np.cos(a) * half
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        s = rng.uniform(0.75, 1.15)
        out.append(
            f'<geom name="twig{i}" type="capsule" size="0.0062" fromto="{x-dx:.4f} {y-dy:.4f} {z:.4f} '
            f'{x+dx:.4f} {y+dy:.4f} {z+rng.uniform(-0.002,0.002):.4f}" '
            f'rgba="{0.58*s:.3f} {0.42*s:.3f} {0.23*s:.3f} 1" contype="1" conaffinity="1"/>')
    return "\n    ".join(out)


def _garden_xml(seed=3, tufts=54, flowers=30):
    """Grass and flowers, kept off the two spots the action happens on."""
    rng = np.random.default_rng(seed)
    out = []

    def free(x, y):
        """Clear of the nest, the meeting spot, and the path between them --
        a duck walking through a rigid grass tuft is the same ghost as any
        other."""
        along = np.clip(y / STAGE_Y, 0.0, 1.0)
        return (np.hypot(x - NEST[0], y - NEST[1]) > NEST_RING_R + 0.09
                and np.hypot(x, y - STAGE_Y) > 0.36
                and abs(x - 0.02 * along) > 0.16)

    n = 0
    while n < tufts:
        x, y = rng.uniform(-1.2, 1.2), rng.uniform(-0.8, 1.6)
        if not free(x, y):
            continue
        n += 1
        for _ in range(int(rng.integers(3, 6))):
            h = rng.uniform(0.02, 0.05)
            lx, ly = rng.uniform(-0.012, 0.012, 2)
            g = rng.uniform(0.45, 0.75)
            out.append(f'<geom type="capsule" size="0.0022" fromto="{x:.3f} {y:.3f} 0 '
                       f'{x+lx:.3f} {y+ly:.3f} {h:.3f}" '
                       f'rgba="{0.22*g:.3f} {g:.3f} {0.26*g:.3f} 1" '
                       f'contype="0" conaffinity="0"/>')

    petals = [(0.99, 0.86, 0.32), (0.96, 0.47, 0.62), (0.97, 0.96, 0.99), (0.78, 0.56, 0.95)]
    n = 0
    while n < flowers:
        x, y = rng.uniform(-1.2, 1.2), rng.uniform(-0.8, 1.6)
        if not free(x, y):
            continue
        n += 1
        h = rng.uniform(0.035, 0.062)
        c = petals[int(rng.integers(len(petals)))]
        out.append(f'<geom type="capsule" size="0.0022" fromto="{x:.3f} {y:.3f} 0 '
                   f'{x:.3f} {y:.3f} {h:.3f}" rgba="0.25 0.60 0.30 1" '
                   f'contype="0" conaffinity="0"/>')
        for k in range(5):
            a = 2 * np.pi * k / 5
            out.append(f'<geom type="ellipsoid" pos="{x+0.008*np.cos(a):.4f} '
                       f'{y+0.008*np.sin(a):.4f} {h:.4f}" size="0.006 0.006 0.0018" '
                       f'rgba="{c[0]} {c[1]} {c[2]} 1" contype="0" conaffinity="0"/>')
        out.append(f'<geom type="sphere" pos="{x:.3f} {y:.3f} {h+0.0012:.4f}" size="0.0038" '
                   f'rgba="0.98 0.80 0.20 1" contype="0" conaffinity="0"/>')
    return "\n    ".join(out)


def _cradle_xml():
    """A little cage of upright straws round each egg, with a woven rim.

    An egg standing on its end falls over, and everything round its BASE
    lets it: a tipping egg rides up and over a ball or a horizontal straw at
    its own height, because that contact's normal points up as much as in.
    An upright post gives a horizontal normal, so a tipping egg meets a wall.
    The posts sit 3 mm outside the equator: the egg can rock about seven
    degrees and no further, and cannot roll away. It reads as an egg bedded
    in straw.

    The two eggs sit 38 mm apart and a cage is 56 mm across, so the post on
    the side facing the other egg would stand inside that egg. Those posts
    are left out; on that side the eggs lean on each other.
    """
    spots = egg_spots()
    out = []
    for i, (x, y) in enumerate(spots):
        other = spots[1 - i]
        keep_out = EGG_SEMI[0] + 0.004 + 0.002
        for j in range(8):
            a = 2 * np.pi * j / 8 + 0.2 * i
            cx, cy = x + CRADLE_R * np.cos(a), y + CRADLE_R * np.sin(a)
            if np.hypot(cx - other[0], cy - other[1]) < keep_out:
                continue
            out.append(f'<geom name="cradle{i}_{j}" type="capsule" size="0.0040" '
                       f'fromto="{cx:.4f} {cy:.4f} {NEST_LINING_Z+0.004:.4f} '
                       f'{cx:.4f} {cy:.4f} {NEST_LINING_Z+CRADLE_H:.4f}" '
                       f'rgba="0.70 0.54 0.29 1" contype="1" conaffinity="1"/>')
            # the rim woven between neighbouring posts
            a2 = 2 * np.pi * (j + 1) / 8 + 0.2 * i
            nx, ny = x + CRADLE_R * np.cos(a2), y + CRADLE_R * np.sin(a2)
            if np.hypot(nx - other[0], ny - other[1]) < keep_out:
                continue
            z = NEST_LINING_Z + CRADLE_H - 0.004
            out.append(f'<geom name="cradle{i}_{j+8}" type="capsule" size="0.0035" '
                       f'fromto="{cx:.4f} {cy:.4f} {z:.4f} {nx:.4f} {ny:.4f} {z:.4f}" '
                       f'rgba="0.66 0.50 0.27 1" contype="1" conaffinity="1"/>')
    return "\n    ".join(out)


def _egg_spots(egg_i, top, n=10):
    """Freckles: an unfrecked ellipsoid reads as a pill, not an egg."""
    rng = np.random.default_rng(11 + egg_i + (0 if top else 50))
    a, c = EGG_SEMI
    col = (0.80, 0.66, 0.45) if egg_i == 0 else (0.74, 0.63, 0.48)
    out = []
    for _ in range(n):
        t = rng.uniform(SPLIT + 0.06, 0.90) if top else rng.uniform(-0.90, SPLIT - 0.14)
        u = rng.uniform(0, 2 * np.pi)
        r = geo.egg_profile(t, a, c) * 0.995
        out.append(f'<geom type="ellipsoid" pos="{r*np.cos(u):.4f} {r*np.sin(u):.4f} {c*t:.4f}" '
                   f'size="0.0045 0.0045 0.0018" quat="{np.cos(u/2):.4f} 0 0 {np.sin(u/2):.4f}" '
                   f'rgba="{col[0]} {col[1]} {col[2]} 1" contype="0" conaffinity="0"/>')
    return "".join(out)


def world_xml():
    """Sky, grass, nest, and every prop as a mocap body.

    Props are mocap bodies, not free joints: a mocap body is a pose you write
    each frame, with no state to integrate -- exactly what an animated prop is.
    """
    ring_v, ring_f = geo.torus(radius=0.0125, tube=0.0024)
    top_v, top_f = geo.egg_shell(*EGG_SEMI, split=SPLIT, top=True)
    bot_v, bot_f = geo.egg_shell(*EGG_SEMI, split=SPLIT, top=False)
    heart_v, heart_f = geo.heart(size=0.034)

    meshes = "".join([
        geo.mesh_xml("ring_mesh", ring_v, ring_f),
        geo.mesh_xml("egg_top_mesh", top_v, top_f),
        geo.mesh_xml("egg_bot_mesh", bot_v, bot_f),
        geo.mesh_xml("heart_mesh", heart_v, heart_f),
        geo.mesh_xml("heart_mesh_s", heart_v, heart_f, scale=0.62),
    ])

    hearts = "\n    ".join(
        f'<body name="heart{i}" mocap="true" pos="0 0 -1">'
        f'<geom type="mesh" mesh="{"heart_mesh" if i % 3 else "heart_mesh_s"}" '
        f'rgba="0.95 0.28 0.46 1" contype="0" conaffinity="0"/></body>'
        for i in range(N_HEARTS))

    # Each egg is two rigid bodies, bottom and lid, on free joints, welded to
    # each other until the moment it hatches. The shell mesh is the collision
    # geom too: MuJoCo collides a mesh by its convex hull, and the hull of a
    # closed cap is a cap. The hull is only BUILT for a mesh that some geom can
    # collide with at compile time, so the shells compile collidable here and
    # get their real bits in `arm_collisions`. Parked out of frame until laid.
    eggs = "\n    ".join(
        f'<body name="egg{i}_{half}" pos="{3 + 0.4 * i} 3 0.05">'
        f'<freejoint name="egg{i}_{half}_free"/>'
        f'<geom name="egg{i}_{half}_shell" type="mesh" mesh="egg_{half}_mesh" '
        f'rgba="{SHELL[0]} {SHELL[1]} {SHELL[2]} 1" mass="{0.006 if half == "bot" else 0.004}" '
        f'contype="1" conaffinity="1"/>'
        f'{_egg_spots(i, half == "top")}</body>'
        for i in (0, 1) for half in ("bot", "top"))
    welds = "\n    ".join(
        f'<weld name="egg{i}_weld" body1="egg{i}_bot" body2="egg{i}_top" solref="0.004 1"/>'
        for i in (0, 1))
    # The two halves of one egg interlock along a zigzag, so their convex
    # hulls overlap. Let them collide and the solver blasts the egg apart while
    # the weld holds it together -- it shoots out of the nest. They never touch.
    excludes = "\n    ".join(
        f'<exclude name="egg{i}_halves" body1="egg{i}_bot" body2="egg{i}_top"/>' for i in (0, 1))

    return f"""
<mujoco model="duck_film">
  <visual>
    <headlight diffuse="0.42 0.42 0.44" ambient="0.34 0.32 0.36" specular="0.08 0.08 0.08"/>
    <rgba haze="0.80 0.87 0.96 1"/>
    <!-- The shadow map is spread over `shadowclip * model extent` metres. The
         default clip of 1.5 spread it far wider than anything worth looking
         at: about 3 mm per texel, on a duck whose head is 50 mm across.
         Self-shadow then landed in blocks a sixth of a head wide, and those
         blocks crawled over the face as the duck moved: a flashing stripe
         under her bow, and rectangles appearing and vanishing on the shell.

         The clip must still cover the whole set, or shadows vanish at its
         edges. This set spans 2.35 m, so the map is tightened to 1.0 (2.4 m)
         and doubled to 8192: 0.29 mm per texel, ten times finer. scene_v2 has
         its own block, with its own number for its own extent. -->
    <map shadowclip="1.0" shadowscale="0.7" znear="0.005"/>
    <quality shadowsize="8192" offsamples="8"/>
    <global azimuth="90" elevation="-15" offwidth="1920" offheight="1088"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.40 0.64 0.92" rgb2="0.93 0.95 0.99"
             width="512" height="3072"/>
    <texture type="2d" name="grass" builtin="checker" mark="none"
             rgb1="0.35 0.57 0.29" rgb2="0.31 0.52 0.26" width="300" height="300"/>
    <material name="grass" texture="grass" texuniform="true" texrepeat="16 16" reflectance="0.03"/>
    {meshes}
  </asset>

  <worldbody>
    <light name="sun" pos="0.7 -0.5 2.2" dir="-0.28 0.22 -1" directional="true"
           diffuse="0.62 0.60 0.57" specular="0.12 0.12 0.12" castshadow="true"/>
    <light name="fill" pos="-1.3 -1.5 0.9" dir="0.6 0.7 -0.4" diffuse="0.26 0.23 0.30"
           specular="0 0 0" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="grass"/>
    {_nest_xml(*NEST)}
    {_cradle_xml()}
    {_garden_xml()}
    <body name="ring" pos="0 0 -1">
      <freejoint name="ring_free"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="4e-8 4e-8 4e-8"/>
      <geom type="mesh" mesh="ring_mesh" rgba="0.93 0.94 0.97 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0.0128 0" size="0.0052" rgba="0.55 0.88 1.0 1"
            contype="0" conaffinity="0"/>
    </body>
    <body name="bow" mocap="true" pos="0 0 -1">
      <geom type="ellipsoid" pos="-0.011 0 0" size="0.011 0.0045 0.008" quat="0.966 0 0 0.259"
            rgba="0.92 0.16 0.42 1" contype="0" conaffinity="0"/>
      <geom type="ellipsoid" pos="0.011 0 0" size="0.011 0.0045 0.008" quat="0.966 0 0 -0.259"
            rgba="0.92 0.16 0.42 1" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.0048" rgba="0.99 0.58 0.72 1" contype="0" conaffinity="0"/>
    </body>
    <body name="bowtie" mocap="true" pos="0 0 -1">
      <geom type="ellipsoid" pos="0 -0.010 0" size="0.0045 0.010 0.0075" quat="0.966 0.259 0 0"
            rgba="0.10 0.10 0.15 1" contype="0" conaffinity="0"/>
      <geom type="ellipsoid" pos="0 0.010 0" size="0.0045 0.010 0.0075" quat="0.966 -0.259 0 0"
            rgba="0.10 0.10 0.15 1" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.0038" rgba="0.34 0.10 0.16 1" contype="0" conaffinity="0"/>
    </body>
    {eggs}
    {hearts}
  </worldbody>
  <equality>
    {welds}
  </equality>
  <contact>
    {excludes}
  </contact>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Building the model
# ---------------------------------------------------------------------------


def scale_spec(spec, k):
    """Shrink a whole robot: meshes, link offsets, geom sizes, inertias.

    Scaling the meshes alone shrinks the parts and leaves the skeleton at full
    size, which comes apart at the joints. Every length in the tree has to move.
    """
    for mesh in spec.meshes:
        mesh.scale = np.asarray(mesh.scale) * k
    for body in spec.bodies:
        body.pos = np.asarray(body.pos) * k
        body.ipos = np.asarray(body.ipos) * k
        body.mass = float(body.mass) * k ** 3
        body.inertia = np.asarray(body.inertia) * k ** 5
        for g in body.geoms:
            g.pos = np.asarray(g.pos) * k
            g.size = np.asarray(g.size) * k
        for s in body.sites:
            s.pos = np.asarray(s.pos) * k
        for j in body.joints:
            j.pos = np.asarray(j.pos) * k
        for f in body.frames:
            f.pos = np.asarray(f.pos) * k
    return spec


def scale_actuation(model, prefix, k):
    """Give a scaled-down duck servos and joints to match its size.

    `scale_spec` scales lengths and masses, but the actuator gains, torque
    limits, joint armature, damping and friction stay the adult's. On a 6 g
    duckling an adult head servo (0.96 Nm) is a catapult: one nod throws it
    onto its back. Torque scales as mass x length (k^4), armature as inertia
    (k^5)."""
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if not name.startswith(prefix + "_"):
            continue
        model.actuator_gainprm[a, 0] *= k ** 4
        model.actuator_biasprm[a, 1:3] *= k ** 4
        model.actuator_forcerange[a] *= k ** 4
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith(prefix + "_") or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        dof = model.jnt_dofadr[j]
        model.dof_armature[dof] *= k ** 5
        model.dof_damping[dof] *= k ** 4
        model.dof_frictionloss[dof] *= k ** 4


def build_model(hook=None):
    """Attach the four ducks into the world and compile.

    `hook(world_spec)` runs after the ducks are attached and before compile,
    for constraints that reference duck bodies (the ring's welds)."""
    world = mujoco.MjSpec.from_string(world_xml())
    for prefix, (_, _, scale) in CAST.items():
        duck = mujoco.MjSpec.from_file(DUCK_XML)
        if scale != 1.0:
            scale_spec(duck, scale)
        world.attach(duck, prefix=prefix + "_", frame=world.worldbody.add_frame())
    if hook is not None:
        hook(world)
    model = world.compile()
    paint(model)
    arm_collisions(model)
    return model


def paint(model):
    """Colour each duck by dropping the CAD material off its geoms.

    Every geom in the export carries a material, and a material wins over
    geom rgba -- so the material has to go before a colour will show.
    """
    for g in range(model.ngeom):
        mid = model.geom_dataid[g]
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        prefix, _, part = name.partition("_")
        if prefix not in CAST:
            continue
        main, soft, _ = CAST[prefix]
        rgb = {"main": main, "soft": soft, "beak": BEAK,
               "eye": EYE, "metal": METAL}[ROLE.get(part, "metal")]
        model.geom_matid[g] = -1
        model.geom_rgba[g] = (*rgb, 1.0)


BIT_WORLD, BIT_SHELL, BIT_CRADLE = 16, 32, 64


def arm_collisions(model):
    """Set who may touch whom.

    Contacts use the robot's OWN collision geoms (group 3: head shells, jaw,
    trunk, hips, shins, soles) -- the contact model the real MJCF ships with.
    Bits: duck i is 1<<i; the ground and nest are 16; egg shells 32; the straw
    cradles 64. A duck never tests against itself (its parts overlap at every
    joint by construction). Ducklings pass through shells: they start life
    folded inside one. Nobody collides with the cradle straws but the eggs.
    """
    prefixes = list(CAST)
    for g in range(model.ngeom):
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0
        model.geom_margin[g] = 0.0
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if gname == "floor":
            model.geom_contype[g] = BIT_WORLD
            model.geom_conaffinity[g] = 0xF | BIT_SHELL
            continue
        if gname.startswith("nest_"):
            # The lining: the eggs rest on it, the ducklings stand on it, the
            # parents walk through it -- its 6 mm edge held a walking duck
            # marching on the spot for a minute. Bits 4 and 8 are the ducklings.
            model.geom_contype[g] = BIT_CRADLE
            model.geom_conaffinity[g] = BIT_SHELL | 0xC
            continue
        if gname.startswith(("twig", "straw")):
            # The nest -- lining, woven twigs, loose straw: an egg or a lid
            # rests on it, a duck walks through it. Solid, the 10 mm rim and
            # then the 6 mm lining edge each caught a foot and held a walking
            # duck marching on the spot for a minute. The policies were trained
            # on flat ground; the nest is scenery to them and eggs to the eggs.
            model.geom_contype[g] = BIT_CRADLE
            model.geom_conaffinity[g] = BIT_SHELL
            continue
        if gname.startswith("cradle"):
            # Straw. It holds an egg, and a duck's foot goes through it -- a
            # rigid straw post that trips a duck is the lie, not the other way.
            model.geom_contype[g] = BIT_CRADLE
            model.geom_conaffinity[g] = BIT_SHELL
            continue
        if gname.startswith("egg") and gname.endswith("_shell"):
            model.geom_contype[g] = BIT_SHELL
            model.geom_conaffinity[g] = 0x3 | BIT_WORLD | BIT_SHELL | BIT_CRADLE
            continue
        mid = model.geom_dataid[g]
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0:
            continue
        if model.geom_group[g] != 3:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        prefix = name.partition("_")[0]
        if prefix not in CAST:
            continue
        i = prefixes.index(prefix)
        adult = CAST[prefix][2] > 0.5
        model.geom_contype[g] = 1 << i
        model.geom_conaffinity[g] = ((0xF & ~(1 << i)) | BIT_WORLD
                                     | (BIT_SHELL if adult else 0))


def stiffen(model, kp=3.0, kv=0.06, force=2.5):
    """Stiffer servos than the XL330's own kp of 0.55.

    At the real gain a 0.1 Nm gravity load sags a hip by ten degrees, and with
    no policy compensating, every pose would slump. This keeps the actuators
    position servos with a torque limit; it just makes them hold what they are
    told, the way the walking policy makes the real ones do.
    """
    for a in range(model.nu):
        if model.actuator_gainprm[a, 0] < 1.0:          # the 14 joint servos
            model.actuator_gainprm[a, 0] = kp
            model.actuator_biasprm[a, 1] = -kp
            model.actuator_biasprm[a, 2] = -kv
            model.actuator_forcerange[a] = (-force, force)
            model.actuator_forcelimited[a] = 1


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]

I_LHIP_P, I_LKNEE, I_LANK = 2, 3, 4
I_NECK, I_HEADP, I_HEADY, I_HEADR, I_MOUTH = 5, 6, 7, 8, 9
I_RHIP_P, I_RKNEE, I_RANK = 12, 13, 14


def mirror(leg):
    return tuple(-v for v in leg)


def pose(left=(0, 0, 0, 0, 0), right=None, neck=(0, 0, 0, 0), mouth=0.0):
    """A 15-vector from logical parts.

    Left and right are mirrored in this model: the same physical motion takes
    opposite signs on the two legs. `right=None` mirrors the left leg, which is
    what a symmetric pose wants; pass it explicitly for anything one-sided.
    """
    right = mirror(left) if right is None else right
    return np.array([*left, *neck, mouth, *right], dtype=float)


# Legs, as (hip_yaw, hip_roll, hip_pitch, knee, ankle) for the LEFT leg.
LEG_STAND = (0.0, -0.0873, -0.4579, -0.0049, 0.4530)   # scene.xml STAND
LEG_SIT = (0.0, 0.0, -0.5236, 1.0472, 0.0)             # scene.xml SIT
# The kneel is SOLVED, not drawn: a random search over the six leg angles for
# the lowest trunk that still has both the back leg and the front foot touching
# the floor. It drops him 44 mm, back leg down, front foot planted ahead.
LEG_KNEEL = (0.0, -0.05, 0.818, 1.380, -0.201)         # back leg, folded down
LEG_FRONT = (0.0, -0.12, -1.459, 0.538, 0.899)         # front foot, planted ahead
LEG_CURL = (0.0, 0.0, 1.35, 1.45, 0.0)                 # still inside the egg

NECK_REST = (0.349, 0.349, 0.0, 0.0)
NECK_UP = (0.05, -0.30, 0.0, 0.0)      # chin lifted
NECK_LOW = (0.70, 0.95, 0.0, 0.0)      # looking down

STAND = pose(LEG_STAND, neck=NECK_REST)
SIT = pose(LEG_SIT, neck=(0.50, 0.55, 0, 0))
BROOD = pose(LEG_SIT, neck=(0.42, 0.42, 0, 0))
KNEEL = pose(LEG_KNEEL, right=mirror(LEG_FRONT), neck=NECK_UP, mouth=0.10)
CURL = pose(LEG_CURL, neck=(1.00, 1.20, 0, 0))
PEEK = pose(LEG_CURL, neck=(0.20, 0.10, 0, 0))



# ---------------------------------------------------------------------------
# What holds a duck up
# ---------------------------------------------------------------------------


class Support:
    """The height of the solid world under any point: grass, nest, eggs.

    A kinematic film has no contacts, so nothing stops a duck sinking through
    an egg or a twig -- it just draws one inside the other, and it reads as a
    ghost. This is the contact: a duck is lifted until no point it owns is
    below the surface it is standing on. It is exact along z, which is the axis
    that matters for a duck standing, sitting or brooding.
    """

    def __init__(self):
        self.eggs = []          # (centre, a, c) domes, set per frame

    def clear_eggs(self):
        self.eggs = []

    def add_egg(self, centre, a=None, c=None):
        self.eggs.append((np.asarray(centre, float),
                          EGG_SEMI[0] if a is None else a,
                          EGG_SEMI[1] if c is None else c))

    def ground(self, xy):
        """Grass, nest lining, and the ring of twigs around it."""
        d = np.hypot(xy[:, 0] - NEST[0], xy[:, 1] - NEST[1])
        lining = NEST_LINING_Z * np.clip((NEST_LINING_R - d) / 0.03, 0.0, 1.0)
        ring = NEST_RING_Z * np.sqrt(
            np.clip(1.0 - ((d - NEST_RING_R) / NEST_RING_W) ** 2, 0.0, 1.0))
        return np.maximum(lining, ring)

    def egg_top(self, xy):
        """Top of whichever egg is under each point (-inf where none is).

        The dome is the egg's ellipsoid, which sits slightly ABOVE the tapered
        shell it is standing in for -- so a duck rests a hair clear of the egg
        rather than a hair inside it.
        """
        if not self.eggs:
            return np.full(len(xy), -np.inf)
        best = np.full(len(xy), -np.inf)
        for centre, a, c in self.eggs:
            u = (((xy[:, 0] - centre[0]) / a) ** 2 + ((xy[:, 1] - centre[1]) / a) ** 2)
            inside = u < 1.0
            top = np.where(inside, centre[2] + c * np.sqrt(np.clip(1 - u, 0, 1)), -np.inf)
            best = np.maximum(best, top)
        return best

    def rest_z(self, xy, on_eggs):
        h = self.ground(xy)
        return np.maximum(h, self.egg_top(xy)) if on_eggs else h


SUPPORT = Support()


# ---------------------------------------------------------------------------
# One duck on the stage
# ---------------------------------------------------------------------------


class Duck:
    """Where one duck's joints live in qpos, and how to put it somewhere."""

    def __init__(self, model, data, prefix):
        self.model, self.data, self.prefix = model, data, prefix
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{prefix}_trunk_base_freejoint")
        self.root = int(model.jnt_qposadr[jid])
        self.qadr = np.array([
            int(model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{j}")]) for j in JOINTS])
        self.jadr = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{j}")
            for j in JOINTS])

        self.geoms = [g for g in range(model.ngeom)
                      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                             model.geom_dataid[g]) or "").startswith(prefix + "_")]
        # Vertex clouds for the floor test. Subsampled: the STLs carry thousands
        # of vertices each and the lowest point does not need all of them.
        self.cloud = {}
        for g in self.geoms:
            mid = model.geom_dataid[g]
            adr, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            self.cloud[g] = model.mesh_vert[adr:adr + n][:: max(1, n // 80)].copy()

        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{n}")
        self.trunk, self.head = bid("trunk_base"), bid("jaw_soft")
        self.neck = bid("neck_pitch")
        self.tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_mouth_tip")
        self.scale = CAST[prefix][2]

    # -- posing ------------------------------------------------------------

    def place(self, x, y, yaw, q, lean=(0.0, 0.0), dz=0.0, ground=True, z=0.12,
              on_eggs=False):
        """Pose the duck, then stand it on whatever is under it.

        The pose is clipped to the model's own joint ranges on the way in, so
        no frame of the film can show a joint past a stop the robot has. Then
        the whole duck is lifted until nothing it owns is inside the ground,
        the nest or an egg -- `dz` offsets that resting height.
        """
        d = self.data
        lo, hi = self.model.jnt_range[self.jadr, 0], self.model.jnt_range[self.jadr, 1]
        self.clip_excess = float(np.max(np.maximum(lo - q, q - hi)))
        q = np.clip(q, lo, hi)
        d.qpos[self.root:self.root + 3] = (x, y, z)
        d.qpos[self.root + 3:self.root + 7] = rpy_quat(lean[0], lean[1], yaw)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        if ground:
            flat = float(np.max(-self.points()[:, 2]))    # what a bare floor would give
            self.last_lift = float(self.sink(on_eggs))
            self.rest_z_flat = float(d.qpos[self.root + 2]) + dz + flat
            d.qpos[self.root + 2] += dz + self.last_lift
            mujoco.mj_kinematics(self.model, d)

    def points(self):
        """Every vertex of the duck's skin, in the world, right now."""
        d = self.data
        return np.vstack([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
                          for g, v in self.cloud.items()])

    def sink(self, on_eggs=False):
        """How far the duck is inside the world -- what `place` lifts it by."""
        p = self.points()
        return float(np.max(SUPPORT.rest_z(p[:, :2], on_eggs) - p[:, 2]))

    def lowest(self):
        return float(self.points()[:, 2].min())

    def out_of_range(self, q):
        """Joint targets outside what the servo can actually reach."""
        lo, hi = self.model.jnt_range[self.jadr, 0], self.model.jnt_range[self.jadr, 1]
        bad = (q < lo - 1e-6) | (q > hi + 1e-6)
        return [(JOINTS[i], q[i], lo[i], hi[i]) for i in np.nonzero(bad)[0]]

    # -- reading the posed duck -------------------------------------------

    def frame(self, body):
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def beak(self):
        """World position of the beak tip and the direction it points."""
        p = self.data.site_xpos[self.tip].copy()
        head = self.data.xpos[self.head]
        v = p - head
        return p, v / max(np.linalg.norm(v), 1e-9)

    def head_top(self):
        """Highest point on the head right now -- where a kiss or a bow lands."""
        d, best, bz = self.data, None, -np.inf
        for g in self.cloud:
            if self.model.geom_bodyid[g] != self.head:
                continue
            w = d.geom_xpos[g] + self.cloud[g] @ d.geom_xmat[g].reshape(3, 3).T
            i = int(np.argmax(w[:, 2]))
            if w[i, 2] > bz:
                bz, best = w[i, 2], w[i]
        return best


# ---------------------------------------------------------------------------
# Small maths
# ---------------------------------------------------------------------------


def rpy_quat(roll, pitch, yaw):
    """Body roll, then pitch, then world yaw -- so `roll` is the duck's own."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def mat_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).ravel())
    return q


def frame_from_z(z_axis, up=(0, 0, 1)):
    """A rotation whose local +Z is `z_axis` -- how a ring is threaded on."""
    z = np.asarray(z_axis, float)
    z = z / max(np.linalg.norm(z), 1e-9)
    up = np.asarray(up, float)
    if abs(z @ up) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= max(np.linalg.norm(x), 1e-9)
    return np.column_stack([x, np.cross(z, x), z])


def smooth(u):
    """Ease in and out. Linear moves read robotic; everything here uses this."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def seg(t, t0, t1):
    """Eased progress through [t0, t1], clamped outside it."""
    return 1.0 if t >= t1 else (0.0 if t <= t0 else smooth((t - t0) / (t1 - t0)))


def bump(t, t0, t1):
    """0 -> 1 -> 0 across [t0, t1]: one beat of an action."""
    if t <= t0 or t >= t1:
        return 0.0
    return float(np.sin(np.pi * (t - t0) / (t1 - t0)) ** 2)


def lerp(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
