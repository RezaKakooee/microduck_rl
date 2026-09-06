"""The set for the v2 love story: the earlier scene, built to the physics rules.

Everything that moves after t = 0 is an actuator on a joint. There are no mocap
bodies, no equality constraints, and the story never edits positions,
velocities, collision masks or colours. So each thing the first film faked has
a mechanism here:

* the two eggs are in the world from the start, standing on a motorised lift in
  a POCKET sunk into the ground behind where she will sit. A nest lift raises
  them to nest level, where they appear behind her tail as she sits; a second,
  NARROW cradle inside it then carries the opened shells 110 mm higher, up to
  a standing parent's head, so both parents can reach their chicks;
* the eggs are the physics film's hinged, hollow, collidable capsules, with
  their ducklings inside from t = 0 and a small motor on each lid. The bottom
  half weighs 60 g with its mass low down: at 8 g a parent leaning in to
  nuzzle its chick knocked the egg out of the nest, and at 200 g the lifts
  could not raise it;
* the hearts are scenery on stakes (balloon hearts were tried -- see
  `heart_stakes_xml`);
* the ring rests on a dais: the beak cannot carry it, measured;
* her bow and his bow tie are geoms fixed to the robot bodies.

The ground near the nest is four box slabs around the pocket instead of the
infinite plane, because a plane cannot have a hole in it. Everything the ducks
walk near is either flush with the ground or off their line: they cannot step
over anything -- a 3 mm rim, an 8 mm dais and one 1.5 mm-proud straw each put
one of them on the floor.
"""

import numpy as np
from scipy.spatial import ConvexHull

import mujoco

from microduck_lab.film import geometry as geo
from microduck_lab.film import stage

ALL = 127
BIT_WORLD = 16

# ---------------------------------------------------------------------------
# Layout, in metres. She faces -y when she sits; the camera watches from -y.
# ---------------------------------------------------------------------------

STAGE_Y = 0.60                       # where they meet
NEST = np.array([0.0, 0.0])
SIT = np.array([0.0, -0.02])         # where a parent sits, the eggs behind it
FACING = -np.pi / 2

EGG_R, EGG_H = 0.028, 0.032          # 56 x 64 mm capsules
INNER_FLOOR = -0.020                 # inside the bottom half, where a chick sits
EGG_XY = (np.array([-0.034, 0.10]), np.array([0.034, 0.10]))

POCKET_C = np.array([0.0, 0.10])
POCKET_HALF = np.array([0.072, 0.042])
POCKET_DEPTH = 0.115                 # deep enough to hide the egg tops from the camera
RIM_PROUD = 0.0002                   # how far the nest rim stands above the grass
RIM_H = 0.0015                       # rim half-thickness
RIM_Z = RIM_PROUD - RIM_H            # centre, so the top lands at +RIM_PROUD
NEST_LIFT = POCKET_DEPTH - 0.003     # stage 1: platform top level with the ground
LIFT_TRAVEL = NEST_LIFT              # the nest lift, stage 1
CRADLE_HALF = np.array([0.040, 0.038])   # the narrow inner platform that carries the eggs
CRADLE_RISE = 0.110                      # stage 2: up to a standing parent's head

DAIS = np.array([-0.07, STAGE_Y + 0.15])
RING_R_TWIGS = 0.20


def numbers(v):
    return " ".join("%.6g" % float(x) for x in np.ravel(v))


def mesh_numbers(v):
    """Vertex lists are long; MuJoCo's XML reader rejects sloppy long numbers."""
    return " ".join("%.5f" % float(x) for x in np.ravel(v))


def geom(name, kind, **attrs):
    """A solid world geom: everything in this set collides with everything."""
    a = dict(name=name, type=kind, contype=str(BIT_WORLD), conaffinity=str(ALL),
             solref=".012 1", solimp=".98 .999 .001", friction="1 .005 .0001")
    a.update({k: numbers(v) if isinstance(v, (tuple, list, np.ndarray)) else str(v)
              for k, v in attrs.items()})
    return "<geom " + " ".join(f'{k}="{v}"' for k, v in a.items()) + "/>"


# ---------------------------------------------------------------------------
# Eggs: the physics film's hollow capsules
# ---------------------------------------------------------------------------


def shell_panels(label, top, offset, R=EGG_R, H=EGG_H):
    """Convex wall panels, 1.2 mm thick, that keep the real hollow interior.

    MuJoCo collides a mesh by its convex hull, so one bowl-shaped mesh would be
    a solid egg. 64 small panels are each convex on their own."""
    assets, geoms = [], []
    angles = np.linspace(0 if top else np.arccos(.12), np.arccos(.12) if top else np.pi, 5)
    for row in range(4):
        for sector in range(16):
            pts = []
            for inner in (False, True):
                a, c = R - .0012 * inner, H - .0012 * inner
                for th, ph in ((angles[row], sector * np.pi / 8),
                               (angles[row], (sector + 1) * np.pi / 8),
                               (angles[row + 1], sector * np.pi / 8),
                               (angles[row + 1], (sector + 1) * np.pi / 8)):
                    r = a * np.sin(th) * (1 - .16 * np.cos(th))
                    pts.append(np.array([r * np.cos(ph), r * np.sin(ph), c * np.cos(th)]) - offset)
            v = np.unique(np.round(pts, 10), axis=0)
            hull = ConvexHull(v)
            faces = hull.simplices.copy()
            for i, f in enumerate(faces):
                if np.dot(np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]]), hull.equations[i, :3]) < 0:
                    faces[i] = f[::-1]
            name = f"{label}_panel_{row}_{sector}"
            assets.append(f'<mesh name="{name}" vertex="{mesh_numbers(v)}" '
                          f'face="{" ".join(str(i) for i in faces.ravel())}"/>')
            geoms.append(geom(name, "mesh", mesh=name, rgba=(*stage.SHELL, 1), group=0))
    return assets, geoms


def freckles(i, n=9):
    """Freckles: two identical white capsules read as pills, not eggs."""
    rng = np.random.default_rng(11 + i)
    col = (0.80, 0.66, 0.45) if i == 0 else (0.74, 0.63, 0.48)
    pivot = np.array([0, EGG_R * .96, EGG_H * .12])
    bot, top = [], []
    for _ in range(n):
        th, ph = rng.uniform(0.35, 2.7), rng.uniform(0, 2 * np.pi)
        r = (EGG_R + 0.0004) * np.sin(th) * (1 - .16 * np.cos(th))
        p = np.array([r * np.cos(ph), r * np.sin(ph), (EGG_H + 0.0004) * np.cos(th)])
        g = ('<geom type="ellipsoid" pos="%s" size=".0035 .0035 .0012" quat="%.4f 0 0 %.4f" '
             'rgba="%s %s %s 1" contype="0" conaffinity="0" mass="1e-6" group="0"/>')
        where, rel = (top, p - pivot) if np.cos(th) > .12 else (bot, p)
        where.append(g % (numbers(rel), np.cos(ph / 2), np.sin(ph / 2), *col))
    return "".join(bot), "".join(top)


def egg_xml(i, xy, z):
    assets, bottom = shell_panels(f"egg{i}_bottom", False, np.zeros(3))
    pivot = np.array([0, EGG_R * .96, EGG_H * .12])
    more, top = shell_panels(f"egg{i}_top", True, pivot)
    assets += more
    spot_bot, spot_top = freckles(i)
    body = f'''
    <body name="egg{i}_base" pos="{numbers((*xy, z))}">
      <freejoint name="egg{i}_free"/>
      <inertial pos="0 0 -.014" mass=".06" diaginertia="4e-5 4e-5 3e-5"/>
      {"".join(bottom)}{spot_bot}
      {geom(f"egg{i}_floor", "cylinder", pos=(0, 0, INNER_FLOOR - .001), size=(.022, .001), rgba=(.91, .84, .64, 1))}
      {geom(f"egg{i}_sole", "cylinder", pos=(0, 0, -EGG_H + .001), size=(.014, .001), rgba=(*stage.SHELL, 1))}
      {geom(f"egg{i}_hinge", "capsule", size=(.0025,), fromto=(-.007, pivot[1], pivot[2], .007, pivot[1], pivot[2]), rgba=(.35, .38, .4, 1))}
      <body name="egg{i}_lid" pos="{numbers(pivot)}">
        <joint name="egg_lid_{i}" type="hinge" axis="1 0 0" range="-160 0" damping=".00015" armature=".000001"/>
        <inertial pos="{numbers((0, -EGG_R * .65, .012))}" mass=".0025" diaginertia="8e-7 8e-7 8e-7"/>
        {"".join(top)}{spot_top}
      </body>
    </body>'''
    # The lid opens to 160 deg, not 126: at 126 it stands upright over the egg
    # and is the first thing a parent's head meets. Pressed, it drives past its
    # stop and levers the shell over, tipping the chick inside. Folded right
    # back it lies against the outside of the shell and the top is clear.
    act = (f'<position name="lid_motor_{i}" joint="egg_lid_{i}" kp=".012" kv=".0003" '
           f'ctrlrange="-2.79 0" forcerange="-.0025 .0025"/>')
    return assets, body, act


# ---------------------------------------------------------------------------
# Ground, nest, garden, props
# ---------------------------------------------------------------------------


def ground_xml():
    """Four slabs with the egg pocket between them, tops at z = 0."""
    px, py = POCKET_C
    hx, hy = POCKET_HALF
    T, L = 0.03, 2.5                                   # half thickness, half extent
    out = [
        geom("ground_s", "box", pos=(0, (py - hy - L) / 2, -T), size=(L, (py - hy + L) / 2, T), material="grass", priority="1"),
        geom("ground_n", "box", pos=(0, (py + hy + L) / 2, -T), size=(L, (L - py - hy) / 2, T), material="grass", priority="1"),
        geom("ground_w", "box", pos=((px - hx - L) / 2, py, -T), size=((L + px - hx) / 2, hy, T), material="grass", priority="1"),
        geom("ground_e", "box", pos=((px + hx + L) / 2, py, -T), size=((L - px - hx) / 2, hy, T), material="grass", priority="1"),
    ]
    # A brown rim marks the nest edge, SUNK ALMOST FLUSH: as a 3 mm lip it
    # caught a foot and put her over. The ducks cannot step over anything.
    #
    # "Almost" is the fix for a flicker. Each rim lies inside a ground slab's
    # footprint, so at a top of exactly z = 0 the two faces were coplanar, the
    # depth buffer could not order them, and the whole rectangle flashed on and
    # off as the camera moved. RIM_PROUD lifts the rim off the grass by a fifth
    # of a millimetre: far more than the depth buffer needs to tell them apart
    # (micrometres at this range), and far less than anything that has ever
    # tripped a duck here -- the straw that caused trouble stood 1.5 mm proud.
    for name, pos, size in (("nest_rim_s", (px, py - hy - .012, RIM_Z), (hx + .024, .012, RIM_H)),
                            ("nest_rim_n", (px, py + hy + .012, RIM_Z), (hx + .024, .012, RIM_H)),
                            ("nest_rim_w", (px - hx - .012, py, RIM_Z), (.012, hy, RIM_H)),
                            ("nest_rim_e", (px + hx + .012, py, RIM_Z), (.012, hy, RIM_H))):
        out.append(geom(name, "box", pos=pos, size=size, rgba=(.44, .31, .17, 1)))
    return "\n    ".join(out)


def nest_xml(seed=7):
    """Two woven arcs, east and west, and straw scattered between them."""
    rng = np.random.default_rng(seed)
    out, k = [], 0
    for a0, a1 in ((np.radians(-38), np.radians(38)), (np.radians(142), np.radians(218))):
        for a in np.linspace(a0, a1, 7):
            a += rng.uniform(-.04, .04)
            r = RING_R_TWIGS + rng.uniform(-.01, .01)
            xy = r * np.array([np.cos(a), np.sin(a)])
            t = rng.uniform(.028, .042) * np.array([-np.sin(a), np.cos(a)])
            s = rng.uniform(.8, 1.15)
            out.append(geom(f"twig{k}", "capsule", size=(.006,),
                            fromto=(*(xy - t), .0, *(xy + t), .001),
                            rgba=(.58 * s, .42 * s, .23 * s, 1)))
            k += 1
    # Straw lies FLAT (2 mm capsules on the centreline z = 0.0005) and keeps
    # clear of the pocket.
    for j in range(18):
        xy = None
        for _ in range(60):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(.05, .17)
            p = r * np.array([np.cos(a), np.sin(a)])
            if (abs(p[0] - POCKET_C[0]) > POCKET_HALF[0] + .03
                    or abs(p[1] - POCKET_C[1]) > POCKET_HALF[1] + .03):
                xy = p
                break
        if xy is None:
            continue
        b, h, s = rng.uniform(0, np.pi), rng.uniform(.014, .026), rng.uniform(.8, 1.2)
        out.append(geom(f"straw{j}", "capsule", size=(.002,),
                        fromto=(xy[0] - h * np.cos(b), xy[1] - h * np.sin(b), .0005,
                                xy[0] + h * np.cos(b), xy[1] + h * np.sin(b), .0005),
                        rgba=(.62 * s, .47 * s, .26 * s, 1)))
    return "\n    ".join(out)


def garden_xml(seed=3, tufts=60, flowers=34):
    """Grass and flowers, all solid, all outside the acting area."""
    rng = np.random.default_rng(seed)
    out, n = [], 0
    while n < tufts:
        x, y = rng.uniform(-1.6, 1.6), rng.uniform(-1.0, 1.9)
        if -0.6 < x < 0.6 and -0.55 < y < 1.05:
            continue
        n += 1
        for k in range(int(rng.integers(3, 6))):
            h, (lx, ly), g = rng.uniform(.02, .05), rng.uniform(-.012, .012, 2), rng.uniform(.45, .75)
            out.append(geom(f"grass{n}_{k}", "capsule", size=(.0022,),
                            fromto=(x, y, 0, x + lx, y + ly, h),
                            rgba=(.22 * g, g, .26 * g, 1)))
    petals = [(0.99, 0.86, 0.32), (0.96, 0.47, 0.62), (0.97, 0.96, 0.99), (0.78, 0.56, 0.95)]
    n = 0
    while n < flowers:
        x, y = rng.uniform(-1.6, 1.6), rng.uniform(-1.0, 1.9)
        if -0.6 < x < 0.6 and -0.55 < y < 1.05:
            continue
        n += 1
        h, c = rng.uniform(.035, .062), petals[int(rng.integers(4))]
        out.append(geom(f"stem{n}", "capsule", size=(.0022,), fromto=(x, y, 0, x, y, h), rgba=(.25, .6, .3, 1)))
        for k in range(5):
            a = 2 * np.pi * k / 5
            out.append(geom(f"petal{n}_{k}", "ellipsoid",
                            pos=(x + .008 * np.cos(a), y + .008 * np.sin(a), h),
                            size=(.006, .006, .0018), rgba=(*c, 1)))
        out.append(geom(f"pistil{n}", "sphere", pos=(x, y, h + .0012), size=(.0038,), rgba=(.98, .8, .2, 1)))
    return "\n    ".join(out)


def heart_stakes_xml():
    """Heart-topped stakes: the film's motif, as scenery.

    Balloon hearts were tried first -- free bodies with gravity compensation,
    packed in a box with a motorised lid. Two grams of thin extruded mesh
    sheared by a lid and by each other went unstable inside a second, every
    time. A prop that stands still says the same thing and cannot destabilise
    the scene. They stand off every walking line: an 8 mm dais in the middle of
    the meeting spot tripped him."""
    hv, hf = geo.heart(size=0.026)
    mesh = (f'<mesh name="heart_mesh" vertex="{mesh_numbers(hv)}" '
            f'face="{" ".join(str(k) for k in hf.ravel())}"/>')
    out = []
    for k, (x, y, h, yaw) in enumerate(((DAIS[0] - .10, DAIS[1] + .03, .10, .35),
                                        (DAIS[0] + .10, DAIS[1] + .03, .085, -.35),
                                        (-.55, -.28, .095, .8),
                                        (.52, -.22, .085, -.8))):
        out.append(geom(f"stake{k}", "capsule", size=(.0025,), fromto=(x, y, 0, x, y, h),
                        rgba=(.35, .52, .30, 1)))
        out.append(geom(f"heart{k}", "mesh", mesh="heart_mesh", pos=(x, y, h + .020),
                        quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)), rgba=(.95, .28, .46, 1)))
    return mesh, "\n    ".join(out)


def world_xml():
    assets, bodies, acts = [], [], []
    for i, xy in enumerate(EGG_XY):
        a, b, c = egg_xml(i, xy, -POCKET_DEPTH + 0.003 + EGG_H)      # on the lowered lift
        assets += a
        bodies.append(b)
        acts.append(c)
    heart_mesh, heart_props = heart_stakes_xml()

    ring_segs = "".join(
        geom(f"ring_seg{k}", "capsule", size=(.0024,),
             fromto=(.0125 * np.cos(2 * np.pi * k / 24), .0125 * np.sin(2 * np.pi * k / 24), 0,
                     .0125 * np.cos(2 * np.pi * (k + 1) / 24), .0125 * np.sin(2 * np.pi * (k + 1) / 24), 0),
             rgba=(.95, .78, .22, 1), friction="1.8 .005 .0001") for k in range(24))
    ring = f'''
    <body name="ring" pos="{numbers((*DAIS, .0075))}">
      <freejoint name="ring_free"/>
      <inertial pos="0 0 0" mass=".001" diaginertia="1e-7 1e-7 1e-7"/>
      {ring_segs}
      {geom("ring_gem", "sphere", pos=(0, .014, 0), size=(.004,), rgba=(.55, .88, 1, 1))}
    </body>'''

    # Guide fingers ride ON the lift, so they hold the eggs the whole way up.
    # Unguided, a rising egg has nothing to lean on and topples (measured:
    # upright 1.00 -> 0.28). They stop at the egg's equator: taller ones stand
    # in the arc the lid sweeps and the lid cannot open. Posts that would stand
    # inside the other egg are left out, and so are the two on the side each
    # parent leans in from: at full height their tops sat 6 mm under his beak
    # and his jaw caught them before it reached the chick.
    guides = []
    for i, xy in enumerate(EGG_XY):
        other = EGG_XY[1 - i]
        outward = -1.0 if i == 0 else 1.0            # west for egg 0, east for egg 1
        for j, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
            p = xy - POCKET_C + (EGG_R + .005) * np.array([np.cos(a), np.sin(a)])
            if np.linalg.norm(p + POCKET_C - other) < EGG_R + .010:
                continue
            if outward * np.cos(a) > 0.7:
                continue                              # leave the parent's side open
            guides.append(geom(f"guide_{i}_{j}", "capsule", size=(.0015,),
                               fromto=(*p, .006, *p, .020), rgba=(.55, .59, .62, 1),
                               friction=".1 .001 .0001"))
    px, py = POCKET_C
    # Two nested lifts. The outer one is the pocket floor, in two halves with a
    # slot between them; the inner one fills the slot, carries the eggs and the
    # guide fingers, and at the end rises 110 mm to a standing parent's head.
    # It is narrow on purpose: a full-width platform at that height is a table
    # the parents walk into, and the walking policy cannot stop precisely
    # enough beside it -- that wrecked the nest every time.
    ox = (POCKET_HALF[0] - .004 + CRADLE_HALF[0]) / 2
    ow = (POCKET_HALF[0] - .004 - CRADLE_HALF[0]) / 2
    lift = f'''
    <body name="lift" pos="{numbers((px, py, -POCKET_DEPTH))}">
      <joint name="lift_slide" type="slide" axis="0 0 1" range="0 {LIFT_TRAVEL}" damping="4"/>
      {geom("lift_plate_w", "box", pos=(-ox, 0, 0), size=(ow, POCKET_HALF[1] - .004, .003), mass=".02", rgba=(.44, .31, .17, 1))}
      {geom("lift_plate_e", "box", pos=(ox, 0, 0), size=(ow, POCKET_HALF[1] - .004, .003), mass=".02", rgba=(.44, .31, .17, 1))}
      {geom("lift_column", "box", pos=(-ox, 0, -.055), size=(.014, .014, .052), mass=".01", rgba=(.30, .46, .62, 1))}
      {geom("lift_column_e", "box", pos=(ox, 0, -.055), size=(.014, .014, .052), mass=".01", rgba=(.30, .46, .62, 1))}
      <body name="cradle" pos="0 0 0">
        <joint name="cradle_slide" type="slide" axis="0 0 1" range="0 {CRADLE_RISE}" damping="3"/>
        {geom("cradle_plate", "box", size=(CRADLE_HALF[0], CRADLE_HALF[1], .003), mass=".03", rgba=(.50, .36, .20, 1))}
        {geom("cradle_column", "box", pos=(0, 0, -.050), size=(.014, .012, .047), mass=".02", rgba=(.30, .46, .62, 1))}
        {"".join(guides)}
      </body>
    </body>'''
    acts.append(f'<position name="cradle_motor" joint="cradle_slide" kp="200" kv="16" '
                f'ctrlrange="0 {CRADLE_RISE}" forcerange="-8 8"/>')
    # kp 200 held the loaded lift 12 mm below its commanded height -- the
    # cradle, its column and two 60 g eggs hang on it, about 2 N, and 2 N over
    # kp 200 is 10 mm of sag. That left the egg pocket a 12 mm hole in the
    # ground right behind where a parent sits, and sitting down carries a duck
    # 73 mm backwards. She stood up out of the nest and went into it.
    #
    # At kp 1200 the same load sags under 2 mm and the pocket is a floor. The
    # old warning about a stiff lift launching the eggs was about a stiff
    # motor chasing a step command; this one follows a 6 s ramp, and the 8 N
    # limit is unchanged.
    acts.append(f'<position name="lift_motor" joint="lift_slide" kp="200" kv="16" '
                f'ctrlrange="0 {LIFT_TRAVEL}" forcerange="-8 8"/>')

    return f"""
<mujoco model="duck_love_story_v2">
  <option timestep="0.005" integrator="implicitfast" iterations="100"/>
  <visual>
    <headlight diffuse="0.42 0.42 0.44" ambient="0.34 0.32 0.36" specular="0.08 0.08 0.08"/>
    <rgba haze="0.80 0.87 0.96 1"/>
    <!-- The shadow map is spread over `shadowclip * model extent` metres. The
         ground here is four 5 m slabs, so the extent is 8.1 m and at the
         default clip of 1.5 the map covered 12 m: 3 mm per texel, on a duck
         whose head is 50 mm across. Self-shadow then landed in blocks about
         a sixth of a head wide, and those blocks crawled over the face as she
         moved: a flashing stripe under her bow, and rectangles appearing and
         vanishing on the shell. Tightening the map to 4 m and doubling it to
         8192 gives 0.5 mm per texel, six times finer, and measurably keeps the
         ground shadows (31.8k shadowed pixels against 31.0k before). The clip
         must still cover the whole set or shadows vanish at its edges; this
         one spans about 1.5 m, well inside 4 m. -->
    <map shadowclip="0.5" shadowscale="0.7" znear="0.005"/>
    <quality shadowsize="8192" offsamples="8"/>
    <global azimuth="90" elevation="-15" offwidth="1920" offheight="1088"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.40 0.64 0.92" rgb2="0.93 0.95 0.99" width="512" height="3072"/>
    <texture type="2d" name="grass" builtin="checker" mark="none" rgb1="0.35 0.57 0.29" rgb2="0.31 0.52 0.26" width="300" height="300"/>
    <material name="grass" texture="grass" texuniform="true" texrepeat="16 16" reflectance="0.03"/>
    {"".join(assets)}
    {heart_mesh}
  </asset>
  <worldbody>
    <light name="sun" pos="0.7 -0.5 2.2" dir="-0.28 0.22 -1" directional="true" diffuse="0.62 0.60 0.57" specular="0.12 0.12 0.12" castshadow="true"/>
    <light name="fill" pos="-1.3 -1.5 0.9" dir="0.6 0.7 -0.4" diffuse="0.26 0.23 0.30" specular="0 0 0" castshadow="false"/>
    {ground_xml()}
    {nest_xml()}
    {garden_xml()}
    {geom("dais", "cylinder", pos=(*DAIS, .0015), size=(.030, .0015), rgba=(.65, .35, .42, 1))}
    {ring}
    {lift}
    {heart_props}
    {"".join(bodies)}
  </worldbody>
  <actuator>
    {"".join(acts)}
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# The ducks, with her bow and his bow tie built in
# ---------------------------------------------------------------------------

BOW_LOCAL = np.array([0.0442, -0.0075, -0.0481])     # top of her head, in the jaw_soft frame


def dress(world):
    """Fixed, non-colliding geoms on the robot bodies: her bow, his bow tie."""
    bodies = {b.name: b for b in world.bodies}

    def add(body, kind, pos, size, rgba, quat=(1, 0, 0, 0)):
        g = body.add_geom()
        g.type = kind
        g.pos = np.asarray(pos, float)
        g.size = np.asarray(size, float)
        g.quat = np.asarray(quat, float)
        g.rgba = np.asarray(rgba, float)
        g.contype = g.conaffinity = 0
        g.group = 2
        g.mass = 0.0005

    ell = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    sph = mujoco.mjtGeom.mjGEOM_SPHERE
    head = bodies["she_jaw_soft"]
    up_l = np.array([1.0, 0.0, 0.0])          # world +z, in the head frame
    side_l = np.array([0.0, 1.0, 0.0])        # world +y, in the head frame
    centre = BOW_LOCAL + 0.004 * up_l
    loop_q = stage.mat_quat(stage.frame_from_z(side_l, up=up_l))
    add(head, ell, centre + 0.011 * side_l, (0.0045, 0.008, 0.011), (.92, .16, .42, 1), loop_q)
    add(head, ell, centre - 0.011 * side_l, (0.0045, 0.008, 0.011), (.92, .16, .42, 1), loop_q)
    add(head, sph, centre, (0.0048, 0, 0), (.99, .58, .72, 1))
    trunk = bodies["he_trunk_base"]
    add(trunk, ell, (0.045, -0.010, 0.030), (0.0045, 0.010, 0.0075), (.10, .10, .15, 1), (0.966, 0.259, 0, 0))
    add(trunk, ell, (0.045, 0.010, 0.030), (0.0045, 0.010, 0.0075), (.10, .10, .15, 1), (0.966, -0.259, 0, 0))
    add(trunk, sph, (0.045, 0.0, 0.030), (0.0038, 0, 0), (.34, .10, .16, 1))


def build_model():
    world = mujoco.MjSpec.from_string(world_xml())
    for prefix, (_, _, scale) in stage.CAST.items():
        duck = mujoco.MjSpec.from_file(stage.DUCK_XML)
        if scale != 1:
            stage.scale_spec(duck, scale)
        world.attach(duck, prefix=prefix + "_", frame=world.worldbody.add_frame())
    dress(world)
    model = world.compile()
    stage.paint(model)
    # The robots' own collision set (group 3) collides with everything but
    # itself; their visual meshes collide with nothing; the world is solid.
    for g in range(model.ngeom):
        bodyname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        prefix = bodyname.split("_")[0]
        if prefix in stage.CAST:
            if model.geom_group[g] == 3:
                bit = 1 << list(stage.CAST).index(prefix)
                model.geom_contype[g] = bit
                model.geom_conaffinity[g] = ALL ^ bit
            else:
                model.geom_contype[g] = model.geom_conaffinity[g] = 0
        model.geom_margin[g] = 0
    for prefix, (_, _, scale) in stage.CAST.items():
        if scale < 1:
            stage.scale_actuation(model, prefix, scale)
    from bam.model import load_model
    torque = load_model(motor_name="xl330", model="m6").kt.value * 1.75
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if name.startswith(("he_", "she_")):
            model.actuator_forcerange[a] = (-torque, torque)
            model.actuator_forcelimited[a] = 1
    return model, mujoco.MjData(model)
