#!/usr/bin/env python3
"""Two microducks fall in love -- a rendered short film.

    1  meet     14 s   he walks up to her with a ring in his beak
    2  propose  15 s   he goes down on one knee and offers it
    3  kiss     12 s   she says yes, the ring goes round her neck, they kiss
    4  eggs     18 s   she sits in the nest and lays two eggs; he kisses her head
    5  brood    16 s   she sits on the eggs, then they swap and he sits on them
    6  hatch    24 s   both eggs crack open; a blue and a pink duckling come out

THIS IS ANIMATION, NOT SIMULATION. `mj_step` is never called: no policy, no
torque, no contact force takes part. Every frame writes `qpos` and calls
`mj_kinematics`, the way an animator poses a rig. So the film says nothing
about what the robot can do -- it is a story told with the real robot model.

What is still honest, because it was free to be:

* the ducks are `robot_allcollisions_mouth.xml`, the variant with the 15th
  servo, so the beak really opens and the ring hangs off the real `mouth_tip`;
* every pose is inside the model's own joint limits (`--check` proves it);
* every duck is GROUNDED by forward kinematics each frame -- its trunk height
  is solved so its lowest vertex rests on the floor. The walk, the kneel and
  the sit all get their height from that, not from a tuned number;
* where they stand to kiss, and where he stands to kiss her head, are solved
  from beak-tip FK, not eyeballed;
* the ducklings are the same MJCF at a third scale, so a duckling is a real
  microduck.

    src/microduck_lab/film/film.py --check                  # no GL needed
    MUJOCO_GL=egl src/microduck_lab/film/film.py --sheet videos/love_story/duckfilm_sheet.png
    MUJOCO_GL=egl src/microduck_lab/film/film.py --out videos/love_story/duckfilm.mp4

Rendering needs EGL, so it needs a GPU:

    bash src/microduck_lab/render/render.sh film
"""

import argparse

import numpy as np

import mujoco

from microduck_lab.film import stage
from microduck_lab.film.stage import (BROOD, CURL, KNEEL, PEEK, SIT, STAND, Duck, bump, frame_from_z,  # noqa: E402
                   lerp, mat_quat, rpy_quat, seg, smooth, wrap)

FPS = 30

# ---------------------------------------------------------------------------
# The clock. One number per beat, so retiming is a one-line edit.
# ---------------------------------------------------------------------------

T_MEET = 0.0
T_PROPOSE = 14.0
T_KISS = 29.0
T_WALK_NEST = 41.0
T_EGGS = 47.0
T_BROOD = 65.0
T_HATCH = 95.0
T_END = 121.0

# ---------------------------------------------------------------------------
# Walking. Signs come from the model: on the LEFT leg a negative hip pitch
# swings the foot forward and a positive knee lifts it, and the right leg is
# mirrored -- so both legs take the SAME delta and move in opposite directions,
# which is exactly a stride.
# ---------------------------------------------------------------------------

STEP = 0.055        # metres per half stride
HIP_SWING = 0.42
KNEE_LIFT = 0.55
BODY_ROLL = 0.16    # the waddle
FULL_SPEED = 0.16   # speed at which the gait reaches full amplitude


class Track:
    """A duck's motion as keyframes, with a walk cycle laid on top.

    A key is a whole state: where the duck is, which way it faces, its pose,
    its lean, and how high off the floor it sits. Between keys everything is
    smoothstepped. The gait is not keyframed -- it is driven by how fast the
    duck is actually moving, so a duck that moves walks and a duck that stops
    stops walking.
    """

    def __init__(self, duck, x, y, yaw, q=STAND, lean=(0.0, 0.0), dz=0.0, look=None,
                 stride=True, eggs=False):
        self.duck = duck
        self.keys = [dict(t=0.0, x=x, y=y, yaw=yaw, q=np.array(q, float),
                          lean=np.array(lean, float), dz=dz, look=look, stride=stride,
                          eggs=eggs)]
        self.phase = 0.0
        self.breathe = np.random.default_rng(abs(hash(duck.prefix)) % 1000).uniform(0, 6.3)

    def to(self, t, x=None, y=None, yaw=None, q=None, lean=None, dz=None,
           look=..., stride=None, eggs=None):
        k = dict(self.keys[-1])
        k["t"] = t
        if x is not None:
            k["x"] = x
        if y is not None:
            k["y"] = y
        if yaw is not None:                      # always turn the short way
            k["yaw"] = k["yaw"] + wrap(yaw - k["yaw"])
        if q is not None:
            k["q"] = np.array(q, float)
        if lean is not None:
            k["lean"] = np.array(lean, float)
        if dz is not None:
            k["dz"] = dz
        if look is not ...:
            k["look"] = look
        if stride is not None:
            k["stride"] = stride
        if eggs is not None:
            k["eggs"] = eggs
        self.keys.append(k)
        return self

    def hold(self, t):
        return self.to(t)

    def sample(self, t):
        keys = self.keys
        if t <= keys[0]["t"]:
            return dict(keys[0])
        for a, b in zip(keys, keys[1:]):
            if t <= b["t"]:
                u = seg(t, a["t"], b["t"])
                s = dict(b)
                s["x"], s["y"] = lerp(a["x"], b["x"], u), lerp(a["y"], b["y"], u)
                s["yaw"] = lerp(a["yaw"], b["yaw"], u)
                s["q"] = lerp(a["q"], b["q"], u)
                s["lean"] = lerp(a["lean"], b["lean"], u)
                s["dz"] = lerp(a["dz"], b["dz"], u)
                if a["look"] is not None and b["look"] is not None:
                    s["look"] = tuple(lerp(a["look"], b["look"], u))
                    s["look_w"] = 1.0
                elif a["look"] is not None:
                    s["look"], s["look_w"] = a["look"], 1.0 - u
                elif b["look"] is not None:
                    s["look"], s["look_w"] = b["look"], u
                else:
                    s["look_w"] = 0.0
                return s
        s = dict(keys[-1])
        s["look_w"] = 1.0 if s["look"] is not None else 0.0
        return s

    def apply(self, t, dt):
        s = self.sample(t)
        p = self.sample(max(0.0, t - dt))
        speed = float(np.hypot(s["x"] - p["x"], s["y"] - p["y"]) / max(dt, 1e-6))
        q = np.array(s["q"], float)
        lean = np.array(s["lean"], float)

        # Anywhere near the nest, the eggs hold a duck up -- see `on_eggs`.
        # A duckling is a fifth of the size, so its stride is a fifth as long:
        # at the shared constants it would take one giant step per second.
        k = self.duck.scale
        gait = min(1.0, speed / (FULL_SPEED * k)) if s.get("stride", True) else 0.0
        if gait > 0.01:
            self.phase += np.pi * speed * dt / (STEP * k)
            sw, co = np.sin(self.phase), np.cos(self.phase)
            dhip = -HIP_SWING * gait * sw
            lift_l, lift_r = max(0.0, co) ** 1.5, max(0.0, -co) ** 1.5
            dkl, dkr = KNEE_LIFT * gait * lift_l, -KNEE_LIFT * gait * lift_r
            q[stage.I_LHIP_P] += dhip
            q[stage.I_RHIP_P] += dhip
            q[stage.I_LKNEE] += dkl
            q[stage.I_RKNEE] += dkr
            q[stage.I_LANK] -= dhip + dkl        # keep the sole flat
            q[stage.I_RANK] -= dhip + dkr
            lean[0] += BODY_ROLL * gait * co
            q[stage.I_NECK] += 0.05 * gait * np.sin(2 * self.phase)
        else:
            b = 0.012 * np.sin(2 * np.pi * 0.32 * t + self.breathe)   # breathing
            q[stage.I_LKNEE] += b
            q[stage.I_RKNEE] -= b
            q[stage.I_LANK] -= b
            q[stage.I_RANK] += b

        self.duck.place(s["x"], s["y"], s["yaw"], q, lean=lean, dz=s["dz"],
                        on_eggs=self.on_eggs(s))

        # Looking is done after the body is posed, so it can aim at something
        # that has moved this frame.
        if s.get("look_w", 0.0) > 0.01:
            self._look(s["look"], s["look_w"], q, s)

    def on_eggs(self, s):
        """A grown duck close to the nest stands on whatever eggs are under it.

        This is not keyframed on purpose. A flag that says "she is brooding
        now" is off during the step that carries a foot over an egg, which is
        exactly the frame that needs it -- and a duck that steps over its own
        eggs, lifting a little as it goes, is what a duck does.
        """
        if self.duck.scale < 0.5:
            return False
        return float(np.hypot(s["x"] - stage.NEST[0], s["y"] - stage.NEST[1])) < 0.34

    def _look(self, target, w, q, s):
        """Aim the head at a point, keeping whatever the pose was doing on top.

        The aim is a DELTA on the pose's own head angles, not a replacement:
        at the rest angle the beak already points level, so an aim of zero must
        leave the head where the pose put it. Whatever that stacks up to is
        clipped to the servo's range inside `place`.
        """
        d = self.duck
        head = d.data.xpos[d.head]
        v = np.asarray(target, float) - head
        yaw = np.clip(wrap(np.arctan2(v[1], v[0]) - s["yaw"]), -1.1, 1.1)
        pitch = np.clip(-np.arctan2(v[2], np.hypot(v[0], v[1])), -0.50, 0.90)
        q = np.array(q)
        q[stage.I_HEADP] += w * pitch
        q[stage.I_HEADY] += w * yaw
        d.place(s["x"], s["y"], s["yaw"], q, lean=s["lean"], dz=s["dz"],
                on_eggs=self.on_eggs(s))


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------


class Prop:
    def __init__(self, model, data, name):
        self.model, self.data = model, data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.mid = int(model.body_mocapid[bid])
        self.geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]

    def set(self, pos, quat=(1, 0, 0, 0), alpha=1.0):
        self.data.mocap_pos[self.mid] = pos
        self.data.mocap_quat[self.mid] = quat
        for g in self.geoms:
            self.model.geom_rgba[g, 3] = alpha

    def hide(self):
        self.set((0, 0, -1.0), alpha=0.0)


def set_alpha(model, geoms, a):
    for g in geoms:
        model.geom_rgba[g, 3] = a


# ---------------------------------------------------------------------------
# The film
# ---------------------------------------------------------------------------


class Film:
    def __init__(self):
        self.model = stage.build_model()
        self.data = mujoco.MjData(self.model)
        self.render_data = self.data
        self.ducks = {p: Duck(self.model, self.data, p) for p in stage.CAST}
        self.props = {n: Prop(self.model, self.data, n) for n in ["ring", "bow", "bowtie"]}
        self.hearts = [Prop(self.model, self.data, f"heart{i}") for i in range(stage.N_HEARTS)]
        self.solve_marks()
        self.build_tracks()

    # -- geometry solved once, before anything is keyframed ----------------

    def solve_marks(self):
        """Positions the story needs measured rather than guessed."""
        she, he = self.ducks["she"], self.ducks["he"]

        def beak_reach(duck, q, lean=(0.0, 0.0)):
            """How far in front of the trunk the beak tip lands, and how high."""
            duck.place(0.0, 0.0, 0.0, q, lean=lean)
            tip, _ = duck.beak()
            trunk = duck.data.xpos[duck.trunk]
            return float(tip[0] - trunk[0]), float(tip[2])

        self.KISS_LEAN = (0.0, -0.26)      # both lean in, nose forward
        r_he, z_he = beak_reach(he, kiss_pose(), self.KISS_LEAN)
        r_she, z_she = beak_reach(she, kiss_pose(), self.KISS_LEAN)
        self.kiss_x = (r_he + r_she) / 2.0 - 0.002       # tips just touching
        self.kiss_z = (z_he + z_she) / 2.0

        # Where she is when the eggs come out, and where they land: the eggs
        # drop behind her, so she lays them INTO the nest and then turns round.
        # One facing for the whole nest sequence, chosen so she faces the
        # camera. She lays in front of the nest, the eggs come out behind her
        # into it, and she BACKS onto them -- a duck that walks forward onto
        # its own eggs treads on them, whichever way round it goes.
        self.brood_yaw = stage.BROOD_YAW
        self.axis = np.array([np.cos(self.brood_yaw), np.sin(self.brood_yaw)])
        self.lay_yaw = self.brood_yaw
        self.lay_xy = np.array(stage.NEST) + 0.050 * self.axis   # eggs land just behind her tail; feet inside the rim
        self.tail_dir = -self.axis
        self.nest_in = np.array(stage.NEST) + 0.20 * self.axis   # wait in front
        self.nest_out = np.array(stage.NEST) + 0.33 * self.axis  # and leave that way
        self.side_l = np.array([-0.30, -0.10])                   # swing wide of the nest
        self.side_r = np.array([0.30, -0.10])

        # He kisses the top of her head while she sits. The beak TIP is the
        # wrong thing to aim: a tip placed on her head puts the head behind it
        # inside hers. So this is solved on the meshes -- he starts on top of
        # her and backs away along one line until MuJoCo's collision detector
        # says no part of him is inside any part of her. Where he stops, his
        # jaw is resting on her head shell. The approach angle and the pose are
        # the ones a search over both found that made the FIRST touch head to
        # head rather than his foot on hers.
        she.place(*self.lay_xy, self.lay_yaw, SIT)
        self.her_head = she.head_top()
        self.head_kiss_pose = STAND.copy()
        self.head_kiss_pose[stage.I_NECK] = 0.05
        self.head_kiss_pose[stage.I_HEADP] = 0.50
        self.head_kiss_pose[stage.I_MOUTH] = 0.06
        self.head_kiss_lean = (0.0, -0.25)
        ang = self.lay_yaw + np.radians(-80.0)
        away = np.array([np.cos(ang), np.sin(ang)])
        self.head_kiss_yaw = float(np.arctan2(-away[1], -away[0]))
        self.head_kiss_xy, self.head_kiss_touch = self.slide_to_touch(
            he, self.head_kiss_pose, self.head_kiss_lean, self.head_kiss_yaw,
            self.her_head[:2], away, "she")
        # where he waits before stepping in: straight back along his own facing
        face = np.array([np.cos(self.head_kiss_yaw), np.sin(self.head_kiss_yaw)])
        self.head_kiss_wait = self.head_kiss_xy - 0.14 * face

        # The eggs go where they will be UNDER her when she turns round and
        # settles: their spots are given in her brooding frame and rotated out.
        cb, sb = np.cos(self.brood_yaw), np.sin(self.brood_yaw)
        R = np.array([[cb, -sb], [sb, cb]])
        self.egg_xy = stage.egg_spots()
        self.egg_pos = [np.array([xy[0], xy[1], stage.NEST_LINING_Z + stage.EGG_SEMI[1]])
                        for xy in self.egg_xy]
        self.egg_top_z = stage.NEST_LINING_Z + 2 * stage.EGG_SEMI[1]

        # A lid lands tipped over. Transform its own vertices by the resting
        # tilt and read the lowest one, so it comes to rest ON the grass.
        mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, "egg_top_mesh")
        adr = int(self.model.mesh_vertadr[mid])
        n = int(self.model.mesh_vertnum[mid])
        verts = self.model.mesh_vert[adr:adr + n]
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, rpy_quat(0.0, 2.5, 0.7))
        self.lid_rest_z = float(-(verts @ Rm.reshape(3, 3).T)[:, 2].min())

        # Where a lid comes to rest: far enough out to be clear of the nest,
        # and high enough that the shell's own rim sits on the grass.
        self.lid_land = [np.array([*(np.array(stage.NEST) + R @ np.array([-0.03, s * 0.30])),
                                   0.012]) for s in (1, -1)]

    # -- the timeline ------------------------------------------------------

    def build_tracks(self):
        she, he = self.ducks["she"], self.ducks["he"]
        Y = stage.STAGE_Y
        her_home = (self.kiss_x + 0.085, Y)
        his_home = (-(self.kiss_x + 0.085), Y)
        her_eye = (her_home[0], Y, 0.24)
        his_eye = (his_home[0], Y, 0.24)

        # ---- SHE ----------------------------------------------------------
        s = Track(she, her_home[0], Y, np.pi, look=(0.35, Y + 0.25, 0.05))
        s.to(4.0, look=(0.30, Y - 0.30, 0.03))          # busy, hasn't seen him
        s.to(7.5, look=(his_home[0], Y, 0.26))          # notices
        s.to(9.5, lean=(0.0, 0.10), look=his_eye)       # leans back a little
        s.to(T_PROPOSE + 1.0, lean=(0.0, 0.0))
        s.to(T_PROPOSE + 5.0, look=(-0.02, Y, 0.10))    # looks at the ring
        s.to(T_PROPOSE + 9.0, q=lean_pose(0.35), look=(-0.02, Y, 0.10))
        s.to(T_PROPOSE + 12.0, q=STAND, look=his_eye)
        # yes: three nods
        s.to(T_KISS + 0.5, q=nod_pose(0.55))
        s.to(T_KISS + 1.1, q=nod_pose(-0.10))
        s.to(T_KISS + 1.7, q=nod_pose(0.55))
        s.to(T_KISS + 2.3, q=nod_pose(-0.10))
        s.to(T_KISS + 3.0, q=STAND, look=his_eye)
        s.to(T_KISS + 6.0, x=self.kiss_x, q=kiss_pose(), lean=self.KISS_LEAN, look=None)
        s.to(T_KISS + 10.5)                                    # the kiss holds
        s.to(T_KISS + 12.0, x=her_home[0], q=STAND, lean=(0, 0), look=his_eye)
        # to the nest
        s.to(T_WALK_NEST + 1.0, yaw=-1.9, look=None)
        s.to(T_WALK_NEST + 4.4, x=self.lay_xy[0] + 0.16, y=self.lay_xy[1] + 0.10)
        s.to(T_WALK_NEST + 5.6, x=self.lay_xy[0], y=self.lay_xy[1], yaw=self.lay_yaw)
        s.to(T_EGGS + 1.5, q=SIT, stride=False, eggs=True)   # settles onto the nest
        # laying: two pushes
        s.to(T_EGGS + 3.6, q=push_pose(1.0))
        s.to(T_EGGS + 4.6, q=SIT)
        s.to(T_EGGS + 6.6, q=push_pose(1.0))
        s.to(T_EGGS + 7.6, q=SIT)
        s.to(T_EGGS + 9.0, look=tuple(self.her_head + np.array([0.10, 0.06, 0.06])))
        s.to(T_EGGS + 14.0, look=None)
        # turn round and settle onto the eggs
        s.to(T_EGGS + 16.5)                                      # he steps back first
        s.to(T_EGGS + 17.5, q=STAND, stride=True)
        s.to(T_BROOD + 1.0, x=stage.NEST[0], y=stage.NEST[1])    # BACKS onto the eggs
        s.to(T_BROOD + 2.5, q=BROOD, stride=False)
        s.to(T_BROOD + 8.5)
        s.to(T_BROOD + 9.5, q=STAND, stride=True)                # her turn ends
        s.to(T_BROOD + 11.5, x=self.nest_out[0], y=self.nest_out[1])  # forwards, off them
        s.to(T_BROOD + 13.5, x=self.side_r[0], y=self.side_r[1], yaw=0.9)
        s.to(T_BROOD + 15.5, x=0.21, y=0.17, yaw=-2.30)          # round to the far side
        s.to(T_HATCH - 1.0, look=(self.egg_xy[0][0], self.egg_xy[0][1], 0.04))
        s.to(T_HATCH + 8.0)
        s.to(T_HATCH + 13.0, look=(self.egg_xy[0][0], self.egg_xy[0][1], 0.06))
        s.to(T_HATCH + 16.0, x=0.21, y=0.155, q=lean_pose(0.5),
             look=(self.egg_xy[0][0], self.egg_xy[0][1], 0.05))
        s.to(T_END, q=lean_pose(0.35))
        self.she = s

        # ---- HE -----------------------------------------------------------
        h = Track(he, his_home[0] - 0.42, Y - 0.02, 0.10, look=her_eye)
        h.to(2.0)
        h.to(8.0, x=his_home[0], y=Y, yaw=0.0)                  # walks up to her
        h.to(10.5, q=shy_pose(), look=(0.0, Y - 0.10, 0.02))    # loses his nerve
        h.to(T_PROPOSE, q=STAND, look=her_eye)
        h.to(T_PROPOSE + 3.5, q=KNEEL, stride=False)            # down on one knee
        h.to(T_PROPOSE + 5.0, q=offer_pose(), look=her_eye)
        h.to(T_KISS + 3.0)
        h.to(T_KISS + 5.0, q=STAND, stride=True)                # she said yes
        h.to(T_KISS + 6.0, x=-self.kiss_x, q=kiss_pose(), lean=self.KISS_LEAN, look=None)
        h.to(T_KISS + 10.5)
        h.to(T_KISS + 12.0, x=his_home[0], q=STAND, lean=(0, 0), look=her_eye)
        h.to(T_WALK_NEST + 1.2, yaw=-1.6, look=None)
        h.to(T_WALK_NEST + 5.8, x=self.head_kiss_wait[0], y=self.head_kiss_wait[1],
             yaw=self.head_kiss_yaw)
        h.to(T_EGGS + 2.0, look=tuple(self.her_head))
        h.to(T_EGGS + 8.5, x=self.head_kiss_xy[0], y=self.head_kiss_xy[1],
             yaw=self.head_kiss_yaw)
        h.to(T_EGGS + 10.0, q=self.head_kiss_pose, lean=self.head_kiss_lean,
             stride=False, look=None)                            # kisses her head
        h.to(T_EGGS + 14.0)
        h.to(T_EGGS + 15.5, q=STAND, lean=(0, 0), stride=True)
        h.to(T_EGGS + 16.5, x=self.head_kiss_wait[0], y=self.head_kiss_wait[1])  # steps back
        h.to(T_BROOD + 2.0, x=self.side_l[0], y=self.side_l[1], yaw=0.6,
             look=(0, 0, 0.10))
        h.to(T_BROOD + 13.0)                                     # waits till she is clear
        h.to(T_BROOD + 15.5, x=self.nest_in[0], y=self.nest_in[1], look=None)
        h.to(T_BROOD + 16.5, yaw=self.brood_yaw)                 # turns in front of it
        h.to(T_BROOD + 19.0, x=stage.NEST[0], y=stage.NEST[1])   # BACKS on, as she did
        h.to(T_BROOD + 20.5, q=BROOD, stride=False)              # his turn
        h.to(T_HATCH - 2.5)
        h.to(T_HATCH - 1.0, q=STAND, stride=True)
        h.to(T_HATCH + 1.0, x=self.nest_out[0], y=self.nest_out[1])   # forwards, off them
        h.to(T_HATCH + 2.5, x=self.side_r[0], y=self.side_r[1], yaw=1.2)
        h.to(T_HATCH + 6.0, x=-0.18, y=0.18, yaw=-0.79,
             look=(self.egg_xy[1][0], self.egg_xy[1][1], 0.04))
        h.to(T_HATCH + 8.0)
        h.to(T_HATCH + 13.0, look=(self.egg_xy[1][0], self.egg_xy[1][1], 0.06))
        h.to(T_HATCH + 16.0, x=-0.20, y=0.165, q=lean_pose(0.5),
             look=(self.egg_xy[1][0], self.egg_xy[1][1], 0.05))
        h.to(T_END, q=lean_pose(0.35))
        self.he = h

        # ---- THE DUCKLINGS -------------------------------------------------
        # Curled inside the shell, hidden, until their egg opens.
        self.kids = []
        for kid, egg_i, look in (
                ("kidb", 1, (-0.19, 0.17, 0.20)),      # blue: the left egg, his side
                ("kidp", 0, (0.20, 0.16, 0.20))):      # pink: the right egg, her side
            ex, ey = self.egg_xy[egg_i]
            t_out = self.t_pop(egg_i) + 2.0
            # dz is its resting height above the nest. It sits ON the nest
            # floor inside its shell, never underground; the opaque shell hides
            # whatever of it is below the rim. It has to come up CLEAR of the
            # rim before it moves sideways, or it walks out through its shell.
            rim = stage.EGG_SEMI[1] * (1 + stage.SPLIT)
            k = Track(self.ducks[kid], ex, ey, -1.4, q=CURL, dz=0.0, stride=False)
            k.to(t_out, q=CURL)
            k.to(t_out + 1.6, q=PEEK)                              # head comes up
            k.to(t_out + 2.6, q=PEEK)
            k.to(t_out + 4.2, q=STAND, dz=rim + 0.005, look=look)  # stands up out of it
            side = -1.0 if kid == "kidb" else 1.0           # blue left, pink right (their parents' sides)
            k.to(t_out + 5.6, x=ex + side * 0.06, y=ey - 0.05, dz=0.0)   # steps out
            k.to(t_out + 7.2, x=ex + side * 0.08, y=ey - 0.11, yaw=-1.5, stride=True)
            k.to(t_out + 9.2, x=side * 0.085, y=ey - 0.155)
            k.to(T_END, look=look)
            k.carried_until = t_out + 5.6      # the hand lifts it only until it has stepped out
            k.egg = egg_i
            self.kids.append((k, t_out))

    # -- duck against duck ---------------------------------------------------

    def slide_to_touch(self, duck, pose, lean, yaw, start_xy, away, other, reach=0.30):
        """Back `duck` away from `start_xy` along `away` until it is no longer
        inside `other`, to a tenth of a millimetre. Returns where it stops and
        the pair of parts that are touching there.

        `other` must already be posed. This is a bisection on one number, so
        it finds the FIRST clear position, which is where the two are in
        contact -- a solved touch, rather than a guessed gap.
        """
        def depth(d):
            xy = np.asarray(start_xy) + d * np.asarray(away)
            duck.place(xy[0], xy[1], yaw, pose, lean=lean)
            mujoco.mj_kinematics(self.model, self.data)
            hits = [p for p in self.penetrations(hidden_ok=True)
                    if {p[1], p[2]} == {duck.prefix, other}]
            return hits[0] if hits else None

        lo, hi = 0.0, reach
        if depth(lo) is None:
            return np.asarray(start_xy, float), None
        if depth(hi) is not None:
            raise RuntimeError("still inside at full reach")
        while hi - lo > 1e-4:
            mid = 0.5 * (lo + hi)
            if depth(mid) is not None:
                lo = mid
            else:
                hi = mid
        hit = depth(lo)
        name = lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH,
                                           self.model.geom_dataid[g])
        pair = (name(hit[3]), name(hit[4]))
        return np.asarray(start_xy) + hi * np.asarray(away), pair

    def penetrations(self, tol=0.0005, hidden_ok=False, data=None):
        """Every place one duck is inside another, right now.

        Returns (depth, name_a, name_b, geom_a, geom_b, pos, normal) sorted
        deepest first. Depth is positive metres of overlap. A duck that is
        invisible (a duckling still folded in its egg) does not count unless
        `hidden_ok`.
        """
        m, d = self.model, (self.data if data is None else data)
        mujoco.mj_collision(m, d)
        owner = {}
        for name, duck in self.ducks.items():
            if not hidden_ok and m.geom_rgba[duck.geoms[0], 3] < 0.99:
                continue
            for g in duck.geoms:
                owner[g] = name
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -tol:
                continue
            a, b = owner.get(int(c.geom1)), owner.get(int(c.geom2))
            if a is None or b is None or a == b:
                continue
            out.append((-float(c.dist), a, b, int(c.geom1), int(c.geom2),
                        np.array(c.pos), np.array(c.frame[:3])))
        out.sort(key=lambda r: -r[0])
        return out

    # -- per-frame ---------------------------------------------------------

    def update(self, t, dt):
        # Order matters: the eggs are what the parents sit ON, so they have to
        # be in place before anyone is stood up on them. The ring, the bow and
        # the hearts hang off the ducks, so they come last.
        self.place_eggs(t)
        self.she.apply(t, dt)
        self.he.apply(t, dt)
        for k, _ in self.kids:
            k.apply(t, dt)
        self.place_worn(t)
        mujoco.mj_kinematics(self.model, self.data)

    def place_worn(self, t, ducks=None):
        ducks = ducks or self.ducks
        she, he = ducks["she"], ducks["he"]

        # Her bow and his bow tie ride along, in the frame of the part they sit on.
        _, hR = she.frame(she.head)
        self.props["bow"].set(she.head_top() + np.array([0.0, 0.0, 0.005]), mat_quat(hR))
        tp, tR = he.frame(he.trunk)
        self.props["bowtie"].set(tp + tR @ np.array([0.045, 0.0, 0.030]), mat_quat(tR))

        self.place_ring(t, ducks)
        self.place_hearts(t, ducks)

    def place_ring(self, t, ducks=None):
        """In his beak until she says yes, then round her neck for good."""
        ducks = ducks or self.ducks
        he, she = ducks["he"], ducks["she"]
        tip, dirn = he.beak()
        on_beak = (tip + dirn * -0.004, mat_quat(frame_from_z(dirn)))

        neck = 0.5 * (she.data.xpos[she.head] + she.data.xpos[she.neck])
        axis = she.data.xpos[she.head] - she.data.xpos[she.trunk]
        on_neck = (neck, mat_quat(frame_from_z(axis)))

        t0, t1 = T_KISS + 3.2, T_KISS + 5.2
        if t < t0:
            pos, quat = on_beak
        elif t > t1:
            pos, quat = on_neck
        else:
            u = seg(t, t0, t1)
            pos = lerp(on_beak[0], on_neck[0], u) + np.array([0, 0, 0.05 * np.sin(np.pi * u)])
            quat = on_neck[1] if u > 0.5 else on_beak[1]
        self.props["ring"].set(pos, quat)

    def place_eggs(self, t):
        """Kinematic film: the eggs as domes the kin rig can rest on. The
        physics film overrides this and moves real egg bodies instead."""
        stage.SUPPORT.clear_eggs()
        lay_t = (T_EGGS + 3.6, T_EGGS + 6.6)
        for i in (0, 1):
            if lay_t[i] <= t < self.t_pop(i):
                stage.SUPPORT.add_egg(self.egg_pos[i])
        for k, _ in self.kids:
            set_alpha(self.model, k.duck.geoms,
                      seg(t, self.t_pop(k.egg) + 0.25, self.t_pop(k.egg) + 0.85))

    @staticmethod
    def t_pop(i):
        """When egg `i` loses its lid."""
        return T_HATCH + 6.0 + 1.2 * i

    def wobble(self, t, i):
        """How hard egg `i` is rocking: nothing, then more and more, then still."""
        t0 = T_HATCH + 2.0 + 1.2 * i
        t1 = T_HATCH + 6.0 + 1.2 * i
        if t < t0 or t > t1 + 0.4:
            return 0.0
        return smooth((t - t0) / (t1 - t0)) if t < t1 else max(0.0, 1 - (t - t1) / 0.4)

    def place_hearts(self, t, ducks=None, life=2.8, rise=0.20):
        """Hearts drift up from between whoever the moment is about."""
        ducks = ducks or self.ducks
        she, he = ducks["she"], ducks["he"]
        bursts = [(T_PROPOSE + 5.0, T_KISS + 1.0, "pair"),
                  (T_KISS + 5.5, T_KISS + 12.0, "pair"),
                  (T_EGGS + 9.5, T_EGGS + 14.5, "pair"),
                  (T_HATCH + 14.0, T_END, "family")]
        rng = np.random.default_rng(5)
        offs = rng.uniform(-1, 1, (stage.N_HEARTS, 3)) * np.array([0.10, 0.06, 0.03])
        for i, heart in enumerate(self.hearts):
            shown = False
            for t0, t1, kind in bursts:
                stagger = t0 + (i * 0.31) % max(0.9, (t1 - t0) - life)
                if not (stagger <= t <= min(stagger + life, t1 + life)):
                    continue
                u = (t - stagger) / life
                if u > 1.0:
                    continue
                if kind == "pair":
                    src = 0.5 * (she.head_top() + he.head_top())
                else:
                    src = 0.25 * sum(d.head_top() for d in ducks.values())
                src = src + np.array([0.0, 0.0, 0.045])
                pos = src + offs[i] + np.array([
                    0.05 * np.sin(3.0 * u + i), 0.0, rise * u + 0.03])
                alpha = min(1.0, 3.5 * u, 3.0 * (1 - u))
                heart.set(pos, rpy_quat(0, 0, np.radians(self.cam_az + 90.0)), alpha)
                shown = True
                break
            if not shown:
                heart.hide()

    # -- camera ------------------------------------------------------------

    CAM = [
        # t,                 lookat,                dist,  az,  elev
        (0.0, (-0.10, 0.60, 0.15), 1.05, 96, -11),
        (6.0, (-0.05, 0.60, 0.16), 0.86, 92, -9),
        (11.0, (0.00, 0.60, 0.16), 0.74, 88, -7),
        (T_PROPOSE + 4.0, (0.00, 0.60, 0.14), 0.64, 84, -5),
        (T_PROPOSE + 12.0, (0.00, 0.60, 0.16), 0.62, 92, -5),
        (T_KISS + 6.0, (0.00, 0.60, 0.20), 0.45, 90, -3),
        (T_KISS + 11.0, (0.00, 0.60, 0.20), 0.45, 98, -4),
        (T_WALK_NEST + 1.0, (0.00, 0.42, 0.16), 0.86, 94, -14),
        (T_WALK_NEST + 6.0, (0.03, -0.05, 0.14), 0.72, 82, -12),
        (T_EGGS + 5.0, (0.01, -0.05, 0.08), 0.50, 78, -10),
        (T_EGGS + 11.0, (0.04, -0.06, 0.15), 0.52, 88, -7),
        (T_BROOD + 2.0, (0.00, 0.00, 0.10), 0.58, 96, -12),
        (T_BROOD + 20.5, (0.00, 0.00, 0.10), 0.58, 76, -12),
        (T_HATCH + 2.0, (-0.013, 0.040, 0.065), 0.42, 88, -10),
        (T_HATCH + 9.0, (-0.013, 0.040, 0.060), 0.38, 94, -8),
        (T_HATCH + 15.0, (-0.01, 0.03, 0.08), 0.50, 84, -10),
        (T_END, (0.00, 0.02, 0.12), 0.76, 68, -13),
    ]
    cam_az = 90.0

    def camera(self, t):
        keys = self.CAM
        a, b = keys[0], keys[-1]
        for k0, k1 in zip(keys, keys[1:]):
            if t <= k1[0]:
                a, b = k0, k1
                break
        u = seg(t, a[0], b[0]) if t <= b[0] else 1.0
        look = lerp(a[1], b[1], u)
        dist, az, elev = (lerp(a[2], b[2], u), lerp(a[3], b[3], u), lerp(a[4], b[4], u))
        az = float(az) + 1.5 * np.sin(2 * np.pi * t / 37.0)   # a slow breath
        self.cam_az = az
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = float(dist), az, float(elev)
        return cam


# ---------------------------------------------------------------------------
# Poses that are built from a base pose
# ---------------------------------------------------------------------------


def kiss_pose(down=0.0):
    q = STAND.copy()
    q[stage.I_NECK] = 0.05
    q[stage.I_HEADP] = -0.10 + down
    q[stage.I_MOUTH] = 0.06
    return q


def offer_pose():
    q = KNEEL.copy()
    q[stage.I_NECK] = -0.05
    q[stage.I_HEADP] = -0.35
    q[stage.I_MOUTH] = 0.16
    return q


def lean_pose(amount):
    """Head down and forward -- curious, or looking at something small."""
    q = STAND.copy()
    q[stage.I_NECK] = 0.349 + 0.30 * amount
    q[stage.I_HEADP] = 0.349 + 0.55 * amount
    return q


def nod_pose(amount):
    q = STAND.copy()
    q[stage.I_HEADP] = 0.349 + amount
    q[stage.I_NECK] = 0.349 + 0.25 * amount
    return q


def shy_pose():
    q = STAND.copy()
    q[stage.I_NECK] = 0.60
    q[stage.I_HEADP] = 0.75
    q[stage.I_HEADY] = -0.35
    return q


def push_pose(amount):
    """The moment an egg arrives: chest up, tail down."""
    q = SIT.copy()
    q[stage.I_NECK] = 0.50 - 0.55 * amount
    q[stage.I_HEADP] = 0.55 - 0.85 * amount
    q[stage.I_LKNEE] += 0.10 * amount
    q[stage.I_RKNEE] -= 0.10 * amount
    return q


# ---------------------------------------------------------------------------
# Driving it
# ---------------------------------------------------------------------------


def check(film):
    """Everything that can be verified without a GL context.

    The important one is GHOSTING: after every duck has been placed, ask how
    far it is still inside the ground, the nest or an egg. Anything above zero
    is a frame where the film draws one solid thing inside another.
    """
    dt = 1.0 / FPS
    n = int(T_END * FPS)
    bad, lowest, heights = [], np.inf, {}
    ghost = {name: (0.0, 0.0) for name in film.ducks}
    hop = {name: (0.0, 0.0) for name in film.ducks}
    pairs = {}
    for i in range(n):
        t = i * dt
        film.update(t, dt)
        for name, duck in film.ducks.items():
            q = film.data.qpos[duck.qadr]
            for j, v, lo, hi in duck.out_of_range(q):
                bad.append((round(t, 2), name, j, round(float(v), 3),
                            round(float(lo), 3), round(float(hi), 3)))
            z = duck.lowest()
            lowest = min(lowest, z)
            heights.setdefault(name, []).append(film.data.qpos[duck.root + 2])
            # ask about EVERY surface, whether or not this duck was resting on it
            if film.model.geom_rgba[duck.geoms[0], 3] > 0.99:   # skip while hidden
                deep = duck.sink(on_eggs=duck.scale > 0.5)
                if deep > ghost[name][0]:
                    ghost[name] = (deep, t)
            # how far an egg or a twig ever jacks a duck above the nest lining
            over = float(film.data.qpos[duck.root + 2]) - duck.rest_z_flat
            if over > hop[name][0]:
                hop[name] = (over, t)
        for depth, a, b, ga, gb, _, _ in film.penetrations():
            key = tuple(sorted((a, b)))
            if depth > pairs.get(key, (0.0, 0, "", ""))[0]:
                ma = mujoco.mj_id2name(film.model, mujoco.mjtObj.mjOBJ_MESH, film.model.geom_dataid[ga])
                mb = mujoco.mj_id2name(film.model, mujoco.mjtObj.mjOBJ_MESH, film.model.geom_dataid[gb])
                pairs[key] = (depth, t, ma, mb)
    print(f"frames            {n} at {FPS} fps = {T_END:.0f} s")
    print(f"lowest vertex     {lowest*1000:+.2f} mm (0 = on the floor)")
    for name, zs in heights.items():
        print(f"  {name:5s} trunk z {min(zs)*1000:6.1f} .. {max(zs)*1000:6.1f} mm")
    print(f"kiss distance     trunks {2*film.kiss_x*1000:.0f} mm apart, beaks at "
          f"z = {film.kiss_z*1000:.0f} mm")
    print(f"head kiss         first contact is {film.head_kiss_touch[1]} on "
          f"{film.head_kiss_touch[0]} (solved on the meshes)")
    for name, (h, t) in hop.items():
        print(f"  {name:5s} highest it is ever jacked up {h*1000:5.1f} mm at t = {t:6.2f} s")
    if pairs:
        print("duck inside duck:")
        for (a, b), (depth, t, ma, mb) in sorted(pairs.items(), key=lambda kv: -kv[1][0]):
            print(f"  {a:5s} x {b:5s} {depth*1000:5.1f} mm at t = {t:6.2f} s   ({ma} / {mb})")
    else:
        print("duck inside duck: never")
    worst = max(ghost.items(), key=lambda kv: kv[1][0])
    for name, (deep, t) in ghost.items():
        print(f"  {name:5s} deepest inside the world {deep*1000:+6.2f} mm at t = {t:6.2f} s")
    print("ghosting          worst is %s at %+.2f mm%s" % (
        worst[0], worst[1][0] * 1000,
        "  -- NOTHING passes through anything" if worst[1][0] < 0.0005 else "  <-- FIX THIS"))
    if bad:
        print(f"OUT OF JOINT RANGE: {len(bad)} frames")
        for row in bad[:12]:
            print("   ", row)
    else:
        print("joint limits      every pose is inside the model's own ranges")
    return 1 if bad else 0


def render(film, out, width, height, t0, t1, sheet=None, stills=()):
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(film.model, height=height, width=width)
    dt = 1.0 / FPS
    grabs, writer = [], None
    if out:
        writer = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    want = sorted(stills)
    n = int(T_END * FPS)
    for i in range(n):
        t = i * dt
        film.update(t, dt)                       # every frame, so motion is continuous
        if not (t0 <= t <= t1):
            continue
        if want and not any(abs(t - w) < dt / 2 for w in want):
            continue
        renderer.update_scene(film.render_data, camera=film.camera(t))
        frame = renderer.render()
        if writer is not None:
            writer.append_data(frame)
        if sheet or want:
            grabs.append((t, frame))
        if i % (FPS * 10) == 0:
            print(f"  t = {t:6.1f} s", flush=True)
    if writer is not None:
        writer.close()
        print(f"wrote {out}")
    if sheet and grabs:
        cols = 4
        rows = int(np.ceil(len(grabs) / cols))
        pad = np.zeros((height, width, 3), np.uint8)
        tiles = [g[1] for g in grabs] + [pad] * (rows * cols - len(grabs))
        img = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        imageio.imwrite(sheet, img)
        print(f"wrote {sheet}  ({len(grabs)} stills at "
              f"{', '.join('%.1f' % g[0] for g in grabs)} s)")
    elif want:
        base = os.path.splitext(sheet or out or "still")[0]
        for t, frame in grabs:
            imageio.imwrite(f"{base}_{t:07.2f}.png", frame)
            print(f"wrote {base}_{t:07.2f}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="mp4 to write")
    ap.add_argument("--sheet", help="contact sheet PNG of sample frames")
    ap.add_argument("--stills", type=float, nargs="*", default=None,
                    help="render only these times (seconds)")
    ap.add_argument("--check", action="store_true", help="FK checks only, no GL")
    ap.add_argument("--range", type=float, nargs=2, default=(0.0, T_END),
                    metavar=("T0", "T1"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    film = Film()
    if args.check:
        raise SystemExit(check(film))

    stills = args.stills
    if args.sheet and stills is None:
        stills = list(np.linspace(1.0, T_END - 1.0, 16))
    render(film, args.out, args.width, args.height, args.range[0], args.range[1],
           sheet=args.sheet, stills=stills or ())


if __name__ == "__main__":
    main()
