#!/usr/bin/env python3
"""The duck love story, v2: the first film's story and scene, on the physics
film's rules.

Nothing is posed and nothing is teleported. The adults are driven by the
pretrained walking / standing / sit-stand / ground-pick policies, closed loop,
as in the kick and pick tasks. After initialisation the only thing the
director may write is `data.ctrl`: a guard raises if a position, velocity,
collision mask, colour or applied force changes any other way. Every prop that
moves is an actuator on a joint (`film/scene_v2.py`):

    1. he walks up to her                        walking policy
    2. he sits in front of her beside the ring and she nods   sit-stand; head command
    3. they kiss, his beak on her face           standing policy, leaned in until contact
    4. she walks in and sits; a lift raises two eggs out of a pocket behind her
                                                 walking + sit-stand; lift motor
    5. he comes to her side and rests his head on hers   standing policy, until contact
    6. she broods; she leaves; he backs in and broods     walking (reverse) + sit-stand
    7. the ducklings stir and the eggs rock; the lids open; the ducklings uncurl
                                                 duckling servos; lid motors
    8. both come round to the nest; a cradle lifts the chicks to their heads
       and each parent leans in until they touch  walking + standing policies;
                                                  cradle motor

    uv run python -m microduck_lab.tasks.love_story.v2 --dry --trace        # no GPU
    MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg \\
        src/microduck_lab/tasks/love_story/v2.py --video videos/love_story/duck_love_story_v2.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mujoco

from microduck_lab.film import scene_v2 as scene
from microduck_lab.tasks.love_story.original import Agent, Story as LegacyPoses, turn_cmd  # noqa: E402

DT = 0.02

# The kiss. Both heads hold these commands; where he stands is then solved from
# the tips each tick rather than fixed, because the policy tracks a head
# command loosely. See Story.kiss and Story.kiss_mark.
#                 neck  pitch   yaw   roll
KISS_CMD_HE = (0.00, -0.20, -0.85, 0.15)
KISS_CMD_SHE = (-0.15, -0.30, -0.85, -0.40)
KISS_TIP_MM = 30.0                 # tips this close, with jaw touching jaw, is a kiss
KISS_MIN_STAND = 0.10              # never solve for a mark inside her
KISS_FACE_TOL = 0.25               # 14 deg; the bang-bang turn cannot hold 7
KISS_HOLD = 3.0                    # seconds to hold it before letting go
KID_TUCK = 0.55                    # the curl: the one pose a passive chick holds. It was
                                   # searched for eight seconds of stability, and the ending
                                   # runs far longer than that -- the chick creeps over about
                                   # 0.01 of upright per second once the cradle is moving. So
                                   # the last beats are kept short. 0.42 was tried and is worse.
KID_WRIGGLE = 0.05                 # a chick is only balanced in its shell, not held:
                                   # +-0.12 rad of wriggle tipped one to 0.6999 against
                                   # a 0.70 limit. Small enough to see, small enough to keep.
SIT_BACKS_UP = 0.040               # half the backward carry of a sit, to clear the pocket


class Story:
    def __init__(self, verbose=True):
        self.model, self.data = scene.build_model()
        m, d = self.model, self.data
        self.verbose = verbose
        self.she = Agent(m, d, "she", 0.16, scene.STAGE_Y, np.pi)
        self.he = Agent(m, d, "he", -0.62, scene.STAGE_Y, 0.0)
        self.adults = (self.she, self.he)
        self.kids = []
        for i, prefix in enumerate(("kidb", "kidp")):
            kid = Agent(m, d, prefix, *scene.EGG_XY[i], scene.FACING, has_policy=False)
            kid.pose_target = LegacyPoses.curl_pose()
            kid.egg = i
            d.qpos[kid.qpos_idx] = kid.pose_target
            d.qpos[kid.adr + 2] = d.qpos[m.jnt_qposadr[self.id("JOINT", f"egg{i}_free")] + 2] + scene.INNER_FLOOR + 0.012
            self.kids.append(kid)
        self.actors = (*self.adults, *self.kids)
        for a in self.adults:
            a.loose_aim = True        # see Agent.drive_to: heading noise near the mark
        act = lambda n: self.id("ACTUATOR", n)
        self.lid_act = [act(f"lid_motor_{i}") for i in (0, 1)]
        self.lid_q = [m.jnt_qposadr[self.id("JOINT", f"egg_lid_{i}")] for i in (0, 1)]
        self.egg_body = [self.id("BODY", f"egg{i}_base") for i in (0, 1)]
        self.lift_act = act("lift_motor")
        self.cradle_act = act("cradle_motor")
        self.lift_q = m.jnt_qposadr[self.id("JOINT", "lift_slide")]
        self.cradle_q = m.jnt_qposadr[self.id("JOINT", "cradle_slide")]
        self.ring = self.id("BODY", "ring")
        mujoco.mj_forward(m, d)

        self.t = 0.0
        self.phase_i = 0
        self.phase_t = 0.0
        self.scratch = {}
        self.events = []
        self.failure = None
        self.min_up = np.ones(4)
        self.max_penetration = 0.0
        self.worst_contact = None
        self.kiss_contact = False
        self.beak_kiss = False
        self.kiss_gap_mm = 1e3
        self.head_kiss_contact = False
        self.duckling_kiss = [False, False]      # he -> the blue chick, she -> the pink one
        self.duckling_gap = [9.9, 9.9]           # closest beak-to-chick-head distance
        self.egg_rise = np.zeros(2)
        self.egg_z0 = d.xpos[self.egg_body, 2].copy()
        self.camera_now = np.array([-0.15, scene.STAGE_Y, 0.16, 0.85, 92.0, -8.0])
        self.immutable = {k: getattr(m, k).copy() for k in ("geom_contype", "geom_conaffinity", "geom_rgba", "body_gravcomp")}
        self.phases = [
            ("Meet", 20, self.meet),
            ("A sit, and a yes", 22, self.propose),
            ("A kiss", 30, self.kiss),
            ("Make room", 4, self.separate),
            ("To the nest", 70, self.to_nest),
            ("She settles", 12, self.sit),
            ("Two eggs", 18, self.lay),
            ("A kiss on her head", 30, self.head_kiss),
            ("Keeping watch", 7, self.brood),
            ("Mother makes room", 18, self.make_room),
            ("Father takes a turn", 48, self.father),
            ("Something stirs", 20, self.hatch),
            ("Father comes to see", 22, self.father_watches),
            ("Both come to the nest", 70, self.gather),
            ("Up you come", 20, self.raise_cradle),
            ("A kiss for each", 16, self.kiss_the_kids),
            ("Hello, little ones", 8, self.finale),
        ]
        # Where a parent stands to bow over its chick: 81 mm to the side of the
        # egg, which is where the ground-pick policy's beak lands (HANDOFF.md).
        # Beside it, not in front: facing east or west its feet straddle the
        # egg north-south and stay on solid ground clear of the pocket.
        self.kiss_spot = [np.array([scene.EGG_XY[0][0] - 0.081, scene.EGG_XY[0][1]]),
                          np.array([scene.EGG_XY[1][0] + 0.081, scene.EGG_XY[1][1]])]
        self.kiss_yaw = [0.0, np.pi]
        self.note(self.phases[0][0])

    # -- helpers -------------------------------------------------------------

    def id(self, kind, name):
        i = mujoco.mj_name2id(self.model, getattr(mujoco.mjtObj, "mjOBJ_" + kind), name)
        if i < 0:
            raise KeyError(name)
        return i

    def note(self, text):
        self.events.append({"time": round(self.t, 2), "event": text})
        if self.verbose:
            print(f"{self.t:6.2f}s {text}", flush=True)

    def held(self, key, condition, seconds):
        if not condition:
            self.scratch.pop(key, None)
            return False
        self.scratch.setdefault(key, self.t)
        return self.t - self.scratch[key] >= seconds

    def heads_touch(self, a, b):
        return any((c.geom1 in a.head_geoms and c.geom2 in b.head_geoms) or
                   (c.geom2 in a.head_geoms and c.geom1 in b.head_geoms) for c in self.data.contact)

    def beaks_touch(self, a, b, near=KISS_TIP_MM / 1000):
        """Jaw on jaw, near the tips.

        Jaw contact alone is not a kiss: the jaw geom runs the width of the
        head, so the first version reported success with the beak tips 92 mm
        apart -- they had met side to side. The tip distance has to be small
        as well.
        """
        if self.beak_gap(a, b) > near:
            return False
        return any((c.geom1 in a.beak_geoms and c.geom2 in b.beak_geoms) or
                   (c.geom2 in a.beak_geoms and c.geom1 in b.beak_geoms)
                   for c in self.data.contact[:self.data.ncon])

    def beak_gap(self, a, b):
        return float(np.linalg.norm(a.beak() - b.beak()))

    def kiss_mark(self, he, she):
        """Where he has to stand for his beak tip to land on hers.

        Heading-free on purpose. The first version assumed he would be facing
        straight back down the axis and solved from that; he arrived within
        10 mm of the mark facing 98 degrees off it, because `drive_to` leaves
        a duck pointing wherever it came from, and the beak then pointed at
        nothing. This asks only where his body must be for his tip -- wherever
        his head happens to be aimed -- to reach hers.
        """
        d = self.data
        ts = d.xmat[she.trunk].reshape(3, 3).T @ (she.beak() - d.xpos[she.trunk])
        th = d.xmat[he.trunk].reshape(3, 3).T @ (he.beak() - d.xpos[he.trunk])
        rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        mark = she.xy() + rot(she.yaw()) @ ts[:2] - rot(he.yaw()) @ th[:2]
        f = np.array([np.cos(she.yaw()), np.sin(she.yaw())])
        ahead = float((mark - she.xy()) @ f)
        if ahead < KISS_MIN_STAND:
            mark = mark + (KISS_MIN_STAND - ahead) * f
        return mark, float(ts[2] - th[2])          # and how much higher her tip is

    def route(self, actor, points, key, stop_r=0.055):
        j = self.scratch.get(key, 0)
        if j >= len(points):
            actor.stop()
            return True
        if actor.drive_to(points[j], stop_r=stop_r):
            self.scratch[key] = j + 1
        return False

    def reverse_to(self, actor, target, tol=0.04):
        delta = np.asarray(target) - actor.xy()
        if np.linalg.norm(delta) < tol:
            actor.stop()
            return True
        heading = float(np.arctan2(delta[1], delta[0]) + np.pi)
        err = float(np.arctan2(np.sin(heading - actor.yaw()), np.cos(heading - actor.yaw())))
        if abs(err) > 0.25:
            actor.vel(wz=turn_cmd(err, 0))
        else:
            actor.vel(-0.3)
        return False

    def eggs_here(self):
        return [self.data.xpos[b][:2] for b in self.egg_body]

    # -- the beats -----------------------------------------------------------

    def meet(self, e):
        if "arrived" not in self.scratch:
            if self.he.drive_to(self.she.xy(), stop_r=0.07, ahead=0.36):
                self.scratch["arrived"] = self.t
        else:
            self.he.stop()
        self.she.look_at(self.data.xpos[self.he.head_body])
        return "arrived" in self.scratch and self.t - self.scratch["arrived"] > 1 \
            and np.linalg.norm(self.he.xy() - self.she.xy()) < 0.45

    def propose(self, e):
        """He sits down in front of her beside the ring, and she nods.

        A kneel is not available: no policy on this robot puts one knee down,
        and nothing may be posed by hand. The ground-pick bow was used first
        and reads as a collapse -- the beak goes to the floor and the body
        folds. The sit-stand policy's sit is a trained, deliberate motion that
        lands him low in front of her with his head up, which is what the beat
        needs."""
        if "sat" not in self.scratch:
            self.he.stop()
            self.he.sit(True)
            self.he.head(neck=-0.25, pitch=-0.35)          # chin up, looking at her
            self.she.look_at(self.data.xpos[self.ring])
            if self.he.z() < 0.085 and self.he.up() > 0.75:
                self.scratch["sat"] = self.t
                self.note("He sits down in front of her, beside the ring")
            return False
        u = self.t - self.scratch["sat"]
        self.he.look_at(self.data.xpos[self.she.head_body])
        self.she.head(pitch=0.45 * max(0.0, np.sin(2 * np.pi * u)))           # three nods
        if u > 0.3 and "yes" not in self.scratch:
            self.scratch["yes"] = True
            self.note("She nods -- yes")
        if u > 4.0:
            self.he.sit(False)
            self.he.relax()
            self.she.relax()
            return self.he.z() > 0.105
        return False

    def kiss(self, e):
        """Beak tip to beak tip, measured.

        Straight on this kiss is impossible: her top head shell reaches 6.4 mm
        further forward than her beak tip, and on the centreline, so two ducks
        nose to nose touch shells 13 mm before the beaks meet. Every
        pitch-only lean was tried, and 29 mm apart was the best of them.

        The way through is the way people do it: stand a little to one side
        and turn your head. Yawed and offset, the head shells pass each other
        instead of meeting. A contact-checked search over the two head poses
        found 3.2 mm of tip gap that way, jaw touching jaw and no other pair
        of geoms in contact anywhere on either duck.

        So the heads take the pose first and he walks to wherever that pose
        puts the kiss. He is routed in along the axis, from a waypoint further
        out, because arriving from the side left him facing 98 degrees away
        with his beak pointing at nothing, and turning on the spot shuffles a
        duck 5 to 13 cm.
        """
        he, she = self.he, self.she
        she.stop()
        axis = self.scratch.setdefault("axis", she.yaw())
        stage = self.scratch.get("stage", 0)

        # The head only turns once he is standing still in roughly the right
        # place. Walking with the head yawed hard confuses the gait, and the
        # mark moves as the head moves, so he would be chasing it.
        lam = self.scratch.get("lean", 0.0)
        if stage >= 2:
            lam = min(1.0, lam + DT / 2.5)
        self.scratch["lean"] = lam
        if stage < 2:
            self.scratch.pop("touch_ready", None)

        mark, dz = self.kiss_mark(he, she)
        # She is standing still, so she can hold her half of the pose from the
        # start. He must not be posed while he still has walking to do: `lean`
        # switches him to the standing policy, and a standing duck ignores the
        # turn command. That cost a run -- he sat 43 degrees off her and never
        # corrected, because `face` was talking to a policy that was not
        # listening.
        she.lean(x=.02 * lam, z=.01 * lam, pitch=.08 * lam)
        she.head(*(np.asarray(KISS_CMD_SHE) * lam))
        if stage >= 2:
            # Her tip sits about 20 mm above his; his head pitch lifts his to
            # meet it, measured at roughly 104 mm of tip per radian of pitch.
            trim = float(np.clip(-dz * 6.0, -0.45, 0.45))
            he.lean(x=.02 * lam, z=-.01 * lam, pitch=-.10 * lam)
            he.head(KISS_CMD_HE[0] * lam, (KISS_CMD_HE[1] + trim) * lam,
                    KISS_CMD_HE[2] * lam, KISS_CMD_HE[3] * lam)

        gap = self.beak_gap(he, she)
        self.kiss_gap_mm = min(self.kiss_gap_mm, gap * 1000)
        if self.heads_touch(he, she):
            self.kiss_contact = True
        if self.beaks_touch(he, she):
            self.beak_kiss = True
            self.kiss_contact = True
            if "touch" not in self.scratch:
                self.note(f"Beak to beak -- measured contact, tips {gap * 1000:.1f} mm apart")
                self.scratch["touch"] = self.t

        if "touch" in self.scratch:
            # Hold the kiss, then let go and end the beat. Without the return
            # the beat sat here holding a perfectly good kiss until it timed
            # out and failed the run.
            he.stop()
            held = self.t - self.scratch["touch"]
            if held > KISS_HOLD:
                for a in self.adults:
                    a.relax()
            return held > KISS_HOLD + 1.0

        err = float(np.arctan2(np.sin(he.yaw() - axis - np.pi),
                               np.cos(he.yaw() - axis - np.pi)))
        if stage == 0:                                   # face her first
            if he.face(axis + np.pi, tol=KISS_FACE_TOL):
                self.scratch["stage"] = 1
        elif stage == 1:                                 # walk in, head still neutral
            if he.drive_to(mark, stop_r=0.020):
                self.scratch["stage"] = 2
        else:                                            # heads turn, close the last bit
            if abs(err) > 0.5:
                self.scratch["stage"] = 0
            else:
                he.drive_to(mark, stop_r=0.010)
        return False

    def separate(self, e):
        self.he.vel(-0.3)
        if e > 2.5:
            self.he.stop()
            return True
        return False

    def to_nest(self, e):
        """She goes round the nest, not over it, and BACKS onto her spot.

        The egg pocket is a hole with a rim, and it lies between the meeting
        spot and where she sits. Walking through it she caught a foot and went
        down. So she loops east to a staging point south of the nest, turns to
        face the camera there, and reverses the last 24 cm -- the same move the
        father uses. He waits well west of the pocket."""
        a = self.route(self.she, [(0.34, 0.36), (0.32, -0.28), (0.04, -0.26)], "she_way", stop_r=0.05)
        b = self.route(self.he, [(-0.36, 0.36), (-0.36, -0.14)], "he_way")
        if b:
            self.he.look_at(self.data.xpos[self.she.head_body])
        if a and "faced" not in self.scratch:
            self.scratch.setdefault("face_t0", self.t)
            if self.she.face(scene.FACING, tol=0.25) or self.t - self.scratch["face_t0"] > 6:
                self.scratch["faced"] = True
        if "faced" in self.scratch and "backed" not in self.scratch:
            # Back onto a mark short of the sit spot. Sitting down carries a
            # duck 73 mm backwards, measured, and backwards here is the egg
            # pocket: landing on the spot itself left her 31 mm from a 12 mm
            # hole, and standing up -- which pushes back again -- put her in
            # it. Half the correction, not all of it: the full 73 mm walked
            # her out from under the head kiss.
            mark = scene.SIT - np.array([0.0, SIT_BACKS_UP])
            if self.reverse_to(self.she, mark, tol=0.05) or self.scratch.setdefault("back_t0", self.t) < self.t - 14:
                self.scratch["backed"] = True
                self.note("She is at the nest, %.0f mm off her mark" % (1000 * np.linalg.norm(self.she.xy() - mark)))
        return self.held("arrived", b and "backed" in self.scratch, 0.6)

    def sit(self, e):
        self.she.stop()
        self.she.sit(True)
        self.he.look_at(self.data.xpos[self.she.head_body])
        return self.held("seated", self.she.z() < 0.085 and self.she.up() > 0.75, 1.0)

    def lay(self, e):
        """The lift raises the two eggs out of the pocket behind her."""
        d = self.data
        d.ctrl[self.lift_act] = scene.LIFT_TRAVEL * np.clip(e / 6.0, 0, 1)
        self.egg_rise = d.xpos[self.egg_body, 2] - self.egg_z0
        self.he.look_at(self.data.xpos[self.she.head_body])
        for i in (0, 1):
            if self.egg_rise[i] > 0.05 and f"egg{i}" not in self.scratch:
                self.scratch[f"egg{i}"] = True
                self.note(f"Egg {i} is up, {1000*self.egg_rise[i]:.0f} mm risen")
        # It settles a little under the commanded height: the cradle, its
        # column and two 60 g eggs hang on it.
        return d.qpos[self.lift_q] > scene.LIFT_TRAVEL - 0.020 and np.all(self.egg_rise > 0.09) and e > 7

    def head_kiss(self, e):
        he, she = self.he, self.she
        head = self.data.xpos[she.head_body][:2]
        if "at" not in self.scratch:
            if self.route(he, [(-0.36, head[1]), (-0.085, head[1])], "kiss_way"):
                self.scratch["at"] = self.t
                he.stop()
                self.note("At her side, %.0f mm from her head" % (1000 * np.linalg.norm(he.xy() - head)))
            return False
        if self.t - self.scratch["at"] < 1.0:
            return False
        touching = self.heads_touch(he, she)
        if touching:
            self.head_kiss_contact = True
            if "touch" not in self.scratch:
                self.note("His head rests on hers")
                self.scratch["touch"] = self.t
        lam = self.scratch.get("lean", 0.0)
        if not touching:
            lam = min(1.0, lam + DT / 2.5)
        self.scratch["lean"] = lam
        creep = self.scratch.setdefault("creep", {"n": 0, "until": -1.0, "last": self.t})
        if self.t < creep["until"]:
            he.vel(0.3)
            return False
        if lam >= 1.0 and not touching and "touch" not in self.scratch and creep["n"] < 3 and self.t - creep["last"] > 3.0:
            creep.update(n=creep["n"] + 1, until=self.t + 0.6, last=self.t)
            self.note("Short of her by a step -- creeping in")
            he.vel(0.3)
            return False
        if he.policy.current_policy == "walking" and "touch" not in self.scratch:
            he.stop()
        he.lean(x=.02 * lam, z=-.03 * lam, pitch=-.35 * lam, neck=.20 * lam, head_pitch=1.0 * lam)
        she.head(pitch=0.3)
        if "touch" in self.scratch and self.t - self.scratch["touch"] > 2.5:
            he.relax()
            she.relax()
            return True
        return False

    def brood(self, e):
        self.she.head(pitch=.15 * np.sin(e * 1.3), yaw=.25 * np.sin(e * .8))
        self.he.look_at([*scene.POCKET_C, 0.05])
        return e > 6

    def make_room(self, e):
        """She stands off the nest and walks clear.

        Deliberately plain. Leaning her forward through the stand, and gating
        the first step on an upright-and-still test, were both tried while
        chasing a fall here; both made it worse. The fall was the egg pocket
        behind her, and it is fixed where it starts -- see SIT_BACKS_UP.
        """
        she = self.she
        she.sit(False)
        she.relax()
        if e < 2:
            return False
        done = self.route(she, [(0.0, -0.36), (0.30, -0.32)], "exit")
        if done:
            she.face(float(np.arctan2(*((scene.POCKET_C - she.xy())[::-1]))), tol=0.3)
        return done

    def wake_walking(self, actor, e, seconds=1.2):
        """Get a duck out of the standing policy before asking it to walk.

        The head kiss leaves him standing, and a standing duck answers a
        forward command but not a pure turn. `drive_to` turns first, so he
        shuffled on the spot 240 mm from his mark for 45 s, touching nothing,
        while the stall detector backed him off every 5 s. A short forward
        walk switches the policy over, and then everything else works.
        """
        if e > seconds:
            return True
        # Clear the head and body commands the previous beat left behind. The
        # head kiss ends with his head yawed hard and his body leaning, and
        # nothing resets either; walking with them still set sent him in a
        # slow circle. He shuffled 240 mm from his mark for 45 s, touching
        # nothing at all, until the beat timed out.
        actor.relax()
        actor.vel(0.25)
        return False

    def father(self, e):
        if not self.wake_walking(self.he, e):
            return False
        target = scene.POCKET_C - self.she.xy()
        if self.she.face(float(np.arctan2(target[1], target[0])), tol=0.3):
            self.she.look_at([*scene.POCKET_C, 0.05])
        if "at" not in self.scratch:
            # First waypoint straight south of wherever he is, not a fixed
            # point west of him. He now finishes the head kiss close beside
            # her, and a fixed waypoint 60 degrees behind him put him in a
            # turn-shuffle he never got out of: 25 s of turning on the spot
            # 290 mm from a mark. Going south first is a few degrees of turn.
            if "entry_way" not in self.scratch:
                self.scratch["entry_way"] = [
                    (float(np.clip(self.he.xy()[0], -0.36, -0.12)), -0.26), (0.0, -0.26)]
            if self.route(self.he, self.scratch["entry_way"], "entry"):
                self.scratch["at"] = True
            return False
        if "faced" not in self.scratch:
            if self.he.face(scene.FACING, tol=0.25):
                self.scratch["faced"] = True
            return False
        if "backed" not in self.scratch:
            # Same correction as hers: the sit carries a duck backwards into
            # the pocket, so the mark he reverses onto sits short of the spot.
            mark = scene.SIT - np.array([0.0, SIT_BACKS_UP])
            if self.reverse_to(self.he, mark, tol=0.05):
                self.scratch["backed"] = True
                self.note("He backs onto the nest, %.0f mm off the mark" % (1000 * np.linalg.norm(self.he.xy() - mark)))
            return False
        self.he.sit(True)
        return self.held("seated", self.he.z() < 0.085 and self.he.up() > 0.75, 3.0)

    def hatch(self, e):
        """The ducklings stir first -- their servos rock the eggs, no torque is
        applied from outside -- then the lid motors open the shells."""
        for i, kid in enumerate(self.kids):
            if e < 4.0:
                kid.pose_target[6] = 0.55 + 0.35 * np.sin(2 * np.pi * 1.6 * e + i)
                kid.pose_target[7] = 0.5 * np.sin(2 * np.pi * 1.1 * e + 2 * i)
            self.data.ctrl[self.lid_act[i]] = -2.45 * np.clip((e - 4.0 - i * 2) / 3.0, 0, 1)
            if self.data.qpos[self.lid_q[i]] < -1.8:
                kid.pose_target[5] = 0.15
                kid.pose_target[6] = -0.10
                kid.pose_target[7] = 0.0
                if f"out{i}" not in self.scratch:
                    self.scratch[f"out{i}"] = True
                    self.note(f"Egg {i} is open")
        self.she.look_at([*scene.POCKET_C, 0.05])
        self.he.head(pitch=0.2)
        return e > 11 and all(self.data.qpos[q] < -2.2 for q in self.lid_q)

    def father_watches(self, e):
        self.he.sit(False)
        self.he.relax()
        if e < 2:
            return False
        if not self.route(self.he, [(0.0, -0.30), (-0.25, -0.26)], "exit"):
            return False
        angle = float(np.arctan2(*((scene.POCKET_C - self.he.xy())[::-1])))
        if self.he.face(angle, tol=0.3):
            self.he.look_at([*scene.POCKET_C, 0.06])
            return True
        return False

    def kid_geoms(self, kid):
        m = self.model
        if not hasattr(kid, "_geoms"):
            kid._geoms = {g for g in range(m.ngeom)
                          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "").startswith(kid.prefix + "_")}
        return kid._geoms

    BOW_REACH = 0.081        # where the ground-pick policy's beak lands (HANDOFF.md)

    def creep(self, parent, target, stop_at=0.115):
        """Walk a parent in until it is `stop_at` from a point, then stop.

        A duck told to stop at walking speed coasts 30-50 mm, so the command
        goes out well before the mark and the coast finishes the job -- the
        pick task's "stop inside the window, a little nearer than the mark".
        Timed bursts were tried instead and were worse in both directions:
        half-second ones barely move it (the gait takes that long to start),
        one-second ones carried him 130 mm, across the nest and over an egg."""
        dist = float(np.linalg.norm(parent.xy() - target))
        if dist <= stop_at:
            parent.stop()
            return True
        parent.vel(0.3)
        return False

    def standoff(self, parent, egg, r):
        """A point `r` from the egg, on the parent's own side of it."""
        v = parent.xy() - np.asarray(egg)
        n = float(np.linalg.norm(v))
        return np.asarray(egg) + (v / n if n > 1e-6 else np.array([-1.0, 0.0])) * r

    def gather(self, e):
        """Both parents come round to the nest, one to each side of the eggs.

        The standoff point is recomputed every tick, so walking drift steers
        itself out -- driving to a fixed point left him 50 mm off the line and
        his head went past the chick. This happens while the nest lift is up
        and the cradle is down, so the whole nest is flat, walkable floor."""
        a = self.route(self.he, [(-0.30, -0.05), (-0.30, 0.10)], "he_way", stop_r=0.05)
        b = self.route(self.she, [(0.34, -0.10), (0.34, 0.10)], "she_way", stop_r=0.05)
        # Walk straight at the chick and stop 150 mm out: the walk overruns
        # 30-50 mm, which lands the parent in the 85-110 mm band the lean
        # reaches from. Aiming at a standoff POINT instead left them at 165 mm
        # (the point recedes as they approach); stopping at 115 mm let the
        # overrun carry them into the egg, and they went down.
        # Aimed at the egg, not at the chick's head: the head sits 12 mm inside
        # the shell and walking at it takes them into the nest.
        for parent, egg in ((self.he, scene.EGG_XY[0]), (self.she, scene.EGG_XY[1])):
            parent.look_at([*egg, 0.05])
        # Settle the chicks into the tucked pose here rather than leaving them
        # in the one the hatch ends on. A chick is a passive body balanced on
        # the curved floor of its shell: touching nothing at all, the pink one
        # drifted from upright 0.94 to 0.72 over two seconds of this beat and
        # failed the run. Tucked, its weight sits lower and it stays put.
        for kid in self.kids:
            kid.pose_target[5] += (0.50 - kid.pose_target[5]) * DT / 0.8
            kid.pose_target[6] += (KID_TUCK - kid.pose_target[6]) * DT / 0.8
        if a:
            a = self.close_in(self.he, scene.EGG_XY[0])
        if b:
            b = self.close_in(self.she, scene.EGG_XY[1])
        if a and b and "there" not in self.scratch:
            self.scratch["there"] = self.t
            self.note("Both at the nest: he %.0f mm from the blue chick, she %.0f mm from the pink one"
                      % (1000 * np.linalg.norm(self.he.xy() - scene.EGG_XY[0]),
                         1000 * np.linalg.norm(self.she.xy() - scene.EGG_XY[1])))
        return self.held("settled", a and b, 1.5)

    def close_in(self, parent, egg, stop_at=0.145, too_close=0.120):
        """Walk AT the chick and stop 150 mm out.

        `drive_to` does the steering; a bare forward command does not, and
        after the route both of them were facing north and walked off the set.
        `drive_to` decelerates well, so it stops near the mark. 145 mm with a
        120 mm backstop: closer than that a chest touches the egg and tips the
        chick inside it (measured at 89 mm). It was 130/105, which held until
        the pair started arriving from slightly different places and a chick
        went over as the cradle came up. The 15 mm costs nothing -- the beaks
        do not reach the chicks at either distance."""
        dist = float(np.linalg.norm(parent.xy() - np.asarray(egg)))
        if dist < too_close:
            parent.vel(-0.3)             # too close to lean safely: give it back
            return False
        return parent.drive_to(np.asarray(egg), stop_r=stop_at)

    def raise_cradle(self, e):
        """The cradle inside the nest lift carries the open shells up to the
        parents' heads. Nobody walks while it moves."""
        # Nothing moves for the first second and a half: the chicks change
        # pose here, and the cradle must not start until they have settled.
        # Smoothstep, not a straight ramp. A linear command steps the
        # acceleration on at the start and off at the end, and each step
        # rocks a chick that is only balanced on the curved floor of its
        # shell: it came off the gather at 0.99 upright and was down to 0.83
        # by the top of the rise, still touching nothing but its own egg.
        # u*u*(3-2u) starts and stops at zero acceleration.
        u = float(np.clip((e - 1.5) / 8.0, 0.0, 1.0))
        self.data.ctrl[self.cradle_act] = scene.CRADLE_RISE * u * u * (3.0 - 2.0 * u)
        # Hold the lids folded back but OFF their stop (-2.45 of -2.79). Driven
        # hard against the stop, a lid has nowhere to give: as the cradle
        # accelerated, egg 0's went 0.05 rad past its limit, levered its own
        # shell and tipped the chick inside (0.99 -> 0.69).
        for act in self.lid_act:
            self.data.ctrl[act] = -2.45
        # Heads neutral and high while the eggs come up past them: aimed down
        # at the chick, a beak clipped a rising shell and tipped it.
        for p in self.adults:
            p.stop()
            p.relax()
        # And the chicks tuck their heads back down for the ride: head up, a
        # chick presses the folded lid as the cradle accelerates, drives it
        # past its stop and levers its own shell over.
        # Eased in, not stepped. The hatch leaves them at (0.15, -0.10) and
        # this pose is (0.50, 0.55); setting it in one tick is a 0.65 rad jump
        # on a joint of a body that is only balanced, not controlled, and it
        # tipped a chick over on the spot. A half-second filter does not.
        for kid in self.kids:
            kid.pose_target[5] += (0.50 - kid.pose_target[5]) * DT / 0.5
            kid.pose_target[6] += (KID_TUCK - kid.pose_target[6]) * DT / 0.5
            kid.mouth_cmd = 0.0
        up = float(self.data.qpos[self.cradle_q])
        # It settles a little under the command with two 60 g eggs on it.
        if up > scene.CRADLE_RISE - 0.020 and "up" not in self.scratch:
            self.scratch["up"] = self.t
            self.note("The cradle is up, %.0f mm; the chicks are at their parents' heads" % (1000 * up))
        return "up" in self.scratch and e > 9

    LEAN_CAP = 0.45          # the lid folds back now, but at full stretch a head
                             # still presses the shell over and tips the chick.
                             # 0.75 worked while the parents stood 130 mm out;
                             # at the 145 mm they stand at now the same cap
                             # leans them further and put the blue chick over.
                             # It costs nothing measurable: the beaks are 75
                             # to 95 mm short of the chicks either way.

    def kiss_the_kids(self, e):
        """Both parents lower their heads over their chicks.

        Standing on the nest floor a duck's beak bottoms out at z = 0.15 and a
        chick in its shell sits at 0.04, so no lean reaches one, and a chick
        cannot lift its head inside its shell (measured: the shell holds it at
        0.036 whatever the neck servo is told). The cradle brings the chick up
        to 0.13 instead and the parent leans down to it.

        The lean stops at 55 per cent of full stretch, which brings the head
        over the chick without pushing anything. Two things were tried and
        measured on the way here:

        * bowing to them -- the ground-pick beak lands at the right height,
          but the bow shuffles the duck 20-100 mm in a direction that varies,
          so contact was a coin toss;
        * leaning to full stretch -- what the head meets first is the OPEN
          LID, which stands upright over the egg. Pressed, it drives past its
          stop, levers the shell and tips the chick inside (0.98 -> 0.72
          upright in a second and a half).

        So this is a head lowered over the chick, not a beak on it, and
        `duckling_gap_mm` in the report says how close each beak got. The lid
        would have to fold away for the beak to reach, and it has nowhere to
        fold to."""
        for act in self.lid_act:
            self.data.ctrl[act] = -2.45
        for i, (parent, kid) in enumerate(((self.he, self.kids[0]), (self.she, self.kids[1]))):
            kid.pose_target[6] = KID_TUCK + KID_WRIGGLE * np.sin(2 * np.pi * 1.2 * e + i)
            parent.stop()
            gap = float(np.linalg.norm(parent.beak() - self.data.xpos[kid.head_body]))
            self.duckling_gap[i] = min(self.duckling_gap[i], gap)
            touching = any((c.geom1 in parent.head_geoms and c.geom2 in self.kid_geoms(kid)) or
                           (c.geom2 in parent.head_geoms and c.geom1 in self.kid_geoms(kid))
                           for c in self.data.contact)
            if touching and not self.duckling_kiss[i]:
                self.duckling_kiss[i] = True
                self.note("%s kisses the %s chick -- measured contact"
                          % ("He" if i == 0 else "She", "blue" if i == 0 else "pink"))
            lam = self.scratch.get(f"lean{i}", 0.0)
            if not touching:
                lam = min(self.LEAN_CAP, lam + DT / 2.0)
            self.scratch[f"lean{i}"] = lam
            parent.lean(x=.02 * lam, z=-.025 * lam, pitch=-.38 * lam, neck=.20 * lam, head_pitch=1.05 * lam)
        return e > 6

    def finale(self, e):
        for i, (parent, kid) in enumerate(((self.he, self.kids[0]), (self.she, self.kids[1]))):
            # Heads DOWN, small wriggle: a chick that raises its head tips
            # itself over inside the shell (0.99 -> 0.70), with nothing near
            # it. The curled pose is the only one a passive chick holds.
            kid.pose_target[6] = KID_TUCK + KID_WRIGGLE * np.sin(e * 2 + i)
            kid.pose_target[7] = KID_WRIGGLE * np.sin(e * 1.5 + i)
            kid.mouth_cmd = 0.20 * max(0.0, np.sin(e * 3 + i))
            parent.stop()
            # The same capped lean as the kiss: at full stretch a head presses
            # the shell over and tips the chick inside it.
            lam = self.LEAN_CAP
            parent.lean(x=.02 * lam, z=-.025 * lam, pitch=-.38 * lam, neck=.20 * lam,
                        head_pitch=1.05 * lam)
        # Short on purpose. A chick holds its pose for about eight seconds
        # before it starts to lean, and the whole ending already spends
        # more than that getting the cradle up.
        return e > 6

    # -- the loop, with the physics film's guards --------------------------------

    def control(self):
        """Only actuator controls may change simulator state during this call."""
        name, timeout, fn = self.phases[self.phase_i]
        done = fn(self.t - self.phase_t)
        if done:
            self.note("Completed: " + name)
            self.phase_i += 1
            self.phase_t = self.t
            self.scratch = {}
            for a in self.adults:
                a.reset_progress()
            if self.phase_i < len(self.phases):
                self.note(self.phases[self.phase_i][0])
        elif self.t - self.phase_t > timeout:
            raise RuntimeError("Timed out: " + name)
        for actor in self.actors:
            actor.clock = self.t
            actor.tick()

    def step(self):
        if self.phase_i >= len(self.phases):
            return False
        d, m = self.data, self.model
        q, v = d.qpos.copy(), d.qvel.copy()
        self.control()
        if not np.array_equal(q, d.qpos) or not np.array_equal(v, d.qvel):
            raise RuntimeError("Controller modified position or velocity")
        for key, value in self.immutable.items():
            if not np.array_equal(getattr(m, key), value):
                raise RuntimeError("Controller modified " + key)
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            raise RuntimeError("Unmodelled external force")
        for _ in range(4):
            mujoco.mj_step(m, d)
            for c in d.contact:
                if -c.dist > self.max_penetration:
                    self.max_penetration = -float(c.dist)
                    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                             or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
                    self.worst_contact = dict(time=round(self.t, 3), geoms=names, penetration_mm=-1000 * float(c.dist))
        mujoco.mj_forward(m, d)
        self.t += DT
        self.min_up = np.minimum(self.min_up, [a.up() for a in self.actors])
        if not np.isfinite(d.qpos).all():
            raise RuntimeError("Nonfinite simulation")
        if any(a.up() < 0.5 for a in self.adults):
            raise RuntimeError("An adult fell")
        if any(a.up() < 0.7 for a in self.kids):
            raise RuntimeError("A duckling tipped inside its egg")
        return True

    # -- camera --------------------------------------------------------------

    def camera(self):
        """Closer and lower than the first pass: the ducks are 25 cm tall and
        the shots were reading as two dots on a lawn. The eye height is the
        head, not the ground, and the pair shots frame whoever is acting."""
        p = min(self.phase_i, len(self.phases) - 1)
        she, he = self.she, self.he
        mid = 0.5 * (she.xy() + he.xy())
        nest = np.array([scene.SIT[0], (scene.SIT[1] + scene.POCKET_C[1]) / 2])
        if p == 0:                                            # meet
            target = np.array([*mid, 0.16, 0.78, 92.0, -8.0])
        elif p == 1:                                          # the bow and the nod
            target = np.array([*(0.5 * (mid + self.data.xpos[self.ring][:2])), 0.15, 0.56, 84.0, -7.0])
        elif p == 2:                                          # the kiss
            target = np.array([*mid, 0.19, 0.42, 90.0, -4.0])
        elif p == 3:                                          # make room
            target = np.array([*mid, 0.16, 0.60, 96.0, -8.0])
        elif p == 4:                                          # the walk round to the nest
            target = np.array([*(0.35 * she.xy() + 0.65 * nest), 0.14, 0.85, 92.0, -14.0])
        elif p in (5, 6):                                     # she settles; the eggs come up
            target = np.array([*(she.xy() + np.array([0.0, 0.045])), 0.11, 0.48, 92.0, -11.0])
        elif p == 7:
            # Three-quarter from the south-west: straight on from the south she
            # faces the lens and her head fills the frame.
            target = np.array([*(0.5 * (she.xy() + he.xy()) + np.array([0.0, 0.02])), 0.14, 0.52, 132.0, -10.0])
        elif p == 8:                                          # keeping watch
            target = np.array([*nest, 0.12, 0.55, 92.0, -12.0])
        elif p == 9:
            # Her exit runs straight at a camera due south of the nest and she
            # fills the lens. From the south-east she crosses the frame instead.
            target = np.array([*nest, 0.13, 0.85, 58.0, -14.0])
        elif p == 10:                                         # his turn
            target = np.array([*nest, 0.12, 0.58, 88.0, -12.0])
        elif p == 11:
            # The eggs open while he is still sitting between them and the
            # camera, so this looks over his shoulder from above instead.
            target = np.array([*scene.POCKET_C, 0.06, 0.42, 58.0, -38.0])
        elif p == 12:                                         # he comes round to see
            target = np.array([*nest, 0.11, 0.55, 84.0, -12.0])
        elif p == 13:                                         # both come to the nest
            target = np.array([*scene.POCKET_C, 0.10, 0.72, 88.0, -14.0])
        elif p == 14:                                         # the cradle lifts them up
            target = np.array([*scene.POCKET_C, 0.13, 0.60, 84.0, -8.0])
        elif p == 15:                                         # heads down over the chicks
            target = np.array([*scene.POCKET_C, 0.14, 0.46, 80.0, -6.0])
        else:
            # High enough to see the ducklings over the parents' backs.
            target = np.array([*(0.5 * (nest + scene.POCKET_C)), 0.08, 0.62, 70.0, -24.0])
        self.camera_now += (target - self.camera_now) * (1 - np.exp(-DT / 1.2))
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = self.camera_now[:3]
        c.distance, c.azimuth, c.elevation = self.camera_now[3:]
        return c

    def report(self):
        return dict(success=self.failure is None and self.phase_i == len(self.phases), failure=self.failure,
                    seconds=round(self.t, 2), completed_phases=self.phase_i, total_phases=len(self.phases),
                    kiss_contact=self.kiss_contact, beak_kiss=self.beak_kiss,
                    kiss_gap_mm=round(self.kiss_gap_mm, 1),
                    head_kiss_contact=self.head_kiss_contact,
                    duckling_kiss=self.duckling_kiss,
                    duckling_gap_mm=[round(1000 * g, 1) for g in self.duckling_gap],
                    egg_rise_mm=(1000 * self.egg_rise).tolist(),
                    lift_travel_mm=1000 * float(self.data.qpos[self.lift_q]),
                    lid_angles_rad=self.data.qpos[self.lid_q].tolist(), minimum_upright=self.min_up.tolist(),
                    maximum_contact_penetration_mm=1000 * self.max_penetration, worst_contact=self.worst_contact,
                    mocap_bodies=self.model.nmocap, equality_constraints=self.model.neq,
                    actuator_only_guard=True, events=self.events)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--report", default="videos/love_story/duck_love_story_v2.json")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    story = Story()
    writer = renderer = None
    if args.video:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(story.model, height=args.height, width=args.width)
        writer = imageio.get_writer(args.video, fps=25, quality=8, macro_block_size=8)
    tick = 0
    try:
        while story.t < args.seconds and story.step():
            cam = story.camera()
            if writer is not None and tick % 2 == 0:
                renderer.update_scene(story.data, camera=cam)
                writer.append_data(renderer.render())
            if args.trace and tick % 50 == 0:
                print("TRACE", round(story.t, 1), story.phase_i,
                      [(a.prefix, np.round(a.xy(), 3).tolist(), round(a.yaw(), 2), round(a.z(), 3)) for a in story.adults], flush=True)
            tick += 1
        if story.phase_i < len(story.phases):
            raise RuntimeError("Duration exhausted before story completion")
    except RuntimeError as exc:
        story.failure = str(exc)
        story.note("FAILED: " + str(exc))
    finally:
        if writer is not None:
            writer.close()
            renderer.close()
    report = story.report()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
