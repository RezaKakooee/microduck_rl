#!/usr/bin/env python3
"""The duck film, simulated: the same story, run through `mj_step`.

`film.py` poses a rig and takes a picture of it. This drives the same rig as
TARGETS for a physics simulation, and takes the picture of the simulation:

* every duck is a rigid articulated body under gravity, on the model's own
  contact geometry (head shells, jaw, trunk, hips, shins, soles);
* its joints are position servos tracking the pose the rig asks for, with a
  torque limit -- the robot's actuators, made stiffer (see `stage.stiffen`);
* a soft HAND steers each trunk in x and y and holds it at the heading and
  lean the rig asks for. It never lifts: height comes from gravity and from
  whatever the duck is standing, kneeling or sitting on. The ducklings are the
  one exception -- they are carried out of the egg, and the hand carries them;
* the eggs are free bodies. They are dropped, they roll into the cradle, a
  brooding parent rests on them by contact, they rock because a torque rocks
  them, and the lid comes off because its weld is released and it is given a
  shove;
* nothing can be inside anything else, because contacts push it out. Where the
  kinematic film had to solve each touch, here the kiss is two beaks pressed
  together by two hands pulling, and the head kiss is his jaw resting on her
  head because that is where gravity and her head shell leave it.

The rig is still an animator's rig: no policy is walking these ducks. The
hand is the puppeteer. What the physics adds is that the puppet has weight, and
the world pushes back.

    src/microduck_lab/film/physics.py --check             # simulate it all, no GL
    MUJOCO_GL=egl src/microduck_lab/film/physics.py --out videos/love_story/duckfilm.mp4
"""

import argparse

import numpy as np

import mujoco

from microduck_lab.film import stage
from microduck_lab.film.film import FPS, T_EGGS, T_END, Film, render, set_alpha
from microduck_lab.film.stage import Duck

G = 9.81


class PhysicsFilm(Film):
    """`Film` computes the targets; this makes a world follow them."""

    HAND_KP = 450.0     # N/m, steering an adult in x and y; foot friction pulls back ~4.5 N
    HAND_KR = 8.0       # Nm/rad, holding an adult's heading and lean
    LIFT_KP = 400.0     # N/m per kg, carrying a duckling in z

    def __init__(self):
        super().__init__()
        m = self.model
        stage.stiffen(m)
        self.kin = self.data                 # the rig: targets only
        self.sim = mujoco.MjData(m)          # the world
        self.render_data = self.sim
        self.substeps = int(round((1.0 / FPS) / m.opt.timestep))

        # Everything that is drawn reads from the simulation.
        for prop in list(self.props.values()) + self.hearts:
            prop.data = self.sim
        self.view = {n: Duck(m, self.sim, n) for n in self.ducks}

        self.trunk_dof = {n: int(m.body_dofadr[d.trunk]) for n, d in self.ducks.items()}
        self.mass = {n: float(m.body_subtreemass[d.trunk]) for n, d in self.ducks.items()}

        self.eggs = []
        for i in (0, 1):
            e = {}
            for half in ("bot", "top"):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"egg{i}_{half}_free")
                e[half] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"egg{i}_{half}")
                e[half + "_q"] = int(m.jnt_qposadr[jid])
                e[half + "_v"] = int(m.jnt_dofadr[jid])
            e["eq"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"egg{i}_weld")
            e["laid"] = e["popped"] = False
            self.eggs.append(e)
        self.lay_t = (T_EGGS + 3.6, T_EGGS + 6.6)

        # A duckling is nobody's business until it hatches.
        self.kid_bits = {}
        for n in ("kidb", "kidp"):
            for g in self.ducks[n].geoms:
                if m.geom_contype[g]:
                    self.kid_bits[g] = (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
                    m.geom_contype[g] = m.geom_conaffinity[g] = 0

        # Start the world where the rig starts.
        self.pose_rig(0.0, 1.0 / FPS)
        self.sim.qpos[:] = self.kin.qpos
        self.sim.qvel[:] = 0.0
        for e in self.eggs:
            for half in ("bot", "top"):
                self.sim.qpos[e[half + "_q"]:e[half + "_q"] + 7] = m.body_pos[e[half]].tolist() + [1, 0, 0, 0]
        mujoco.mj_forward(m, self.sim)
        self.worst_gap = {}

    # -- targets -----------------------------------------------------------

    def pose_rig(self, t, dt):
        stage.SUPPORT.clear_eggs()
        self.she.apply(t, dt)
        self.he.apply(t, dt)
        for k, _ in self.kids:
            k.apply(t, dt)
        mujoco.mj_kinematics(self.model, self.kin)

    def targets(self):
        out = {}
        for n, d in self.ducks.items():
            root = self.kin.qpos[d.root:d.root + 7].copy()
            out[n] = (root[:3], root[3:7], self.kin.qpos[d.qadr].copy())
        return out

    # -- one frame -----------------------------------------------------------

    def update(self, t, dt):
        self.pose_rig(t, dt)
        tg = self.targets()
        self.eggs_step(t)
        for k, _ in self.kids:
            set_alpha(self.model, k.duck.geoms,
                      float(np.clip((t - self.t_pop(k.egg) - 0.25) / 0.6, 0.0, 1.0)))
        for _ in range(self.substeps):
            self.hands(tg, t)
            for n, d in self.ducks.items():
                self.sim.ctrl[self.act_slice(d)] = tg[n][2]
            mujoco.mj_step(self.model, self.sim)
        self.place_worn(t, self.view)
        mujoco.mj_kinematics(self.model, self.sim)

    def act_slice(self, duck):
        """This duck's 15 actuators, in the rig's joint order.

        Actuators are named after their joints, so the order is looked up
        rather than assumed."""
        if not hasattr(duck, "_acts"):
            duck._acts = np.array([
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{duck.prefix}_{j}")
                for j in stage.JOINTS])
        return duck._acts

    def hands(self, tg, t=0.0):
        """The puppeteer: a force in x, y and a torque, on each trunk."""
        m, d = self.model, self.sim
        carried = {k.duck.prefix: k.carried_until for k, _ in self.kids}
        for n, duck in self.ducks.items():
            pos_t, quat_t, _ = tg[n]
            b, dof = duck.trunk, self.trunk_dof[n]
            mass, k = self.mass[n], duck.scale
            pos, quat = d.xpos[b], d.xquat[b]
            R = d.xmat[b].reshape(3, 3)
            v = d.qvel[dof:dof + 3]                  # world linear
            w = R @ d.qvel[dof + 3:dof + 6]          # body angular -> world

            kp = self.HAND_KP * mass / 0.757
            kd = 2.0 * np.sqrt(kp * mass)
            f = np.zeros(3)
            f[:2] = kp * (pos_t[:2] - pos[:2]) - kd * v[:2]
            if k < 0.5 and t < carried.get(n, 0.0):  # a duckling is carried out of its egg
                kpz = self.LIFT_KP * mass
                f[2] = kpz * (pos_t[2] - pos[2]) - 2.0 * np.sqrt(kpz * mass) * v[2] + mass * G

            err = np.zeros(3)
            mujoco.mju_subQuat(err, quat_t, quat)    # in the trunk's frame
            err = R @ err
            inertia = mass * (0.06 * k) ** 2
            kr = self.HAND_KR * (mass / 0.757) * k ** 2
            if k < 0.5:
                kr *= 4.0        # a duckling is all leg; hold it upright harder
            tau = kr * err - 2.0 * np.sqrt(kr * inertia) * w

            d.xfrc_applied[b, :3] = f
            d.xfrc_applied[b, 3:] = tau

    # -- eggs ----------------------------------------------------------------

    def eggs_step(self, t):
        m, d = self.model, self.sim
        for i, e in enumerate(self.eggs):
            bot = e["bot"]
            d.xfrc_applied[bot] = 0.0
            if t >= self.lay_t[i] and not e["laid"]:
                # Laid: it appears under her tail, a finger above its cradle,
                # and drops in.
                xy = self.egg_xy[i]
                for half in ("bot", "top"):
                    q = e[half + "_q"]
                    d.qpos[q:q + 3] = (xy[0], xy[1], stage.NEST_LINING_Z + stage.EGG_SEMI[1] + 0.010)
                    d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
                    d.qvel[e[half + "_v"]:e[half + "_v"] + 6] = 0.0
                e["laid"] = True
            if e["laid"] and not e["popped"]:
                wob = self.wobble(t, i)
                if wob > 0:
                    # An egg this size has I ~ 1.6e-6 kg m^2 about its middle;
                    # tens of micro-newton-metres rock it against its straw.
                    d.xfrc_applied[bot, 3] = wob * 4e-5 * np.sin(2 * np.pi * 2.1 * t + 2.1 * i)
                    d.xfrc_applied[bot, 4] = wob * 3e-5 * np.sin(2 * np.pi * 1.4 * t + i)
            if t >= self.t_pop(i) and not e["popped"]:
                # Hatched: the weld lets go, the lid is shoved, and the
                # duckling inside starts touching the world.
                d.eq_active[e["eq"]] = 0
                v = e["top_v"]
                side = -1.0 if i == 0 else 1.0
                d.qvel[v:v + 3] = (side * 0.45, -0.30, 1.05)
                d.qvel[v + 3:v + 6] = (4.0 * side, 6.0, 2.0)
                for g, (ct, ca) in self.kid_bits.items():
                    if self.model.geom_bodyid[g] in self.kid_bodies(i):
                        m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
                e["popped"] = True

    def kid_bodies(self, i):
        """Bodies of the duckling that hatches from egg `i`."""
        duck = next(k.duck for k, _ in self.kids if k.egg == i)
        return {int(self.model.geom_bodyid[g]) for g in duck.geoms}


# ---------------------------------------------------------------------------


def check(film):
    """Simulate the whole film and say how the world behaved."""
    dt = 1.0 / FPS
    n = int(T_END * FPS)
    track = {name: 0.0 for name in film.ducks}
    tilt = {name: 0.0 for name in film.ducks}
    deepest = (0.0, 0.0, "", "")
    fell = []
    for i in range(n):
        t = i * dt
        film.update(t, dt)
        tg = film.targets()
        for name, duck in film.ducks.items():
            pos = film.sim.xpos[duck.trunk]
            track[name] = max(track[name], float(np.linalg.norm(tg[name][0][:2] - pos[:2])))
            up = film.sim.xmat[duck.trunk].reshape(3, 3)[2, 2]
            ang = float(np.degrees(np.arccos(np.clip(up, -1, 1))))
            tilt[name] = max(tilt[name], ang)
            if ang > 60 and not fell:
                fell.append((name, round(t, 2), round(ang)))
        for j in range(film.sim.ncon):
            c = film.sim.contact[j]
            if -c.dist > deepest[0]:
                deepest = (-float(c.dist), t,
                           mujoco.mj_id2name(film.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                           or mujoco.mj_id2name(film.model, mujoco.mjtObj.mjOBJ_MESH,
                                                film.model.geom_dataid[c.geom1]),
                           mujoco.mj_id2name(film.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                           or mujoco.mj_id2name(film.model, mujoco.mjtObj.mjOBJ_MESH,
                                                film.model.geom_dataid[c.geom2]))
        if i % (FPS * 20) == 0:
            print(f"  t = {t:6.1f} s", flush=True)
    print(f"frames            {n} at {FPS} fps = {T_END:.0f} s, "
          f"{film.substeps} physics steps of {film.model.opt.timestep*1000:.0f} ms each")
    for name in film.ducks:
        print(f"  {name:5s} hand tracking error up to {track[name]*1000:5.1f} mm, "
              f"trunk tilted up to {tilt[name]:4.1f} deg")
    print(f"deepest contact   {deepest[0]*1000:.2f} mm at t = {deepest[1]:.2f} s "
          f"({deepest[2]} / {deepest[3]})  -- contact softness, not ghosting")
    for i, e in enumerate(film.eggs):
        p = film.sim.xpos[e["bot"]]
        print(f"  egg {i} bottom ended at {np.round(p, 3)} (cradle at {np.round(film.egg_xy[i], 3)})")
    print("fell over         " + (f"{fell[0]}" if fell else "nobody"))
    return 1 if fell else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--sheet")
    ap.add_argument("--stills", type=float, nargs="*", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--range", type=float, nargs=2, default=(0.0, T_END), metavar=("T0", "T1"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    film = PhysicsFilm()
    if args.check:
        raise SystemExit(check(film))
    stills = args.stills
    if args.sheet and stills is None:
        stills = list(np.linspace(1.0, T_END - 1.0, 16))
    render(film, args.out, args.width, args.height, args.range[0], args.range[1],
           sheet=args.sheet, stills=stills or ())


if __name__ == "__main__":
    main()
