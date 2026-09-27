#!/usr/bin/env python3
"""Human bridge: a big brother lies across a gap so his little sister can cross.

After the famous clip of a boy lying across a gap outside an apartment so
his little sister can step over him. Two real microducks, BAM XL330 motors,
MuJoCo physics. After setup only motor targets change the world; `World.step`
refuses anything else.

    1. Both wait; she faces along her ledge         standing policy
    2. He falls forward across the gap              one joint-target blend
       and holds himself straight for SETTLE_S      HoldStraight
    3. She steps sideways over his back             Mjlab-Bridge-Sideways ONNX, constant vy
    4. She stands on the far ledge                  standing policy

`Judge` holds the rules for "she walked across". Read it before claiming a
success.

Why sideways: walking forward her feet sit 84 mm apart and his trunk is 64 mm
wide. Sideways her feet go one behind the other along his length and each
sole lies across his back. The policy was fine-tuned on his baked body
(`rl/microduck_bridge_sideways_env_cfg.py`); here he is the real, physical
brother again.

    python -m microduck_lab.tasks.human_bridge.rl_policy.story --policy bridge_v1.onnx
    MUJOCO_GL=egl python -m microduck_lab.tasks.human_bridge.rl_policy.story --policy bridge_v1.onnx \\
        --video videos/human_bridge/rl_policy/sidewalk_try1.mp4
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.tasks.human_bridge.rl_policy.brother import LIE_DOWN_S, PLANK, SETTLE_S, HoldStraight, smooth
from microduck_lab.tasks.human_bridge.rl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.rl_policy.world import World, Brain

# Same start as in training (microduck_bridge_sideways_env_cfg).
START_X = -0.25
HEADING = -math.pi / 2       # facing -y: her left is +x, across the gap
SIDE_SPEED = 0.11            # m/s, middle of the trained 0.08-0.14
LIE_S = LIE_DOWN_S           # his fall
DROP_Z = min(L.near_z, L.far_z) + 0.05


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


class Story:
    PHASES = ("wait", "he_lies", "cross", "across")

    def __init__(self, policy, verbose=True, blind=False):
        import onnxruntime as ort
        import mjlab_microduck.tasks  # noqa: F401  (registers tasks; must load before the lab cfg)
        from microduck_lab.rl.microduck_bridge_sideways_env_cfg import surface_table
        self.table = surface_table()
        self.blind = blind      # policies trained before the bridge-state slots existed
        self.world = w = World(L.design())
        self.he, self.she = w.ducks["he"], w.ducks["she"]
        self.brain_he = Brain(self.he, stand_only=True)
        # Her standing policy is alpha_stand; her walking one is the bridge policy.
        self.brain_she = Brain(self.she)
        self.brain_she.pol.walking_session = ort.InferenceSession(str(policy))
        # PolicyInference starts in walking mode with ort_session = its OWN walking
        # session (alpha_walking). Swap the active one too, or a walk that starts
        # without a stand-to-walk switch silently runs the old walker.
        if self.brain_she.pol.current_policy == "walking":
            self.brain_she.pol.ort_session = self.brain_she.pol.walking_session
        self.he.stand_on(L.gap_start - 0.005, 0.0, L.step_z, self.brain_he.default_pose)
        self.she.place((START_X, 0.0, L.near_z + 0.13), yaw=HEADING, q=self.brain_she.default_pose)
        soles = self.she.points(self.she.sole_geoms)
        p = self.she.pos()
        p[2] += L.near_z + 0.0005 - soles[:, 2].min()
        self.she.place(p, yaw=HEADING, q=self.brain_she.default_pose)
        w.start()
        self.judge = Judge(w, self.she, self.he)
        self.hold = HoldStraight(self.he)
        self.verbose = verbose
        self.phase, self.t_phase = "wait", 0.0
        self.events, self.failure = [], None
        self.scratch = {}
        self.max_offset = 0.0

    def note(self, text):
        self.events.append(dict(t=round(self.world.t, 2), event=text))
        if self.verbose:
            print(f"{self.world.t:6.2f}s  {text}", flush=True)

    def go(self, phase):
        self.note(f"{self.phase} -> {phase}")
        self.phase, self.t_phase = phase, self.world.t
        self.scratch = {}

    def command(self, vy):
        # The body-pose slots carry bridge state while she walks on him.
        on = vy and not self.blind
        self.brain_she.pol.body_cmd = bridge_state(self.she, self.table) if on else np.zeros(6, np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            self.brain_she.pol.set_vel_cmd(0.0, vy, 0.0)

    # -- one 50 Hz tick --------------------------------------------------------

    def tick(self):
        w, he, she = self.world, self.he, self.she
        u = w.t - self.t_phase
        if self.phase == "wait":
            self.brain_he.act()
            self.command(0.0)
            if u >= 1.5:
                self.go("he_lies")
        elif self.phase == "he_lies":
            # The blend, then he holds himself straight until he has settled.
            if u < LIE_S:
                start = self.scratch.setdefault("start", he.target.copy())
                he.hold((1 - smooth(u / LIE_S)) * start + smooth(u / LIE_S) * PLANK)
            else:
                self.hold()
            self.command(0.0)
            if u >= LIE_S + SETTLE_S:
                self.go("cross")
        elif self.phase == "cross":
            self.hold()
            self.command(SIDE_SPEED)
            if L.near_edge < she.pos()[0] < L.far_edge:
                self.max_offset = max(self.max_offset, abs(she.pos()[1]))
            if rearmost(she) > L.far_edge + 0.005:
                self.note(f"all of her is on the far ledge after {u:.1f} s")
                failure = self.judge.finish()
                if failure:
                    raise RuntimeError(failure)
                self.go("across")
        elif self.phase == "across":
            self.hold()
            self.command(0.0)
        self.brain_she.act()
        w.step()
        self.check()

    def check(self):
        if self.she.up() < 0.5:
            raise RuntimeError(f"she fell over at x={self.she.pos()[0]:.3f} m")
        if self.phase in ("cross", "across"):
            failure = self.judge.tick()
            if failure:
                raise RuntimeError(failure)

    def run_until(self, phase, limit=60.0):
        while self.phase != phase:
            if self.world.t > limit:
                raise RuntimeError(f"never reached {phase}")
            self.tick()


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="exported Mjlab-Bridge-Sideways ONNX")
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--video")
    ap.add_argument("--report")
    ap.add_argument("--blind", action="store_true",
                    help="feed zeros in the body-pose slots (policies from before bridge-v4)")
    a = ap.parse_args()
    if a.video and Path(a.video).exists():
        raise SystemExit(f"{a.video} exists; videos are never overwritten, pick a new name")
    s = Story(a.policy, blind=a.blind)
    writer = cam = None
    if a.video:
        import imageio.v2 as imageio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        cam = Camera(1280, 720)
        writer = imageio.get_writer(a.video, fps=25, codec="libx264", quality=None, macro_block_size=8,
                                    output_params=["-crf", "22", "-movflags", "+faststart"])
    tick = 0
    try:
        while s.world.t < a.seconds and not (s.phase == "across" and s.world.t - s.t_phase > 3.0):
            s.tick()
            if writer is not None and tick % 2 == 0:
                writer.append_data(cam.frame(s))
            tick += 1
        if s.phase != "across":
            raise RuntimeError(f"ran out of time in phase {s.phase} at x={s.she.pos()[0]:.3f} m")
    except RuntimeError as e:
        s.failure = str(e)
        s.note("FAILED: " + str(e))
    finally:
        if writer is not None:
            # A failure keeps filming a moment so the fall is on the clip.
            for _ in range(50):
                try:
                    s.world.step()
                except RuntimeError:
                    break
                writer.append_data(cam.frame(s))
            writer.close()
    report = dict(policy=str(a.policy), success=s.failure is None, failure=s.failure,
                  seconds=round(s.world.t, 2), max_offset_mm=round(1000 * s.max_offset),
                  he_up=round(s.he.up(), 3), **s.judge.report(), events=s.events)
    text = json.dumps(report, indent=1)
    if a.report:
        Path(a.report).write_text(text + "\n")
    print(text)
    return 0 if s.failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
