"""Film every trial on a GPU node. Keep its settings and measurements.

sbatch -M cluster --cpus-per-task=8 --time=00:30:00 local_storage/hb_dev/scripted_policy/video.sbatch -m microduck_lab.tasks.human_bridge.scripted_policy.run
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import signal
import time
import traceback
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from microduck_lab.tasks.human_bridge.scripted_policy.world import World, Brain, CONTROL_DT
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.scripted_policy.brother import LieDown, HoldStraight
from microduck_lab.tasks.human_bridge.scripted_policy.story import Camera, rearmost, DROP_Z
from microduck_lab.tasks.human_bridge.scripted_policy.controller import HEADING, Settings, Supervisor
from microduck_lab.tasks.human_bridge.scripted_policy.metrics import audit, nonfoot_support
from microduck_lab.tasks.human_bridge.scripted_policy.runtime import brain as single_thread_brain, require_policy


def reserve(folder):
    """Reserve a new number, even when two batches start together."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        stem = folder / f"scripted_try{number}"
        if stem.with_suffix(".mp4").exists() or stem.with_suffix(".json").exists():
            continue
        try:
            fd = os.open(stem.with_suffix(".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return stem
    raise RuntimeError("No free video number")


class Trial:
    def __init__(self, cfg, dx=0.0, dy=0.0):
        self.world = w = World(L.design())
        self.he, self.she = w.ducks["he"], w.ducks["she"]
        self.bh = single_thread_brain(self.he, stand_only=True)
        self.bs = single_thread_brain(self.she)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.bs.pol.walking_session = ort.InferenceSession(cfg.policy, sess_options=options,
                                                           providers=["CPUExecutionProvider"])
        self.he.stand_on(L.gap_start - .005, 0, L.step_z, self.bh.default_pose)
        self.she.place((-.25 + dx, dy, L.near_z + .13), yaw=HEADING, q=self.bs.default_pose)
        p = self.she.pos()
        p[2] += L.near_z + .0005 - self.she.points(self.she.sole_geoms)[:, 2].min()
        self.she.place(p, yaw=HEADING, q=self.bs.default_pose)
        w.start()
        self.control = Supervisor(self.bs, cfg)
        self.phase, self.t_phase = "wait", 0.0
        self.lie, self.hold = None, HoldStraight(self.he)
        self.control.brother_hold = self.hold
        self.prediction_errors = []
        self.events = []
        self.min_up, self.min_z, self.max_he_up, self.max_offset = 1., 9., 0., 0.
        self.furthest_x = self.she.pos()[0]

    def note(self, event):
        self.events.append({"t": round(self.world.t, 3), "event": event})

    def go(self, phase):
        self.note(f"{self.phase} -> {phase}")
        self.phase, self.t_phase = phase, self.world.t

    def tick(self):
        w = self.world
        if self.phase == "wait":
            self.bh.act()
            if w.t >= 1.5:
                self.lie = LieDown(self.he, self.he.target.copy(), w.t)
                self.go("he_lies")
        elif self.phase == "he_lies":
            if self.lie(w.t):
                self.hold()
            if w.t - self.t_phase >= 3.5:
                self.go("cross")
        else:
            self.hold()
        self.control.act(self.phase == "cross" or
                         (self.phase == "across" and self.control.cfg.settle_with_preview))
        w.step()
        mujoco.mj_forward(w.model, w.data)
        preview = self.control.preview
        if preview is not None and preview.expected is not None:
            time, state = preview.expected
            if abs(w.t - time) < 1e-6:
                self.prediction_errors.append(float(np.max(np.abs(w.data.qpos-state))))
        she = self.she
        self.min_up = min(self.min_up, she.up())
        self.min_z = min(self.min_z, she.pos()[2])
        self.furthest_x = max(self.furthest_x, she.pos()[0])
        if she.up() <= .5:
            return "she fell over"
        if she.pos()[2] < DROP_Z:
            return "she dropped below DROP_Z"
        if self.phase in ("cross", "across"):
            self.max_he_up = max(self.max_he_up, abs(self.he.up()))
            if abs(self.he.up()) >= .35:
                return "he was knocked loose"
            if L.near_edge < she.pos()[0] < L.far_edge:
                self.max_offset = max(self.max_offset, abs(she.pos()[1]))
        return self.check_landing()

    def check_landing(self):
        cfg = self.control.cfg
        ready = (rearmost(self.she) > L.far_edge + .005 + cfg.landing_margin
                 and self.she.up() > cfg.landing_up
                 and (cfg.landing_support_limit < 0
                      or nonfoot_support(self.she) <= cfg.landing_support_limit))
        if self.phase == "cross" and ready:
            if (cfg.landing_entry_speed >= 0 and np.linalg.norm(
                    self.she.data.qvel[self.she.dof:self.she.dof+2]) > cfg.landing_entry_speed):
                return None
            if cfg.landing_entry_height >= 0 and self.she.pos()[2] < cfg.landing_entry_height:
                return None
            self.go("across")
        elif self.phase == "across" and not ready:
            if cfg.settle_with_preview:
                self.go("cross")
            else:
                # Keep standing while it recovers. Only the hold timer resets;
                # the hard fall and height checks still run on every tick.
                self.t_phase = self.world.t + CONTROL_DT
        return None

    def row(self):
        she, w = self.she, self.world
        contacts = []
        for g in she.sole_geoms:
            contacts.append(sum(c[2] for c in w.contacts({g}, set(self.he.solid))))
        return [w.t, *she.pos(), she.up(), she.yaw(), self.he.up(), rearmost(she),
                *self.control.command, *np.ravel(she.feet()), *contacts, nonfoot_support(she)]


def save_trace(stem, rows, states, preview):
    temporary = stem.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, trace=np.array(rows), qpos=np.array(states),
                            preview=np.array(preview.scores if preview is not None else []),
                            sequences=np.array(getattr(preview, 'sequences', [])))
    temporary.replace(stem.with_suffix('.npz'))


def run(cfg, dx=0., dy=0., seconds=30., folder="videos/human_bridge/scripted_policy", width=640):
    """Each fresh physical trial has a video, a JSON report and a trace."""
    if os.environ.get("MUJOCO_GL") != "egl" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run through video.sbatch on a GPU node; no GL on the login node")
    require_policy(cfg.policy)
    import imageio.v2 as imageio
    stem = reserve(folder)
    started = time.monotonic()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    shared_folder = Path(__file__).parent
    input_files = [shared_folder / name for name in ('world.py', 'scene.py', 'brother.py', 'story.py', 'bake.py', 'bridge_pose.json')]
    input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    with zipfile.ZipFile(stem.with_suffix('.sources.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(set(Path(__file__).parent.glob('*.py')) | set(input_files)):
            archive.write(p, arcname=p.relative_to(Path.cwd()) if p.is_absolute() else p)
    trial = Trial(cfg, dx, dy)
    camera = Camera(width, width * 9 // 16)
    writer = imageio.get_writer(str(stem.with_suffix(".mp4")), fps=25, codec="libx264",
                                quality=None, macro_block_size=2,
                                output_params=["-crf", "24", "-g", "25", "-flush_packets", "1", "-movflags",
                                               "+frag_keyframe+empty_moov+default_base_moof"])
    rows, states = [], []
    failure = None
    error = None
    end = None
    try:
        writer.append_data(camera.frame(trial))
        for tick in range(round(seconds / CONTROL_DT)):
            failure = trial.tick()
            rows.append(trial.row())
            if tick % 50 == 49:
                print(f"{stem.name}: t={trial.world.t:.1f} phase={trial.phase} "
                      f"x={trial.she.pos()[0]:.3f} y={trial.she.pos()[1]:.3f} "
                      f"up={trial.she.up():.3f}", flush=True)
            if tick % 2 == 1 or failure:
                writer.append_data(camera.frame(trial))
                states.append(trial.world.data.qpos.copy())
            if tick % 50 == 49:
                save_trace(stem, rows, states, trial.control.preview)
            if failure or (trial.phase == "across" and trial.world.t - trial.t_phase >= 3.):
                break
        if not failure and (trial.phase != "across" or trial.world.t - trial.t_phase < 3.):
            failure = "time limit"
        end = dict(time=trial.world.t, pos=trial.she.pos().tolist(), up=trial.she.up(),
                   he_up=trial.he.up(), rearmost=rearmost(trial.she))
        # Keep the failed landing in view. It does not change the scored result.
        if failure:
            trial.note(f"FAILED: {failure}")
            for i in range(50):
                trial.hold()
                trial.world.step()
                if i % 2 == 1:
                    writer.append_data(camera.frame(trial))
    except Exception:
        error = traceback.format_exc()
        failure = "runtime error (see exception in JSON)"
        trial.note(failure)
    finally:
        writer.close()
        if camera.renderer is not None:
            camera.renderer.close()
        if trial.control.preview is not None:
            trial.control.preview.close()
    if end is None:
        end = dict(time=trial.world.t, pos=trial.she.pos().tolist(), up=trial.she.up(),
                   he_up=trial.he.up(), rearmost=rearmost(trial.she))
    result = dict(success=failure is None, failure=failure, settings=asdict(cfg), dx=dx, dy=dy,
                  end=end, min_up=trial.min_up, min_z=trial.min_z,
                  furthest_x=trial.furthest_x, max_he_up=trial.max_he_up,
                  max_offset_mm=trial.max_offset * 1000, events=trial.events,
                  video=str(stem.with_suffix(".mp4")),
                  max_prediction_qpos_error=max(trial.prediction_errors, default=None),
                  wall_seconds=time.monotonic()-started, source_sha256=source_hashes,
                  input_sha256=input_hashes,
                  layout={**asdict(L), 'near_z': L.near_z, 'far_z': L.far_z, 'shelf_z': L.shelf_z},
                  exception=error,
                  task_audit=audit(rows),
                  policy_sha256=hashlib.sha256(Path(cfg.policy).read_bytes()).hexdigest())
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    preview = trial.control.preview
    save_trace(stem, rows, states, preview)
    print(json.dumps(result), flush=True)
    if error:
        raise RuntimeError(f"{stem}: {error}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--config", help="JSON list of settings and optional dx, dy")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--start-index", type=int, choices=range(5),
                    help="Run one of the five suite starts in this job")
    ap.add_argument("--case", type=int, help="Run one entry from a configuration list")
    ap.add_argument("--seconds", type=float, default=30.)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()
    if a.start_index is not None and not a.suite:
        ap.error("--start-index requires --suite")
    cases = json.loads(Path(__file__).with_name('sequence_validation.json').read_text())
    if a.baseline:
        cases = [dict(policy=f"local_storage/hb_dev/scripted_policy/policies/{p}.onnx", speed=v)
                 for p in ("sideways_v1_iter250", "sideways_v2_iter4750") for v in (.08, .14, .2)]
    if a.config:
        cases = json.loads(Path(a.config).read_text())
    if a.case is not None:
        cases = [cases[a.case]]
    def interrupted(signum, frame):
        raise TimeoutError(f"Slurm sent signal {signum}; saving this attempt")
    signal.signal(signal.SIGTERM, interrupted)
    for case in cases:
        case = case.copy()
        offsets = [(case.pop("dx", 0.), case.pop("dy", 0.))]
        if a.suite:
            offsets = [(0., 0.), (-.02, -.01), (-.02, .01), (.02, -.01), (.02, .01)]
        if a.start_index is not None:
            offsets = [offsets[a.start_index]]
        for dx, dy in offsets:
            run(Settings(**case), dx, dy, a.seconds, width=a.width)


if __name__ == "__main__":
    main()
