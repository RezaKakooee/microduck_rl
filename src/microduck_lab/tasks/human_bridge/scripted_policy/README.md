Latest review: [trial 111 crosses upright; shifted starts remain unresolved](LATEST_REVIEW.md).
The current test suite has 24 passing checks. The default gait file is available; the two interrupted starts are being rerun.

Current input status: `sideways_v1_iter250.onnx` is now available in
`local_storage/hb_dev/scripted_policy/policies/`. The user supplied its source path,
and the copied file matches the earlier trials by SHA-256. This is the
policy used by the current controller. Other historical configurations may
still need their own policy inputs.

# Self-contained Codex controller

The former base files were copied into this folder. `world.py`, `scene.py`
and `brother.py` keep the same physics and behavior. `story.py` contains the
copied camera, judging and observation helpers, without the other controller.
`bake.py` writes our own `bridge_pose.json` and a numbered bake video.
`baked_surface.py` reads only that local pose for optional bridge observations.
`protected_sources.json` records hashes of our own code and pose.

Run the filmed bake and six-second regression trial with:

```bash
sbatch -M cluster local_storage/hb_dev/scripted_policy/isolation.sbatch
```

The short trial compares its entire trace against the first six seconds of
saved trial 111. Its expected time limit is not a full-crossing success.
Results are in `ISOLATION_CHECK.json`. Videos and logs stay in Codex folders.

Verified: 22 tests passed. The six-second trial matched saved trial 111
exactly (maximum trace difference 0.0). It ended at the requested time limit;
a full crossing was not tested. The local bake wrote 39 body geoms.
The videos are `videos/human_bridge/scripted_policy/bake_try1.mp4` and
`videos/human_bridge/scripted_policy/scripted_try1.mp4`.

# Human bridge: codex's code

This folder belongs to codex. Other agents must not edit it.
Its videos go to videos/human_bridge/scripted_policy/ and its scratch work to local_storage/hb_dev/scripted_policy/.
The physics, scene, brother controller, camera, judging helpers and baked pose are owned here. Do not import bridge code from another agent's folder.
Its RL code (task configs, MDP terms) goes to src/microduck_lab/rl/scripted_policy/.

# Scripted human bridge

Work in progress. [Trial 105](../../../../../videos/human_bridge/scripted_policy/previous/scripted_try105.mp4)
crossed, recovered from a stumble, and stayed on the far ledge for 3 s.
It ended at trunk x = 0.560 m, up = 0.994, z = 0.4347 m, and rearmost
x = 0.4481 m. Minimum up was 0.7935 and minimum z was 0.3886 m.
His maximum absolute up during crossing was 0.0852.
She used non-foot support for 0.68 s, then ended on her feet. This is a
scramble with a recovery, not careful uninterrupted sideways walking.
The five-start validation in `sequence_validation.json` is pending.

The shared input files changed outside this controller during the session.
In particular, the far shelf changed from 0.282 m to 0.256 m, and the world
now configures collision masks before compilation. This controller did not
edit those files. Trial 105 uses the newer inputs. Its improvement cannot
be attributed solely to controller changes. New runs record input hashes,
the layout and a source ZIP beside the video.

Trial 89 also met the requested numerical end conditions for its final
10 seconds. It failed the runner's stricter landing rule. She rested on her
head, so this is not a walking success either. Trial 103 reached trunk
x = 0.215 m, then stalled with body support on him.

## Earlier five-start test

These used `robustness_cases.json`, including the older prediction state
copy and older scene. They are one fixed controller configuration, with five start offsets.
The nominal run differed from trial 89 because the worker predictions used
different cached BAM friction forces. Later trials copy those forces too.

| Video | Start x/y offset (mm) | End trunk x (m) | Minimum up | Minimum z (m) | Maximum brother abs(up) | Numerical result |
|---|---:|---:|---:|---:|---:|---|
| [92](../../../../../videos/human_bridge/scripted_policy/previous/scripted_try92.mp4) | 0 / 0 | 0.088 | 0.494 | 0.3713 | 0.0945 | Fell |
| [94](../../../../../videos/human_bridge/scripted_policy/previous/scripted_try94.mp4) | -20 / -10 | 0.098 | 0.499 | 0.3681 | 0.1210 | Fell |
| [95](../../../../../videos/human_bridge/scripted_policy/previous/scripted_try95.mp4) | -20 / +10 | 0.102 | 0.481 | 0.3680 | 0.1112 | Fell |
| [96](../../../../../videos/human_bridge/scripted_policy/previous/scripted_try96.mp4) | +20 / -10 | 1.048 | 0.501 | 0.3661 | 0.1084 | Passed, with head contact |
| [97](../../../../../videos/human_bridge/scripted_policy/previous/scripted_try97.mp4) | +20 / +10 | 0.109 | 0.398 | 0.3645 | 0.1090 | Fell |

Trial 90 was interrupted before the batch wall limit. Its 5.6 s video is
intact, but it has no scored end. It is not included as a completed test.

[Every attempt and its video](ATTEMPTS.md).

## Controller

`run.py` waits 1.5 s, calls the existing `LieDown`, then uses the existing
`HoldStraight`. She waits until 5.02 s before moving. These classes and the world and scene are local copies. The controller does not import another agent's bridge or RL code.

`controller.py` has three modes:

- `policy`: a saved sideways gait with position and heading feedback.
  Optional foot IK and pitch corrections are development experiments.
- `gait`: alternating foot targets and leg IK. These attempts fell on the
  near ledge. This mode does not solve the dynamic balance problem.
- `preview`: try commands in a separate two-robot world, then send the
  selected command to the live gait. Optional residuals adjust both ankle
  pitch targets, both hip roll targets, neck pitch and swing-foot lift.
  `sequence.py` searches short sequences of these corrections. It shifts
  the previous plan forward and refines it before executing the next part.

The preview restores only its own world. It copies live positions,
velocities, motor state and policy history. Each prediction includes his
`HoldStraight` controller. The live world receives joint targets only.
The existing `World.start()` checks remain active. Hypothetical predictions
are internal controller calculations; every live episode gets a video.

BAM also reads cached constraint forces, damping and friction. Copying
these into the prediction removed the observed next-step mismatch: trials
98–104 recorded zero maximum qpos prediction error. This is a check of
one-step reproduction, not proof that a chosen plan will cross successfully.
Optional CPU workers evaluate candidates in independent prediction worlds.

The preview uses finite candidate search, with a fixed random seed. No
policy is trained. The best candidate is recomputed every few control ticks.
This is an offline simulation controller; it has not been shown to run
at 50 Hz in wall time or to transfer to the robot.

`surface.py` intersects vertical lines with mesh convex hulls. It does not
ray-cast through the hollow shell or change collision geometry.

## Checks and outputs

A run fails when her trunk up is at most 0.5, her trunk z is below
0.365 m, or his absolute up reaches 0.35 during crossing or settling.
Clearance uses the original `rearmost()` bounding spheres, not trunk x.
The runner asks for 3 s beyond x = 0.255 m. It can either switch to the
standing policy or keep the preview active during this interval. The landing
margin and upright threshold control when the interval starts. With preview
settling, moving back resets the interval. `metrics.py` separately reports
the requested task limits, without these extra landing preferences.

Every attempt reserves a number atomically. Existing videos, JSON files
and reservation files prevent reuse. Videos use GPU EGL rendering at 25 fps.
Failed runs include a 1 s tail after the scored end. The score excludes it.

For `scripted_tryN`, the files are:

- `.mp4`: the full live episode and any failure tail.
- `.json`: settings, start offset, limits, result, events and policy hash.
- `.npz`: a 50 Hz trace, rendered qpos states and preview decisions.
- `.sources.zip`: the controller and shared Python inputs at the start of a new run.
- `.lock`: the permanent number reservation.

Trace columns are time, trunk x/y/z, trunk up, yaw, brother up, rearmost,
requested vx/vy/wz, left foot x/y/z, right foot x/y/z, and the two sole
normal forces against him. New traces add the normal force on body parts
other than her feet. The qpos frames omit the initial frame and
the failure tail. Earlier development files have no preview trace or
source hashes. New runs also save source hashes and wall time.
The trace is saved once per simulated second. New videos use short MP4
fragments so they remain readable if the job ends early.

## Run

From the repository root:

```bash
sbatch -M cluster --cpus-per-task=8 --time=00:30:00 local_storage/hb_dev/scripted_policy/video.sbatch \
  -m microduck_lab.tasks.human_bridge.scripted_policy.run \
  --config src/microduck_lab/tasks/human_bridge/scripted_policy/sequence_validation.json \
  --seconds 18
```

`--suite` repeats each configuration at the centre and four corners:
x offsets -20/+20 mm, y offsets -10/+10 mm. All are fresh filmed episodes.
Use `--suite --start-index 0` through `--start-index 4` for separate jobs.
Configuration files contain lists of `Settings` values. They are experiment
records, not claims of successful controllers.
Without `--config` or `--baseline`, the runner uses `sequence_validation.json`.

Several starts can share one GPU allocation. Each has its own process,
prediction workers, video and result. Reserve eight CPUs per start for the
default six-worker controller:

```bash
sbatch -M cluster --cpus-per-task=24 --time=00:30:00 local_storage/hb_dev/scripted_policy/video.sbatch \
  -m microduck_lab.tasks.human_bridge.scripted_policy.batch --indices 2 3 4 --seconds 18
```

`validation.py` takes the five saved trial JSON paths. It writes a table
and a JSON summary, checks the start offsets and controller settings, and
reports body contact separately. It also compares recorded controller,
policy and shared input hashes. Missing hashes are reported as missing.

For expensive preview tests, run one configuration per job with `--case N`.
`preview_workers` sets the number of prediction processes. Request enough
CPUs with `sbatch --cpus-per-task=N`. The six-worker experiments use eight
CPUs and `--time=00:30:00`; they still render through `video.sbatch` on a GPU.

CPU checks do not advance physics or render:

```bash
.venv/bin/python -m pytest -q \
  src/microduck_lab/tasks/human_bridge/scripted_policy/test_controller.py
.venv/bin/python -m microduck_lab.tasks.human_bridge.scripted_policy.report
```

The tests cover video reservation, invalid settings, command direction,
unchanged live positions and velocities during control, and the real gap
in the height map. Physical performance is measured by the filmed runs.
