# Salmon jump experiments

Status (2026-09-30): clean back-to-feet jumps in simulation (trials 1212, 1216).
They pass the clean-landing judge. The hop is small and only just above the
judge's flight limit. This is a motor-target search plus a closed-loop catch,
not a trained RL policy. Not ready for hardware.

Use `reference_salmon.json` (trial 1216). It is trial 1212's launch, but the
push keyframe waits until the IMU pitch reaches -0.86 rad, and the crouch
before the handoff is 0.1 s.

## First clean landings (trials 1212-1217)

400 Hz audits (trial 1215 replays 1212, trial 1217 replays 1216; slow videos
`salmon_try1215_slow.mp4`, `salmon_try1217_slow.mp4`):

| measure | 1212 | 1216 (with sync) |
|---|---|---|
| flight, no ground contact | 47.5 ms | 47.5 ms |
| peak clearance | 4.0 mm | 4.0 mm |
| COM speed at takeoff | 0.38 m/s up | 0.35 m/s up |
| COM rise during flight | 6.8 mm | 5.9 mm |
| body contact after takeoff | 0 N | 0 N |
| standing at the end | 4.8 s | 4.8 s |

The motion: roll up from the back, land on the feet in a crouch, push, then
fold the legs fast (a 40 ms keyframe). The feet leave the ground for 47.5 ms.
The robot lands in a crouch and the standing policy takes over. The hop is too
small to see in real time; use the slow replay.

Trials 1213 (another launch) and 1214 (0.1 s crouch) also pass.
Cases: `handoff_cases.json`.

### Why every earlier hop failed

Every earlier hop landed with the COM 43-61 mm behind the sole centre.
The sole is 54 mm long, so the COM was behind the heel. No standing controller
can hold that. Measured by re-running all 250 qualifying hops of `review_2.json`.

Other measured facts:

- The robot cannot hop from a balanced crouch. The COM rises 30-64 mm but the
  feet never leave the ground. The flight needs the roll-up momentum.
- Under load the servos lag their targets by up to 0.8 rad. A scripted rise
  from the crouch failed 54 of 54 times. Hand over to the standing policy.
- In flight, moving the head forward turns the trunk backward, and the other
  way round. The servos move the legs only about 0.1-0.2 rad in a flight.

### What was added

- `catch.py`: the catch after takeoff. In flight it solves leg angles that put
  a flat sole under the COM (inverse kinematics on a scratch MjData). After
  touchdown it holds a crouch with ankle and hip feedback, then hands over.
  It uses only joint angles and the IMU, as the real robot can.
- `run.py`: the physics loop is now one `Episode` class, shared by the filmed
  runner and the search. Old trials replay exactly (trials 1211, 1212: trace
  difference 0.0). New: `recovery='catch'`, and an optional keyframe that
  waits for the IMU pitch (`sync_pose`, `sync_pitch`).
- `catch_search.py`: headless CPU search on Slurm, no video. It scores the COM
  position over the soles at arrival and at touchdown. `--perturb N` scores
  each candidate on N noisy copies. Film the winners with `batch.py`.

### Robustness (better, not solved)

The same 60 script changes for both (`lander/robust_check.py`, fixed draws):

| change | 1212 | 1216 (with sync) |
|---|---|---|
| keyframe angles, noise 0.02 rad | 6 / 30 | 16 / 30 |
| start pitch, +-0.03 rad | 2 / 10 | 7 / 10 |
| durations, noise 5 ms | 2 / 20 | 5 / 20 |
| total | 10 / 60 | 28 / 60 |

Physics changes (`lander/physics_check.py`; applied before `World.start()`):

| change | 1212 | 1216 (with sync) |
|---|---|---|
| none | pass | pass |
| battery 6.8 / 7.0 / 7.2 V | fail (landing) | fail (flight too short) |
| battery 7.6 / 7.8 V | fail (landing) | pass |
| floor friction x0.5 / x0.7 | pass | pass |
| floor friction x1.3 | fail | pass |
| mass x0.95 / x0.98 | fail | pass |
| mass x1.02 / x1.05 | fail / pass | fail (flight too short) |
| total | 4 / 13 | 8 / 13 |

With the sync, the landing held in every physics case. The failures are now
a hop that is too small: 2.4-2.9 mm peak clearance, and the judge needs two
50 Hz samples above 2 mm. The flight also needs the leg fold at the moment of
peak upward speed; changing the push or fold time by a few ms loses it.

Next useful work: search for a higher hop with `--sync-pose 2 --perturb 6`
and the `height` objective. A higher hop gives margin at low battery voltage.

### Reproduce

```bash
# score candidates headless (CPU node, no video)
.venv/bin/python -m microduck_lab.rl.scripted_policy.salmon_jump.catch_search \
  rescore --cases src/microduck_lab/rl/scripted_policy/salmon_jump/reference_salmon.json --workers 1
# film (GPU node), then the 400 Hz audit with a slow replay
sbatch -M cluster local_storage/hb_dev/scripted_policy/video.sbatch \
  -m microduck_lab.rl.scripted_policy.salmon_jump.batch \
  --config src/microduck_lab/rl/scripted_policy/salmon_jump/reference_salmon.json --workers 1 --seconds 9
sbatch -M cluster local_storage/hb_dev/scripted_policy/video.sbatch \
  -m microduck_lab.rl.scripted_policy.salmon_jump.audit \
  --trial videos/human_bridge/scripted_policy/salmon_jump/salmon_try1216.json --slow-motion
# robust search from successes (big CPU node; about 1 min per generation)
.venv/bin/python -m microduck_lab.rl.scripted_policy.salmon_jump.catch_search search \
  --seeds local_storage/hb_dev/scripted_policy/salmon_jump/catch_search/rescore_6.json --num-seeds 14 \
  --fix '{"rise_s": 0.0, "pitch_rate_limit": 100.0}' --sync-pose 2 --perturb 6 --population 24 --workers 60
```

Give each search worker about 1.5 GB of memory. With less, the pool stalls.

Search outputs are in `local_storage/hb_dev/scripted_policy/salmon_jump/catch_search/`.
Scratch scripts (trace, checks, Slurm drivers) are in `.../salmon_jump/lander/`.

## Earlier work (before 2026-09-30)

Status then: a small upward hop is demonstrated, but a clean back-to-feet
landing has not yet been demonstrated. The existing standing policy is used
for recovery.

All code, scratch data and videos belong to `scripted_policy`. No other
human-bridge policy implementation is used. The world uses the owned
`tasks/human_bridge/scripted_policy/world.py`, with its original BAM XL330
motors, mass, gravity, voltage and force limits. Live trials change motor
targets only; initial placement occurs before `World.start()`.

## What changed after trial 457

- Arbitrary-length symmetric sagittal keyframes replace the three-pose limit.
  Smooth, linear and step target commands are available.
- A flight-triggered recovery handoff, optional brief home-pose brace, gradual
  head transition and fading hip/ankle target offsets can be compared.
- Candidate evaluations run in isolated processes on one GPU. Every episode
  receives a fresh numbered video, telemetry, source hashes and source ZIP.
- Search resumption re-evaluates the seed under the current objective rather
  than trusting an old score. Batch inputs are validated before any rollout.
- Source snapshots are captured before the episode. Telemetry includes joint
  velocities and motor torques as well as contacts, clearance and COM motion.
- The judge measures takeoff velocity at initial contact loss, rather than
  when the robot later exceeds 2 mm clearance. A hop after a settled standing
  pause does not count. Body contact after takeoff disqualifies a clean landing.

Historical JSON `success` flags use the judge archived with that trial.
In particular, some earlier hop-then-stumble-then-stand episodes passed the
old final-standing-only check. They fail the current clean-landing check.
`local_storage/hb_dev/scripted_policy/salmon_jump/review_1.json` re-evaluates
trials 458–774 with the stricter judge: 48 upward hops, zero clean successes.
Do not overwrite historical results to change their original verdicts.

## Measured findings

- Trial 457 rocks without a qualifying flight.
- Extra push/tuck stages can generate real but small hops. Several earlier
  contact-free intervals were merely falling and were rejected.
- Trial 995 replays trial 930 with a read-only 400 Hz observer. The 50 Hz
  physical trace is identical (maximum absolute difference 0). It measures
  75 ms continuously unsupported, 0.380 m/s upward velocity at takeoff,
  4.94 mm maximum whole-robot clearance and 6.93 mm COM rise *during flight*.
  It subsequently stumbles and therefore fails the clean-landing criterion.
  COM rise from the supine start is a different quantity and is not jump height.
- In trial 930 the feet are roughly 5 cm ahead of the COM near landing. The
  recovery controller then bends the knees and the robot tips backward.
- Head-transition sweeps did not solve the landing. An early version also
  clipped untouched leg targets; that confound was fixed and head-only probes
  repeated. Policy target overshoot is intentionally preserved.
- Hip/ankle offsets alone did not produce a clean landing. A scripted brace
  before standing-policy handoff reduced some stumbles but did not solve landing.
- All 48 earlier-catch timing probes removed the qualifying hop. Preserving
  the launch while preparing the feet earlier remains unresolved.

## Latest review

Trial 1211 is the latest audited replay of the saved `reference_hop.json`
candidate, with wider framing. Watch
`videos/human_bridge/scripted_policy/salmon_jump/salmon_try1211_slow.mp4`
for the labelled quarter-speed replay, or `salmon_try1211.mp4` for real time.
At 400 Hz it has 77.5 ms continuously unsupported, 8.45 mm peak whole-robot
clearance and 0.324 m/s COM velocity at takeoff. The COM rises only 5.05 mm
during flight. The recorded trajectory matches the source trial exactly.
It lands, tips backward, touches its body and later stands. This is a small
hop with failed landing, not a completed salmon jump.

`review_2.json` in the owned scratch directory rechecks 753 completed trials
458–1210: 250 qualifying upward hops and **zero clean successes**. Twelve
CPU tests pass. There has been no RL training and no hardware deployment.
The replay video can be generated with `--slow-motion`; this renders saved
400 Hz joint/root states at quarter speed and labels contact and clearance.
It does not interpolate a fake flight or run altered dynamics.

Next useful work is to prepare foot placement without cutting off the push,
and/or train a catch controller on the measured takeoff states. More broad
head/hip offsets have not addressed the remaining failure. Any RL branch
must first preserve the 61D observation and BAM/DR contracts and pass the
required 64-environment, five-iteration smoke test.

## Reproduce

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/microduck_lab/rl/scripted_policy/salmon_jump/test_salmon.py

sbatch -M cluster --cpus-per-task=8 --mem=16G \
  local_storage/hb_dev/scripted_policy/video.sbatch \
  -m microduck_lab.rl.scripted_policy.salmon_jump.batch \
  --config src/microduck_lab/rl/scripted_policy/salmon_jump/brace_cases.json \
  --workers 4 --seconds 9

sbatch -M cluster --cpus-per-task=2 --mem=4G \
  local_storage/hb_dev/scripted_policy/video.sbatch \
  -m microduck_lab.rl.scripted_policy.salmon_jump.audit \
  --trial videos/human_bridge/scripted_policy/salmon_jump/salmon_try773.json \
  --slow-motion
```

The audit creates another numbered video plus `.physics.npz` and
`.audit.json`. It records solved contacts and pre-integration kinematics at
each physics step, without changing the controller. Compare its replay trace
before trusting a new observer implementation.

Current clean success requires a stable supine start, upward takeoff, at
least two 50 Hz samples with zero ground force and >2 mm whole-robot clearance,
COM rise from the supine start of at least 25 mm, no subsequent body-supported
landing, and at least two final seconds standing on feet. These are minimum
numerical gates: watch the video before describing the skill as successful.
