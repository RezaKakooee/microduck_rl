# Salmon jump experiments

Status: a small upward hop is demonstrated, but a clean back-to-feet landing
has not yet been demonstrated. This is a motor-target search, not a newly
trained RL policy. The existing standing policy is used for recovery.

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
