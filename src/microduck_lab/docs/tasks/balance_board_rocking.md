# Active board play: sustained rocking

> **Follow-up (2026-09-05):** the motion here is perfectly symmetric left/right,
> which is part of why it is easy to hold. Asymmetric, faster (up to 3 Hz) and
> wider (31.5 deg of range) variants are in
> [balance_board_rocking_asym.md](balance_board_rocking_asym.md); they use
> `src/microduck_lab/tasks/balance_board/rocking_asym.py` and do not modify anything here.


> **Camera note (measured 2026-09-05).** MuJoCo's tracking-camera azimuth for
> this scene is: **180 = FRONT** (the face and camera lens are visible),
> **0 = BACK**, 90 and 270 = the two sides. An earlier render used
> `--azimuth 0` and was labelled "front"; it is actually the back of the head.
> Front view: `videos/balance_board/board_rocking_front.mp4`.


**2026-09-05, revised after user review.** The intended behavior is for the duck
to play with the board by rocking it repeatedly while balancing. The earlier
static LQR hold addressed survival but missed this visible-motion requirement.

The new expert rocks the original board **about 6.7° to each side at 1.2 Hz**,
a complete left/right cycle every 0.83 s. It ramps up over the first 3 s, then
continues throughout the episode. The roller moves about **26 mm peak to peak**
each cycle in the nominal video. All movement is produced by the duck's existing
14 servos, with the original torque limits and geometry.

- **[30-second video](../../videos/balance_board/board_rocking.mp4)** — no external pushes.
- [Opposite phases at 4, 14 and 28 seconds](../../videos/balance_board/board_rocking_frames.jpg).
- [Motion over the whole video](../../videos/balance_board/board_rocking_motion.png).
- [Expert source](../../src/microduck_lab/tasks/balance_board/rocking.py).
- [Saved policy](../../videos/balance_board/board_rocking_policy.npz).
- [Video results](../../videos/balance_board/board_rocking.json), [CSV trace](../../videos/balance_board/board_rocking.csv).
- [Validation battery](../../videos/balance_board/board_rocking_validation.json).

## Reproduce

From `$REPO`:

```bash
# Build and run the active expert for 30 seconds.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking

# Replay the saved expert.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking \
  --load-policy videos/balance_board/board_rocking_policy.npz --seconds 30

# Reproduce the 17 motion-validation episodes.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking \
  --suite --output videos/balance_board/board_rocking_validation.json

# All static and active balance regression tests.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_balance_board_*.py' -v

# Render on a GPU node.
  MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.rocking --load-policy videos/balance_board/board_rocking_policy.npz --azimuth 0 --video videos/balance_board/board_rocking.mp4
```

`--amplitude-deg`, `--frequency-hz`, and `--ramp-seconds` control a newly designed
motion. Loading a saved policy uses its saved motion settings. The default
**8° reference** produces roughly **6.7° measured** rocking because the local
linear model does not exactly predict the nonlinear contact dynamics. Larger
or slower commands can fail; their success is not assumed.

## Controller

The static LQR gain remains the feedback stabilizer. A new calculation finds
a periodic trajectory for the *coupled* duck, plank and free roller. It solves
for joint commands and state amplitudes at the requested frequency, including
their phase differences, while minimizing body displacement and servo effort.

At each 20 ms control step:

```
reference   = ramp(t) * real(state_wave  * exp(j * omega * t))
feedforward = ramp(t) * real(action_wave * exp(j * omega * t))
command     = clip(u0 + feedforward - K * (state_error - reference))
```

The smooth ramp reaches full amplitude after 3 s. Feedback balances about the
moving trajectory instead of opposing every board motion. The simulator supplies
time explicitly, so replaying the same saved expert starts a fresh cycle sequence
for every episode. No controller changes simulator state, applies board forces,
or pins the roller. The nominal video has no scripted disturbance.

Slow rocking is not automatically easier: in the tested 0.35 Hz, 5° design,
the linear model required roughly 73 mm of roller displacement amplitude, and
the rollout failed. At 1.2 Hz, the default trajectory needs much less travel
and stays inside the original plank.

## What counts as success now

The original contact checks still run at every 5 ms physics step. Additionally,
**every complete cycle after the ramp** must:

1. Reach at least **5.6° in both directions** (70% of the 8° reference).
2. Move the roller at least **5 mm peak to peak**.
3. Have a full cycle's worth of recorded samples.

Success requires both the contact/height checks and this continuous-motion
criterion. Standing still, reacting once to a push, or stopping late in the
video fails. The tests explicitly cover those three failure cases. The 30 s
video contains **32/32 qualifying cycles** after its ramp; the 60 s test
contains **68/68**.

| Validation | Episodes | Result |
|---|---:|---|
| Nominal 60 s continuous rocking | 1 | Passed, 68 cycles |
| Random initial velocities, seeds 0–9, 20 s each | 10 | All passed |
| ±0.01 m/s lateral kicks at 5.0 or 5.2 s, 20 s each | 4 | All passed |
| Friction 1.0 and 1.8, 20 s each | 2 | Both passed |

**17/17 episodes, 388/388 required cycles passed.** Across this battery there
was no foot contact loss, plank-roller contact loss, plank-floor contact,
foot-floor contact, or non-foot robot contact. Minimum board-floor clearance
was **41.2 mm**. All **11 static and rocking regression tests** passed.

The nominal video reaches **±6.7° board tilt**, has at least **43.0 mm** floor
clearance, and keeps both feet in contact throughout. Opposite tilt phases
were visually inspected early, midway and late in the actual video.

This remains a simulation expert with privileged robot/platform state, not a
61D hardware ONNX policy. It does not establish hardware performance. The
[static LQR report](balance_board_lqr.md) retains the original survival results
and research references.
