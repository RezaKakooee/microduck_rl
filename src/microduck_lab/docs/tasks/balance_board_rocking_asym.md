# Asymmetric and faster board play

**2026-09-05.** Follow-up to [balance_board_rocking.md](balance_board_rocking.md),
after a user observation: the duck balanced easily because the rocking was
perfectly symmetric left and right. Three questions were asked and answered:
can the motion be made **asymmetric**, **faster**, and **wider**?

All three: yes. Script: `src/microduck_lab/tasks/balance_board/rocking_asym.py`. Nothing in
the symmetric task, its policy file, or its videos was modified.

> Camera: **azimuth 180 = FRONT**, 0 = back, 90/270 = sides. Verified by
> rendering one frame at each angle and looking for the face.

---

## 1. What the symmetric motion was, and why it is easy

The solved rocker drives a pure sinusoid:

    reference(t)   = Re( state_wave  * exp(i * 2*pi*f*t) )
    feedforward(t) = Re( action_wave * exp(i * 2*pi*f*t) )

with f = 1.2 Hz. The board roll is a clean +/-6.7 deg sine of constant amplitude
and period — a limit cycle. It is left/right symmetric by construction: every
push one way is mirrored exactly the other way, so the net lateral impulse over
a cycle is zero and the roller returns to the same place each time.

## 2. Breaking the symmetry

Three modulations of the *same* solved wave, applied to the reference and the
feedforward together so the two stay consistent:

| knob | effect | formula |
|---|---|---|
| `skew` | unequal dwell: linger one side, snap through the other | `theta(t) = 2*pi*f*t + skew * sin(2*pi*f*t)` |
| `bias` | unequal amplitude, one side swings wider | `amp(theta) = 1 + bias * cos(theta)` |
| `drift` | breaks exact periodicity | `amp *= 1 + drift * sin(2*pi*t / drift_period)` |

`skew` warps only the *speed* along the trajectory, not the path. `bias` scales
the two halves. The feedforward was solved for a constant rate `2*pi*f`, so
warping the phase leaves a torque mismatch the LQR feedback has to absorb —
that is the real question these runs answer.

**Result (20 s runs, amplitude 8 deg, 1.2 Hz):**

| skew | bias | drift | peak + | peak - | held |
|---|---|---|---|---|---|
| 0 | 0 | 0 | +6.7 | -6.7 | 20/20 cycles |
| 0.6 | 0.25 | 0 | +9.3 | -7.2 | 20/20 |
| 0.6 | 0.25 | 0.2 | +11.1 | -8.5 | 20/20 |
| 0.9 | 0.45 | 0.3 | **+14.3** | -9.1 | balanced 30 s |

At the strongest setting one side swings **57% further** than the other and it
still never falls. The balance does not depend on the symmetry.

Video: `videos/balance_board/board_rocking_asym.mp4`, plot `videos/balance_board/board_rocking_asym_motion.png`.

## 3. Faster

The frequency is baked into the solved wave: `state_wave` / `action_wave` are the
periodic solution of the linearised robot/plank/roller dynamics **at one rate**.
Driving the saved wave faster would leave the feedforward solving the wrong
problem, so `--frequency-hz` re-solves via `rocking.design()`.

At the original 8 deg amplitude:

| rate | result |
|---|---|
| 1.2 Hz | holds |
| 1.6 Hz | holds |
| 2.0 Hz | holds |
| 2.5 Hz | falls at 12 s |
| 3.0+ Hz | falls in 3-5 s |

Speed trades against swing, because a faster rock needs larger accelerations.
Over 30 s:

| rate | amplitude | result |
|---|---|---|
| 2.0 Hz | 8 deg | falls at 15 s |
| **2.0 Hz** | **6 deg** | **holds** |
| 2.5 Hz | 6 deg | falls at 18 s |
| **3.0 Hz** | **4 deg** | **holds** |

So **2.5x the original rate** is reachable at a reduced swing. Every failure is
the same event: a **leg touches the plank**. It runs out of leg clearance before
it runs out of balance.

Videos: `board_rocking_fast_2hz.mp4`, `board_rocking_fast_3hz.mp4`.

## 4. Wider, at the original speed

| | original | wide |
|---|---|---|
| peak one way | +6.7 deg | **+19.5 deg** |
| peak other way | -6.7 deg | **-12.0 deg** |
| total range | 13.4 deg | **31.5 deg** |
| held | 30 s | 30 s, all 32 cycles |

Settings: 1.2 Hz, amplitude 12, skew 0.9, bias 0.20, drift 0.2.

**The bias had to come DOWN from 0.45 to 0.20.** A strong bias puts all the extra
swing on one side, so that peak reaches the limit early while the other side
stays small. Sharing it out buys more total range. The skew stays high: dropping
it from 0.9 to 0.3 made the duck fall within 3 s.

**Where the ceiling is, and it is the ankle again.** The failure is *not* the
plank hitting the floor, which would need 24 deg each way. It is a **foot lifting
off the plank**: amplitude 13 lets go at 16 s, amplitude 14 at 9 s, peaks +22.5
and +24.2 deg. The plank tilts under the sole and, with no ankle-roll joint, the
sole tilts with it; past about 20 deg the foot is on its edge and loses the
plank. The same missing joint that closed the one-legged pose sets this limit.

Going *slower* does not help: at 0.8 and 0.6 Hz the solve produced controllers
that fell in under 3 s.

Video: `board_rocking_wide.mp4`, plot `board_rocking_wide_motion.png`.

## 5. Commands

```bash
cd $REPO
# asymmetric, saved 1.2 Hz wave
uv run python src/microduck_lab/tasks/balance_board/rocking_asym.py --skew 0.9 --bias 0.45 --drift 0.3
# how much asymmetry survives / how fast it can go
uv run python src/microduck_lab/tasks/balance_board/rocking_asym.py --sweep
uv run python src/microduck_lab/tasks/balance_board/rocking_asym.py --speed-sweep
# a new rate or amplitude re-solves the wave
uv run python src/microduck_lab/tasks/balance_board/rocking_asym.py --frequency-hz 3.0 --amplitude-deg 4
# render (GPU node; 180 = front)
 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
  uv run --with imageio --with imageio-ffmpeg python src/microduck_lab/tasks/balance_board/rocking_asym.py \
  --frequency-hz 1.2 --amplitude-deg 12 --skew 0.9 --bias 0.20 --drift 0.2 \
  --seconds 30 --azimuth 180 --video videos/balance_board/board_rocking_wide.mp4
```

## 6. Honest limits

- One deterministic run per row. No noise, no domain randomisation.
- The expert reads full simulator state (plank quaternion, roller position). It
  is not a hardware-ready 61-obs ONNX policy.
- `--sweep` and `--speed-sweep` are coarse grids, not optimisations. The
  boundaries quoted are where these particular settings failed.
- The motion scorer counts a "cycle" by a fixed threshold, so heavily skewed or
  slow-lingering runs can score fewer cycles while still balancing the whole
  time. Read `balance_success` separately from `completed_rocking_cycles`.

---

# 7. Varied patterns, and a routine that ends in a fall

**2026-09-05, second follow-up.** A user observation on the wide-rocking plot:
the waveform is skewed and lopsided, but **every cycle is identical**. It is one
shape repeating. So the four motion parameters were made functions of time.

Script: `src/microduck_lab/tasks/balance_board/pattern.py`. Nothing earlier was modified.

## 7.1 How

    theta(t)  = INTEGRAL of 2*pi*f(t) dt
    theta_eff = theta + skew(t) * sin(theta)
    amp       = A(t) * (1 + bias(t) * cos(theta_eff))

The phase must be **integrated**, not evaluated as `2*pi*f*t`. With `f` varying,
the direct formula jumps the phase every time the rate changes, and the board
cannot follow a phase step. This was the only real implementation subtlety.

The wave is **not** re-solved per rate. The feedback already absorbs large rate
deviations — `skew = 0.9` alone swings the instantaneous rate by +/-90% and it
holds — so one 1.2 Hz wave covers the whole routine.

## 7.2 Three programmes

| `--program` | what it does | result |
|---|---|---|
| `choreography` | six named moves, cross-faded over 1 s | held 34/34 s |
| `wander` | every parameter a sum of sines with incommensurate periods, so it never repeats | held 30/30 s on seeds 0-4 |
| `finale` | the choreography **plus one extra move** past the limit | rides 35.3 s, then falls |

The choreography, each move inside the measured speed/swing envelope:

| move | rate | swing | skew | bias |
|---|---|---|---|---|
| slow and wide | 1.0 Hz | 12 deg | 0.9 | 0.20 |
| quick and small | 2.0 Hz | 6 deg | 0.3 | 0.00 |
| long lean, one way | 0.9 Hz | 8 deg | 0.9 | 0.55 |
| build up | 1.4 Hz | 9 deg | 0.6 | 0.10 |
| flutter | 3.0 Hz | 4 deg | 0.2 | 0.00 |
| slow and wide again | 1.1 Hz | 11 deg | 0.9 | 0.25 |

## 7.3 The deliberate fall

`--program finale` is the choreography with exactly one difference: a seventh
move, `("too far", 8 s, 1.2 Hz, 18 deg, skew 0.9, bias 0.30)`. Amplitude 12 is
the largest that holds at 1.2 Hz; 18 is well past it.

    finale: held 35.29 s, board roll peak +25.6 / -11.4 deg
    fell at 35.29 s (foot left plank for more than 0.10 s)

It rides all six moves, reaches +25.6 deg on the seventh, and the **foot lifts
off the plank** — the same failure mode as every other range limit here, and the
same root cause as the one-legged pose: no ankle-roll joint, so the sole tilts
with the board and past ~20 deg it is on its edge.

`Routine(loop=False)` holds the last move instead of wrapping round, which is
what a routine ending in a fall needs.

**Filming the whole fall.** `lqr.evaluate` breaks out of its loop the instant a
failure trips, which is right for scoring but ends a recording at the moment the
fall STARTS — the duck is still upright in the last frame. `--show-fall` uses
`render_including_the_fall()`, which records the failure time and keeps stepping
for `--after-fall` more seconds. Verified: the trunk ends at 47 mm, flat on the
floor with the plank on top of it. That path is for RENDERING only; every
measured number elsewhere still comes from `lqr.evaluate`.

## 7.4 Files

    src/microduck_lab/tasks/balance_board/pattern.py
    videos/balance_board/board_pattern_choreo.mp4          six moves, holds
    videos/balance_board/board_pattern_wander.mp4          never repeats, holds
    videos/balance_board/board_pattern_finale_fall.mp4     same six moves, then falls
    videos/balance_board/board_pattern_motion.png          the three roll traces compared

```bash
uv run python src/microduck_lab/tasks/balance_board/pattern.py --program choreography --seconds 34
uv run python src/microduck_lab/tasks/balance_board/pattern.py --program wander --seed 2 --seconds 34
uv run python src/microduck_lab/tasks/balance_board/pattern.py --program finale --seconds 40
```
