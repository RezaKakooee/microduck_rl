# Open problem: make the Microduck balance on an unstable support

**Latest clarification and result:** the duck should **actively play with the
board**, continuously rocking left/right while balancing. The new
[`task_balance_board_rocking.py`](../src/microduck_lab/tasks/balance_board/rocking.py)
delivers about ±6.7° at 1.2 Hz, including a 60 s hold with sustained motion and
17/17 motion-validation episodes. [Watch the new video](../videos/balance_board/board_rocking.mp4)
and read the [active-play report](tasks/balance_board_rocking.md).
A still hold or a single push response no longer counts as task success.

**Extended 2026-09-05 (asymmetric / faster / wider).** The symmetric rocking is
a pure sine and is left/right symmetric by construction, which is part of why it
holds so easily. `src/microduck_lab/tasks/balance_board/rocking_asym.py` breaks that:
one side can swing 57% further than the other; the rate goes to 2.0 Hz at the
original swing or 3.0 Hz at a smaller one; and the range reaches 31.5° (+19.5 /
−12.0) at 1.2 Hz, against 13.4° originally — all holding 30 s. The two ceilings
are mechanical, not control: a **leg touching the plank** limits speed, and a
**foot lifting off the plank past ~20° of tilt** limits range (no ankle-roll
joint, so the sole tilts with the plank). Report:
[balance_board_rocking_asym.md](tasks/balance_board_rocking_asym.md). A third step made the motion
**non-repeating**: `src/microduck_lab/tasks/balance_board/pattern.py` runs either a
six-move choreography, a never-repeating "wander", or a `finale` that is the
choreography plus one move past the limit so the duck falls at 35.3 s on
purpose. Videos: `board_pattern_choreo.mp4`, `board_pattern_wander.mp4`,
`board_pattern_finale_fall.mp4`.

**Update 2026-09-05: SOLVED IN SIMULATION.** A new whole-body LQR expert holds
the original free roller for 60 s and passes 33/33 specified stress episodes.
No geometry or robot changes were needed. See [the solution report](tasks/balance_board_lqr.md)
and `src/microduck_lab/tasks/balance_board/lqr.py`; video: `videos/balance_board/board_lqr_push.mp4`.
This expert uses full simulator state and is not a hardware-ready ONNX policy.
The original problem statement and measurements below are retained as history.

Written 2026-09-05. Read this first in a new chat, then `HANDOFF.md`, then
`docs/project_journey/` (six files, the whole story with the numbers).

---

## 1. The task

Make the duck hold its balance on something that does not hold still:

- **Rocker board.** A plank resting on a loose cylinder, like a child on a
  wobble board. The duck stands on the plank with both feet. Success = the
  plank never touches the floor for 10 s. **Best so far: 1.8 s.**
- **Related, already closed as impossible:** stand on ONE leg (a
  figure-skating spiral). See section 5.

Everything else we tried is done and works or is closed. This is the one open
problem.

## 2. Where the code is

| Path | What |
|---|---|
| `src/microduck_lab/tasks/balance_board/board.py` | scene builder + `Board` class + `run()` + baseline PD |
| `src/microduck_lab/tasks/balance_board/expert.py` | the expert controller (open stance + differential leg length) |
| `docs/tasks/balance_board.md` | full report, both controllers |
| `videos/balance_board/board_expert_narrow.mp4` | best run, 1.8 s |
| `videos/balance_board/board_expert_open.mp4` | opened stance, 1.3 s |
| `videos/balance_board/board_A_open_fail.mp4` | no feedback at all, 0.0 s |

Run it:

```bash
cd $REPO
uv run python src/microduck_lab/tasks/balance_board/expert.py --open 0.0 --kp 1 --kd 0.4 --sign -1
uv run python src/microduck_lab/tasks/balance_board/expert.py --grid          # 108-point search
uv run python src/microduck_lab/tasks/balance_board/board.py --orientation A --controller pd
```

`run()` takes any callable `controller(state, dt) -> (14 joint targets, u)`.
Write a new controller, pass it to `tb.run("A", ctrl, seconds=10.0)`. Do not
edit the shared files.

## 3. What has been tried, and the numbers

Ten-second runs, orientation A (board tilts side to side), free roller r=30 mm,
plank 300 x 120 x 12 mm.

| Controller | Held (s) |
|---|---|
| Open loop, servos hold the home pose | 0.00 |
| Upright PD only (no board feedback) | 1.66 |
| Board PD: lean sideways with both hip_rolls | 1.48 |
| **Expert: differential leg length, normal stance** | **1.80** |
| Expert: differential leg length, opened stance | 1.26 |
| Expert + lean, both channels together | 1.24 |

Searches run: ~900 rollouts over both signs of every gain, roller radii to
80 mm, a floor-fixed roller; then a 108-point grid over stance/sign/kp/kd and a
focused search over both channels with a doubled command cap. Nothing beat
1.80 s.

## 4. The physics, measured

These are facts about this robot. Trust them; they cost a lot to find.

**The robot has no ankle-roll joint.** Five leg joints: hip_yaw, hip_roll,
hip_pitch, knee, ankle_pitch. This is the single most important fact.

**hip_roll range is only ±0.384 rad (±22°).** It is the binding limit on
anything sideways.

**Leaning does not move the support.** The ankle axis sits 35 mm above the
roller contact, and the roller is 30 mm in radius. Tilting the plank does not
bring the support under the body. A two-body model reproduces the simulator's
unstable pole of **+6.5 /s** and needs ~140 rad/rad of trunk gain, i.e. 2.4 rad
of ankle per degree of lean. The servos cannot do that.

**The plank is light (70 g) and re-angles instantly** under any joint torque,
so the only remaining channel is rolling the roller away, which is
non-minimum-phase (it moves the wrong way first).

**Differential leg length works, but is small.** All three sagittal joints tilt
the sole equally, `d(sole_pitch)/dq = (+0.996, -0.996, +0.996)` for
(hip_pitch, knee, ankle), so keeping the sole flat requires

    hip_pitch - knee + ankle = 0

Maximising leg-length change inside that constraint gives

    SHORTEN_LEFT  = (0.521, 0.805, 0.284)     16.9 mm per rad, sole tilt 0.000
    SHORTEN_RIGHT = -(0.520, 0.805, 0.285)

Verified: 0.5 rad shortens the leg 9.9 mm with 0.00° of sole tilt.

**A wider stance makes it worse, and this is the interesting part.** Opening
both hip_rolls outward:

| open (rad) | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|
| foot separation (mm) | 83.6 | 103.3 | 122.3 | 140.4 | 157.4 |
| sole tilt (deg) | 0.0 | 5.7 | 11.5 | 17.2 | 22.9 |

The wide-stance intuition ("longer lever, more torque") holds when the limit is
FORCE. Here the limit is TRAVEL: the command saturates at about 14 mm of leg
length, and the foot line tilts by `atan(2*dL / separation)`:

| stance | separation | foot-line tilt at dL = 14 mm |
|---|---|---|
| home | 83.6 mm | **18°** |
| opened 0.30 | 140.4 mm | 11° |

Widening divides the correction angle by the same factor it multiplies the
lever. Narrow beats wide for this robot. It would reverse if the servos were
torque-limited rather than travel-limited.

An end of the plank hits the floor at **24°** of tilt, so 18° of authority is
not enough even before the roller dynamics.

## 5. Closed as impossible: one-legged balance

Do not retry this without changing the robot. Six RL runs and a scripted
controller panel all failed. The reason:

With no ankle-roll joint, leaning the hip to put the body over one foot rolls
the SOLE with it. The foot then rests on its outer edge only. Measured:

    support foot mesh spans   y = -34.1 .. +6.3 mm
    centre of mass            y = +2.0 mm
    actual floor contact      ONE point at y = -26.6 mm

The centre of mass ends up 28.6 mm outside the only contact line. That is a
knife edge. The best any controller managed was 0.96 s. Full story in
`docs/project_journey/05_scripted_expert_and_physics.md`.

The head does not rescue it either: the head is 38% of the mass, but its roll
axis passes ~3 mm from its own centre of mass, so head_roll moves the whole-body
CoM only **1.2 mm per radian**. As a reaction wheel it is ~100x too weak.

## 6. Ideas not yet tried

1. **RL instead of a scripted controller.** Everything above is hand-written.
   A policy could use the whole body and find a dynamic strategy a PD cannot.
   The task registration pattern is in `src/mjlab_microduck/tasks/__init__.py`;
   `AGENTS.md` has the repo's reward-design rules, which are good and hard-won.
2. **Let it step.** Every attempt so far keeps both feet planted. A person on a
   wobble board shuffles. Allowing foot repositioning changes the problem.
3. **Bigger roller.** Radii to 80 mm were tried with the old controller but not
   with differential leg length. A larger radius lowers the instability.
4. **Change the robot.** An ankle-roll joint, or hip_roll range past ~0.6 rad,
   would make both this and the one-legged pose feasible. This is the honest
   answer for the hardware, not the software.

## 7. Rules that saved us, and traps

- **Never trust a reward curve or a metric. Watch the video.** Several times a
  metric said success while the duck was doing something else entirely
  (a walk that scored as a one-legged glide; a policy marching in place with a
  healthy air-time reward).
- **Measure the thing that separates success from the nearest cheat.** For the
  spiral that was hold duration on the *same* foot, not single-support percent.
- **Compute the reward landscape before training.** One numpy line would have
  saved a 5-hour run where standing still scored 24% of a perfect pose.
- **`lstsq` minimum-norm solutions can zero your best actuator.** For the leg
  direction it zeroed the knee and gave 2.8 mm/rad instead of 17.
  Ignore `EGLError` tracebacks that print after a successful render.
- Rough-terrain training needs a 20 GB card
  (RTX A4500), not the 10 GB RTX 3080.
