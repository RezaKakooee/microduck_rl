# Balance beam — walk a narrow raised beam without stepping off

**Date:** 2026-09-04.
**Script:** `src/microduck_lab/tasks/walking/balance_beam.py`.
**Scene:** `src/microduck_lab/models/scene_beam.xml` (template; the
script rewrites the beam line into `scene_beam_w<W>.xml` for each width).
**Policy:** the pretrained `alpha_walking.onnx`, unchanged. The script adds
only a steering loop on top of it.

## Result in three lines

- From a centred start the duck walks the full 2 m beam down to **115 mm**
  wide. **110 mm** is the first width that fails (steps off after 0.23 m).
- With the start shifted -10..+10 mm off the centreline, **120 mm** is the
  narrowest width that passes every trial (5/5). 115 mm passes 3/5.
- **110 mm and narrower never reached the end** in 11 trials each. The limit
  is set by where the policy puts its feet (43-48 mm out from the trunk) plus
  the trunk's per-step sway (+/-12 mm), which adds to about 60 mm each side.

## Setup

- Beam: a box, 2.0 m long, 40 mm high, along +x, from x = -0.30 to +1.70 m.
  The floor stays under it. Beam friction is the floor's default (1.0), so
  only the width differs from flat ground.
- The duck starts at x = 0 on the centreline, heading +x, standing on the
  beam. Its trunk height comes from a binary search for the lowest
  contact-free height (117.2 mm above the beam top), plus 2 mm of clearance.
  It stands still for 1 s, then walks with forward command 0.3 (the policy's
  maximum; it actually walks 0.125 m/s: 1.60 m in 12.8 s).
- Finish line at x = +1.60 m, 100 mm before the far end. Crossing it is
  SUCCESS. The run stops there.
- A step-off is any of: a foot geom touches the `floor` geom (not the beam);
  any other robot geom touches the floor or the beam; the trunk drops below
  70 mm above the beam top. The script records which foot, which side, the
  foot centre, and the trunk's lateral error and heading at that moment.
- All numbers come from the simulator state. The sim is deterministic: the
  same width and start give the same run every time.

## Stance, measured

In the home pose, via `duck_sim.make_policy`:

| | |
|---|---|
| foot centres (`left_foot` / `right_foot` sites) | y = +/-41.8 mm, so **83.6 mm apart** (the brief said ~100 mm; it is 84) |
| sole footprint | 41.2 mm wide, 54 mm long; spans 21.2 to 62.4 mm from the centreline |
| outer edge to outer edge | **124.8 mm** |
| gap between the soles | 42.4 mm |
| trunk rest height | 117.2 mm above the surface (duck_sim starts it at 125 mm on the floor) |

So a 125 mm beam fits the whole standing stance. At 120 mm each foot
overhangs its edge by 2.4 mm; at 100 mm by 12.4 mm; at 84 mm the foot centres
sit on the edges; below 42 mm the feet would not touch the beam at all.

## Steering

The policy cannot strafe (a lateral command gives zero lateral motion), so a
lateral error can only be removed by aiming at the centreline. P on lateral
error sets a desired heading, P on heading error sets the yaw-rate command.
The forward command is constant.

```
aim  = clip(-8 * y, +/-0.15 rad)         y = trunk offset from the centreline, +left
turn = clip(4 * (aim - yaw), +/-1.0 rad/s)
command = (0.3, 0, turn)
```

Measured turn response while walking at 0.3 (mean yaw rate over 2-8 s):

| turn command (rad/s) | 0 | +0.10 | -0.10 | +0.25 | -0.25 | +0.50 | -0.50 |
|---|---|---|---|---|---|---|---|
| yaw rate (rad/s) | -0.037 | +0.010 | -0.086 | +0.120 | -0.197 | +0.316 | -0.371 |

About 0.6 x the command, minus a 0.04 rad/s built-in right-hand veer. There is
no deadband on yaw, unlike forward speed. With zero turn command the duck
drifts 97 mm to the right in the first 0.61 m.

The gains come from a grid: k_lat 4/6/8, k_yaw 2/4, max_aim 0.10/0.15/0.20,
max_turn 0.3/0.5/1.0, and a 0 or 1 s steering delay after the walk starts.
108 configurations, each run at widths 125/120/115/110 mm from 5 start
offsets. Best: k_lat 8, k_yaw 4, max_aim 0.15, max_turn 1.0, no delay. The
max_aim value made no difference; the steering delay made things worse; k_lat
4 / k_yaw 2 was clearly worse (2/5 at 120 mm). A constant bias to cancel the
veer was also tried and made things worse. Note the gains were tuned on the
same trials as the table below; the held-out check further down is the
control.

## Width table

Speed 0.3, beam 40 mm high, 2.0 m long, finish at 1.60 m. Two protocols:
one run from a centred start (the task as set), and five runs from starts
offset -10, -5, 0, +5, +10 mm off the centreline.

| width (mm) | overhang per foot (mm) | centred start: distance (m) | stepped off at (s) | centred result | passes, 5 offsets | how it failed |
|---|---|---|---|---|---|---|
| 300 | 0 | 1.60 | - | SUCCESS | 5/5 | - |
| 200 | 0 | 1.60 | - | SUCCESS | 5/5 | - |
| 150 | 0 | 1.60 | - | SUCCESS | 5/5 | - |
| 130 | 0 | 1.60 | - | SUCCESS | 4/5 | +10 mm start: left foot over the left edge at 2.3 s, 0.12 m |
| 125 | 0 | 1.60 | - | SUCCESS | 4/5 | +10 mm start: left foot over the left edge at 2.4 s, 0.13 m |
| **120** | 2.4 | 1.60 | - | SUCCESS | **5/5** | - |
| **115** | 4.9 | 1.60 | - | **SUCCESS** | 3/5 | +5 and +10 mm starts: left foot over the left edge at 2.0-2.1 s |
| 110 | 7.4 | 0.23 | 3.1 | FAIL | 0/5 | centred: right foot over the right edge; others 0.08-1.41 m |
| 100 | 12.4 | 0.14 | 2.4 | FAIL | 0/5 | right foot over the right edge (centred); all within 0.3 m |
| 90 | 17.4 | 0.09 | 1.8 | FAIL | 0/5 | left foot over the left edge; all within 0.1 m |
| 80 | 22.4 | 0.06 | 1.8 | FAIL | 0/5 | left foot over the left edge; all within 0.1 m |
| 60 | 32.4 | 0.02 | 1.4 | FAIL | 0/5 | tips off while still standing (0.2-1.4 s) |

Max lateral error of the trunk on successful runs: 17-18 mm from a centred
start, up to 31 mm with the offset starts (that includes the offset itself).
The 5-offset column is not monotonic: 125 and 130 mm each lose the +10 mm
trial while 120 mm wins all five. That is how sensitive the outcome is to
the first steps (see below), and why the boundary is a band, not a line.

**Held-out check** (start offsets -12, -7, -3, +3, +7, +12 mm; six per
width, same gains): 130 mm 5/6, 125 mm 5/6, 120 mm 4/6, 115 mm 4/6,
110 mm 0/6. Same picture, slightly lower pass rates.

**Narrowest width that reached the end:** 115 mm from a centred start;
120 mm if it has to survive +/-10 mm start offsets. 110 mm and narrower:
0 of 11 trials.

## What failure looks like

Every failure is a foot tipping over an edge, then dropping the 40 mm to the
floor. Two patterns.

**Early, in the first steps (t = 1.4-3.5 s, within 0.3 m).** The policy's
first steps out of a standstill are the widest of the whole walk.
110 mm, centred start: the first right-foot steps land at -51, -54, -58 mm
(edge at -55). The trunk is carried right (-20, -32 mm), the steering asks
for a left turn (+0.5 to +0.65), but the next right step lands at -72 mm,
tips, and at 3.1 s the right foot is on the floor at -84 mm. Trunk lateral
error at the last upright moment: -67 mm, heading +7.5 deg.
With a +10 mm start (130, 125 and 115 mm all fail the same way): the left
foot already starts at +52 mm. Its steps land at +62, +64, +70, +74, +77 mm,
each further out; the trunk follows (+16, +32, +47 mm); the foot drops off the
left edge at 2.3 s with its centre at +96 mm (edge +65). The steering was
turning right the whole time (-0.3 to -0.7) and the heading did go right
(-8 to -12 deg), but the foot placement ran away first.

**Late, after a long clean walk (110 mm, +10 mm start).** 1.41 m in 11.2 s
with the trunk within +/-14 mm of the centreline. At 11.5 s the left foot
landed at +57 mm, 2 mm past the +55 mm edge. It tipped, the trunk went to
+38 mm in half a second, the next left step landed at +71 mm, and the foot was
on the floor at 12.2 s. The biggest lateral error the steering saw before the
tipping step was +14 mm.

Which foot and which side: from a centred start it is the right foot over
the right edge (110 and 100 mm), which is the policy's built-in right veer.
From a left-shifted start it is the left foot over the left edge. The
trunk's lateral error when the foot reaches the floor is 84-140 mm; at the
last moment the trunk is still at walking height it is 52-67 mm; the last
value the steering could act on before the tipping step is only 10-25 mm.

## Why the limit is about 120 mm

Measured on the flat floor at command 0.3:

- The feet land **43 mm** to each side of the trunk (range 37-48 mm,
  in the trunk's own frame). This does not change with the turn command:
  -1.0 to +1.0 rad/s moves the mean landing point by under 3 mm. The policy
  does not step wider or narrower to turn; steering only moves the trunk.
- The trunk sways **+/-11.5 mm** per step about its mean path (7.8 mm rms).
- So the outermost foot centre sits about 48 + 12 = 60 mm off the mean path.
  A foot whose centre is past the edge tips; the ankle roll servo does not
  hold a cantilevered foot. Two sides gives 120 mm, the same number the sweep
  found. Any lateral steering error eats into that margin directly.

The steering removes the slow drift (the 0.04 rad/s veer) but cannot remove
the sway, and it cannot act within one step: a foot that lands 2 mm over the
edge is already the end.

## Videos

Rendered on a debug GPU node (`MUJOCO_GL=egl`), three views each:
`_az130` (three-quarter, from behind-left), `_az90` (side, from the right),
`_az180` (front, looking back at the duck).

| file | what |
|---|---|
| `videos/beam_w115_success_az{130,90,180}.mp4` | narrowest success: 115 mm, centred start, 1.60 m in 12.8 s |
| `videos/beam_w110_fail_az{130,90,180}.mp4` | first failure: 110 mm, centred start, right foot off the right edge at 3.1 s |
| `videos/beam_w110_fail_late_az{130,90,180}.mp4` | 110 mm, +10 mm start: 1.41 m clean, then the left foot lands 2 mm over the edge |
| `videos/balance_beam/beam_w300_reference.mp4` | 300 mm, for comparison |

The clips play at 1.2x real time: `duck_sim.Recorder` captures every second
control step (25 Hz) and writes at 30 fps. Every clip in `videos/` shares
this. Sim time = video time x 1.2; the 3.1 s step-off is at 2.6 s in the clip.

## Run it

```bash
cd $REPO
uv run python -m microduck_lab.tasks.walking.balance_beam --width 0.115               # one run, prints the summary
uv run python -m microduck_lab.tasks.walking.balance_beam --sweep 0.30 0.20 0.15 0.13 0.125 0.12 0.115 0.11 0.10 0.09 0.08 0.06 \
    --y0 -0.01 -0.005 0 0.005 0.01                               # the table above, ~2 s per run
# video, on a GPU node:
MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg src/microduck_lab/tasks/walking/balance_beam.py --width 0.115 --video videos/balance_beam/beam_w115.mp4 --azimuth 130 90 180
```

Options: `--width`, `--length` (2.0), `--height` (0.04), `--speed` (0.3),
`--seconds` (25), `--y0` (start offsets, one or more), `--k-lat` (8),
`--k-yaw` (4), `--max-aim` (0.15), `--max-turn` (1.0), `--turn-bias` (0),
`--steer-delay` (0), `--runway` (0; length of a 0.30 m wide start platform),
`--sweep W...`, `--video`, `--cam-distance` (0.9), `--azimuth` (one or more),
`--log-every` (2 s). `--walking` defaults to
`$POLICIES/alpha_walking.onnx`.

## Honest limits

- One deterministic simulator, no observation or actuator noise. The
  +/-10 mm start offsets are the only perturbation. The pass/fail boundary
  is a band from 115 to 130 mm where single trials flip on mm-level starts.
- The steering gains were chosen on the same widths and offsets as the table.
  The held-out offsets gave the same ranking with slightly lower pass rates.
- Slower did not help. Commanded 0.25 and 0.27 needed a 150 mm beam, and
  below 0.25 the policy does not move at all. There is no "careful" speed.
- Starting from a standstill on the narrow beam is the main early failure,
  but a 0.30 m wide start platform (`--runway 0.8`) did not rescue 100-110 mm:
  they still stepped off 60-90 mm into the narrow part.
- The edge behaviour (a foot tipping over a sharp box edge) is MuJoCo's
  mesh-box contact with the default solver settings. It has not been checked
  against hardware, and the real foot rubber may grip an edge differently.
- The finish line is 100 mm before the far end, so the last two foot lengths
  are never walked. "Distance walked" is the trunk's x at the step-off, not
  the last foot's position.
- Every run writes `scene_beam_w<W>.xml` next to the template. They are
  generated files; the template is the one to edit.
