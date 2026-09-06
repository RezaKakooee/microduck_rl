# Rocker balance board — historical PD experiments

**Latest demo:** [continuous active board play](balance_board_rocking.md), with
visible left/right rocking throughout the rollout.

**Update:** orientation A is now solved in simulation by a separate
[whole-body LQR expert](balance_board_lqr.md): 60 s nominal, 33/33 specified
validation episodes. The PD results below are historical.

**Date:** 2026-09-05.
**Script:** `src/microduck_lab/tasks/balance_board/board.py`.
**Scene:** `src/microduck_lab/models/scene_board.xml` (template; the
script writes `scene_board_<tag>.xml` per configuration and the robot variant
`robot_allcollisions_boardfeet.xml`, see "Contact modelling").
**Controller:** scripted, no policy. Home pose + an upright PD + a board PD.

## Result in four lines

- Target was 10 s. Best hold on the free board: **A (roll) 2.22 s, B (pitch)
  0.86 s.** No gain set in ~900 rollouts came close, in either sign.
- The bare servos ("hold the home pose, no feedback") do not stand at all: the
  duck folds forward in 0.8 s even on the flat floor. Every board run of that
  baseline is 0.00 s.
- The board is a "rola bola" (plank on a FREE roller). It is unstable for any
  load and any roller radius, so the "make it less stable" clause never applied.
- The reason is geometry, not gains: the ankle axis is 35 mm above the contact
  and the roller radius is 30 mm, so tilting the plank does not move the
  support under the body. A linear model says the needed trunk gain is ~140
  rad/rad; the ankle would saturate at a 0.6 deg lean.

## Geometry

| | |
|---|---|
| roller | radius 30 mm, 180 mm long, 60 g, axis on the floor, friction 1.5, free (6 dof) |
| plank | 300 x 120 x 12 mm, 70 g, friction 1.5, free (6 dof), orange |
| A | plank long axis along world y (the duck's lateral axis), roller axis along x: the board tilts in ROLL |
| B | plank long axis along x (forward), roller axis along y: the board tilts in PITCH |
| plank top | 72 mm above the floor; an end touches the floor at 23.6 deg of tilt |
| duck | home pose, trunk 115.6 mm above the plank top, CoM 126 mm above the plank top (138 mm above the contact line), 0.737 kg |
| foot contact patch | 45.6 x 34.0 mm per foot (the flat part of the sole hull), centres 83.8 mm apart |

The roller geom is a **capsule** of radius 30 mm. The plank (+/-60 mm) rests on
its straight part (+/-90 mm), so it rolls exactly like a cylinder. Why not a
cylinder: see "Contact modelling".

## Setup procedure (what the script does)

1. Write the scene for the asked orientation and sizes.
2. Settle the roller and plank on the floor for 1.5 s with the duck parked in
   the air.
3. Rest the duck on the plank: binary search on the trunk height for the
   lowest height with no robot contact (plank/roller contacts are ignored),
   plus 1 mm. Shift the duck so the combined CoM (duck + plank) is over the
   roller axis (a 0.0 mm shift in A, +7 mm in B).
4. Settle 1.0 s with the board HELD level and the upright PD on. "Held" =
   the roller and plank are made 1e4 x heavier and reset to their settled
   pose every physics step. (Resetting alone is not enough: a 70 g plank
   accelerates away under a 7 N load inside one step and the duck sinks
   through it. `mj_setConst` must be re-run after the mass change, on a
   scratch `MjData`, or it overwrites the live `qpos`.)
5. Release the board. t = 0. The board controller starts here.

Failure = any plank-floor contact, a foot with no plank contact for 0.1 s,
any non-foot robot body touching floor/plank/roller, or the trunk below
70 mm above the plank. Seconds held = time to the first failure.

## Controllers

Joint order: 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5-8
head, 9-13 right leg. Angles from projected gravity: roll = atan2(g_y, -g_z),
pitch = atan2(g_x, -g_z), for the trunk and for the plank (from its body
quaternion). tilt > 0 = the +axis side of the plank is down (left down in A,
front down in B). `s` = roller offset from the plank centre along the plank.

**open** — `ctrl = HOME`. No feedback.

**upright** — HOME plus a PD on the trunk's projected gravity, both axes
(the `two_leg_glide.py` structure):

```
ankles   (L +v, R -v)   v = 5.0 * pitch + 0.3 * d(pitch)/dt
hip_rolls (both -w)     w = 2.0 * roll  + 0.1 * d(roll)/dt
```

Gains from a grid on the held (static) board: pitch (5, 0.3) sags 0.9 deg
and holds 5 s; (8, <=0.1) oscillates. Roll is stable for every gain 0-8.

**pd** — upright plus board terms on the tilt axis:

```
u = (kp_s + kp_t) * trunk_tilt + (kd_s + kd_t) * d(trunk_tilt)/dt
    + kp_b * plank_tilt + kd_b * d(plank_tilt)/dt - ks * s - kv * ds/dt
A: hip_rolls -= u      B: L_ANKLE += u, R_ANKLE -= u
```

u > 0 always means "move the CoM toward the -axis". While the board is held
the plain upright PD runs (a board law with negative trunk gain falls off a
fixed floor).

Best gains (grid, see below):

| | kp_t | kd_t | kp_b | kd_b | ks | kv | held |
|---|---|---|---|---|---|---|---|
| A | -4.0 | -0.7 | 0 | 0 | 0 | 0 | 2.22 s |
| B | -5.0 | -0.3 | 0 | 0 | -10 | 0 | 0.86 s |

Negative kp_t means less trunk feedback than the upright PD (A: total roll
gain -2; B: total pitch gain 0). Nothing with plank-tilt feedback did better.

### A sign trap: hip roll with both feet planted

The one-leg harness moves the body over the right foot with +0.384 on BOTH
hip_rolls. With both feet planted it is the opposite: +0.15 rad on both hips
rolls the TRUNK 5.5 deg to the LEFT and moves the CoM 2 mm left (measured:
15 mm/rad). The legs stay vertical on the planted feet and the trunk + head
swing about the hips. The first version of the roll PD used the harness sign
and ran away to a 12 deg lean at kp >= 3. The script maps hip_rolls -= u.

Ankles (L +, R -): the trunk pitches nose-up, CoM -140 mm/rad at 0.3 s
(measured as the difference between an offset and a no-offset run with the
pitch PD off, since the duck falls in 0.8 s without it).

## Results (free roller, r = 30 mm, 10 s runs, push = 0.15 m/s on the trunk at t = 2 s)

| orientation | controller | seconds held (of 10) | max plank tilt while held | with the push | how it ended |
|---|---|---|---|---|---|
| A | open | 0.00 | - | 0.00 | folds forward on the held board; beak on the floor at release |
| A | upright | 1.66 | 21.0 deg | 1.66 | plank end on the floor, -y side, roller 14 mm off centre |
| A | **pd** | **2.22** | 20.6 deg | 2.18 | plank end on the floor, +y side, roller 15 mm off centre |
| B | open | 0.00 | - | 0.00 | folds forward on the held board |
| B | upright | 0.60 | 17.6 deg | 0.60 | plank end on the floor, -x side |
| B | **pd** | **0.86** | 17.9 deg | 0.86 | plank end on the floor, -x side, roller 15 mm off centre |

No run reached t = 2 s, so the push never landed; "with the push" repeats the
unpushed number. The sim is deterministic: one run per cell.

**Which orientation is achievable: neither.** A lasts longer only because
the roll axis starts perfectly symmetric and the instability needs ~1.5 s to
grow from numerical noise; once it grows the hips (15 mm/rad, +/-0.3 rad =
+/-4.5 mm of CoM) cannot touch it. B tips within 0.2 s of any action.

### Gain search

Six grids, ~120-320 rollouts each, both orientations: plank gains kp_b in
[-4, 8], kd_b in [-0.3, 1.0]; trunk total gain in [-8, +9] with D in
[-1.5, 0.6]; roller gains ks in [-40, 40], kv in [-2, 2]; roller radius 45,
60, 80 mm. Spread of all A results: 1.6-2.2 s. Spread of all B results:
0.2-1.1 s (the 1.1 s was at r = 80 mm). Every row is in the run logs of
`--grid A` / `--grid B`.

### Open loop and the passive board

- Bare servos on the flat floor: pitch 1.3 deg at 0.1 s, 6.5 deg at 0.3 s,
  22.6 deg at 0.6 s, beak down at 1.0 s. The identified XL330 position gain
  (0.55 N m/rad per servo) is softer than gravity's 0.94 N m/rad about the
  ankles, and the head's weight sags the neck. So "open loop" is not a
  standing controller on this robot, and the upright PD is the real
  no-board-feedback baseline.
- Plank + a 0.737 kg dead weight, no duck, 0.1 deg start tilt: with the
  FREE roller the plank end is on the floor at 0.80-0.88 s for every load
  height from 15 to 138 mm. With the roller PINNED to the floor it is
  stable only with the load CoM at 15 mm (25 mm tips at 1.6 s): the rocker
  rule h < r. So there is no radius or thickness at which open loop holds;
  the "make the board less stable" clause never applied.

## What failure looks like

- **A, pd (best):** the board sits level for 1.3 s (tilt < 0.5 deg). The
  tilt then doubles every ~0.1 s; the roller rolls 15 mm toward the rising
  side (the plank moves twice as far as the roller, so the contact line
  slides toward the high end); the hips saturate at 0.30 rad; the +y plank
  end hits the floor at 2.2 s with the trunk leaning 12 deg.
- **B, upright:** at 0.1 s the plank is 0.9 deg rear-down, at 0.2 s 3.3 deg.
  The ankle PD keeps the trunk within 2.5 deg the whole time, but the roller
  wanders (+/-4 mm) and the plank rocks -3.5, -2, -6, -19 deg and hits the
  floor at 0.6 s. The duck stays upright on a board that leaves from under it.
- **B, pd with plank-tilt feedback (kp_b = 4):** +1 rad of ankle command
  rotates the 70 g plank 17 deg front-down in 0.16 s while the 700 g body
  pitches 5 deg. The plank re-angles under the ankle torque; the plank
  angle is an actuator output, not a state to servo with high gain.
- **open:** the duck folds forward on the held board during the 1 s settle;
  at release the beak is already on the floor.

## Why it cannot work (measured and modelled)

A two-body planar model — plank + feet (0.13 kg, angle theta) on a massless
roller (x_c), body (0.68 kg, CoM 105 mm above the ankle) on the ankle servo
(1.1 N m/rad for both) — with no-slip rolling (plank centre = 2 x_c, contact
offset s = r theta - x_c):

1. The plank's gravitational stiffness about the contact is ~0: tilting it
   by theta moves the ankle by a*theta (a = 35 mm) and the contact by
   r*theta (r = 30 mm). Net 5 mm/rad. **Tilting the plank does not move the
   support under the body.** The measured 2.22 s (A) and 0.86 s (B) come
   from the roller running away, not from any correction.
2. The only lasting channel is the roller: a front-down plank pushes the
   stack forward with M g theta/2, the plank moves twice as far as the
   roller, and the contact slides BACKWARD under the body. Non-minimum
   phase: the sign that helps now hurts 0.2 s later.
3. Open-loop unstable pole +6.5/s (sim: a doubling every 0.1 s). Trunk
   feedback of any gain -4..+8 rad/rad leaves it at +5.7..+6.0/s. Plank
   feedback of either sign makes it worse. An LQR stabilises the model only
   with ~140 rad/rad on the trunk lean and ~1700 rad/m on the roller offset:
   2.4 rad of ankle per degree of lean. The ankle range is +/-1.5 rad.
4. Roller radius helps slowly in the model (80 rad/rad at r = 100 mm) and
   in the sim (B: 1.1-1.2 s at r = 80 mm). Not a way out.
5. A roller FIXED to the floor (a true wobble board, `--fixed-roller`) does
   not rescue it either: A <= 2.1 s, B <= 1.2 s at r = 30 and 60 mm. With
   a fixed roller the same a ~ r geometry applies; the plank spins freely
   under the ankle torque.

What would change the answer: a roller much larger than the ankle height
(r >> 35 mm) AND a stiff, fast ankle; or a heavy board and roller that give
the ankle something to push against; or feet spread along the tilt axis
(a force couple on the plank instead of a torque on a 0.6 g m^2 plank).
The duck has none of these. One-legged variants were not attempted
(docs/project_journey/05, sections 3-5).

## Contact modelling (three things that had to change)

- **Foot boxes.** The sole is a mesh; mesh-on-box goes through MuJoCo's
  convex collider, which returned 2-4 contacts clustered within ~10 mm of
  the deepest point and let the foot roll forward on a level, held plank
  (10 deg in 0.4 s; the same pose stands on the floor's mesh-plane
  collider). `ensure_robot_variant()` writes `robot_allcollisions_boardfeet.xml`:
  the stock robot plus one box per foot matching the flat patch of the sole
  hull (vertices within 1 mm of the bottom), contype/conaffinity 0. The
  scene collides only those boxes with the plank (explicit `<pair>`s; the
  sole meshes are `<exclude>`d from the plank body). On the floor the sole
  meshes collide as before, so the robot is unchanged there.
- **Capsule roller.** Box-on-cylinder also goes through the convex collider:
  under the duck's load it dropped to 0-1 contacts and the plank sank 21 mm
  into the roller. Box-on-capsule is native: 2 contacts, 0.1 mm dip.
- **Stiff contacts.** MuJoCo's soft contact scales with the LIGHTER body's
  inverse mass. With the defaults the 60 g roller sank 5 mm into the floor
  and the 70 g plank 13 mm into the roller at release. Plank, roller and the
  foot pairs use `solref="0.02 1" solimp="0.99 0.999 0.001"`, priority 1.
  Sink at release: 0.4 mm.

## Videos (debug GPU node, `MUJOCO_GL=egl`)

Each clip shows the 1 s held settle first, then the run from release, then
1 s after the failure. 25 fps, written at the captured rate.

| file | what | camera |
|---|---|---|
| `videos/balance_board/board_A_pd_best.mp4` | A, pd gains: level for 1.3 s, then the +y end goes down at 2.2 s (106 frames) | behind (azimuth 180) |
| `videos/balance_board/board_B_pd_best.mp4` | B, pd gains: the -x end goes down at 0.86 s (72 frames) | right side (azimuth 270) |
| `videos/balance_board/board_A_open_fail.mp4` | A, bare servos: folds forward on the held board (51 frames) | behind |
| `videos/balance_board/board_B_upright_fail.mp4` | B, upright PD only: the trunk stays up, the board rocks away under it (66 frames) | right side |

## Run it

```bash
cd $REPO
uv run python -m microduck_lab.tasks.balance_board.board --orientation A --controller pd            # one run, prints the summary
uv run python -m microduck_lab.tasks.balance_board.board --orientation B --controller open
uv run python -m microduck_lab.tasks.balance_board.board --compare                                   # the table above (12 runs)
uv run python -m microduck_lab.tasks.balance_board.board --grid B                                    # gain search, ~120 rollouts
uv run python -m microduck_lab.tasks.balance_board.board --levers                                    # the sign/lever measurement
uv run python -m microduck_lab.tasks.balance_board.board --stand-grid                                # upright PD gains on the held board
uv run python -m microduck_lab.tasks.balance_board.board --orientation B --radius 0.06 --fixed-roller
# video, on a GPU node:
MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg python src/microduck_lab/tasks/balance_board/board.py --orientation A --video videos/balance_board/board_A.mp4 --azimuth 180
```

Options: `--orientation A|B`, `--controller open|pd` (pd with all board gains
0 is the upright baseline), `--seconds` (10), `--push` (m/s, at
`--push-at`, default 2 s), the six gains `--kp-b --kd-b --kp-t --kd-t --ks
--kv`, `--radius` (0.03), `--plank-len` (0.30), `--plank-thk` (0.012),
`--fixed-roller`, `--video`, `--azimuth` (180 for A, 270 for B by default),
`--cam-distance` (0.8), `--elevation` (-12), `--log-every` (0.5 s).
Constants at the top of the script: roller and plank sizes and masses,
friction, contact parameters, settle time, foot grace, stand gains.

## Honest limits

- One deterministic simulator, no sensor or actuator noise; one run per
  cell. The gain search is a coarse grid, not an optimiser. With ~900
  rollouts spanning both signs of every term and a model that agrees with
  the sim on the unstable rate, a hidden 10 s gain set is unlikely, but not
  excluded.
- The controller reads the roller position and the plank quaternion
  straight from the simulator. A real duck would have neither.
- The upright PD is part of every non-open row; "upright" is the honest
  no-board-feedback baseline, since the bare servos cannot stand.
- The push was never tested: nothing survived to t = 2 s.
- The foot boxes are the flat 45.6 x 34 mm patch; the full sole hull is
  54 x 41 mm with rounded edges. The support polygon on the plank is
  therefore a few mm smaller than on the floor.
- The two-body model is planar, small-angle, and treats the roller as
  massless (60 g vs 810 g). It is used to explain, not to tune.
- The capsule's rounded ends never touch anything here; a longer plank
  than 180 mm along the roller axis would need a longer roller.
- Every run writes `scene_board_<tag>.xml` next to the template, and the
  first run writes `robot_allcollisions_boardfeet.xml`. Generated files;
  delete them to regenerate.

---

# Expert controller: open stance + differential leg length

Added after the first report, to test a specific idea: a person on a wobble
board stands WIDE and presses down alternately with each foot, rather than
leaning. Script: `src/microduck_lab/tasks/balance_board/expert.py`.

## The two things it needed, both measured

**1. Opening the stance.** Both hip_rolls outward from HOME (left +, right -):

| open (rad) | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.47 |
|---|---|---|---|---|---|---|
| foot separation (mm) | 83.6 | 103.3 | 122.3 | 140.4 | 157.4 | 168.6 |
| sole tilt (deg) | 0.0 | 5.7 | 11.5 | 17.2 | 22.9 | 26.9 |

**2. Changing one leg's length with the sole staying flat.** All three sagittal
joints tilt the sole equally, `d(sole_pitch)/dq = (+0.996, -0.996, +0.996)` for
(hip_pitch, knee, ankle), so the sole-flat constraint is

    hip_pitch - knee + ankle = 0

which leaves a 2-D family. Maximising length change inside it gives

    SHORTEN_LEFT  = (0.521, 0.805, 0.284)      16.9 mm of leg per rad
    SHORTEN_RIGHT = -(0.520, 0.805, 0.285)     sole tilt exactly 0

Verified: 0.5 rad shortens the leg 9.9 mm with 0.00 deg of sole tilt.

**Trap:** a minimum-norm `lstsq` for this returns `(-0.707, 0, 0.707)` — it
zeroes the KNEE, the joint carrying most of the length authority, and yields
only 2.8 mm/rad. Six times worse. Maximise inside the null space instead.

## Result: it does not help, and opening the stance makes it WORSE

Ten-second runs, orientation A, free roller r = 30 mm:

| controller | held (s) |
|---|---|
| open loop (no feedback) | 0.00 |
| upright PD only | 1.66 |
| their board PD (lean with both hip_rolls) | 1.48 |
| expert: differential leg length, HOME stance | **1.80** |
| expert: differential leg length, OPENED stance 0.30 | 1.26 |
| expert + lean channel together | 1.24 |

A 108-point grid over open/sign/kp/kd, then a focused search over both channels
with a doubled command cap, never beat 1.80 s. Every best row had `open = 0.0`.

## Why a wider stance is the wrong move here

The intuition — wider stance, longer lever, more torque — is right when the
limit is FORCE. Here the limit is DISPLACEMENT. The command saturates at about
14 mm of leg length, and the angle that tilts the foot line through is

    theta = atan( 2 * dL / separation )

| stance | separation | foot-line tilt at dL = 14 mm |
|---|---|---|
| HOME | 83.6 mm | **18 deg** |
| opened 0.30 | 140.4 mm | 11 deg |

So opening the legs *divides* the available angular correction by the same
factor it multiplies the lever by. With a position-controlled leg of limited
travel, narrow beats wide. It would be the other way round if the servos were
torque-limited rather than travel-limited.

And 18 deg of foot-line authority is still under the 24 deg of plank tilt that
puts an end on the floor, on top of the non-minimum-phase roller dynamics the
first report measured. So the channel is real, it is simply too small.
