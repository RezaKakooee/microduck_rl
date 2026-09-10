# Duck on a playground swing

Video: `videos/swing/duck_swing.mp4` (60 seconds, 1280×720, 25 fps).
The duck starts on a stationary swing. It pumps to about ±22° using its legs
and head. It stays seated the whole time.

`videos/swing/duck_swing_2min.mp4` is a two minute clip that pumps to 53° and
then drops the duck onto the grass. See **Long video** and **Getting off the
swing**.

## Run

From the `microduck_rl` checkout:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.swing.expert
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.swing.expert --suite --report videos/swing/swing_suite.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_swing -v
srun -M cluster -p debug --gres=gpu:1 -c 4 -t 15 bash src/microduck_lab/render/render.sh swing
```

Use `--passive` for the identical setup with constant robot joint targets.
`--target-degrees` sets the amplitude to aim for, `--video-speed` the playback
speed, `--video-label` the text on the frame, and `--dismantle-at` the time the
seat lets go.
`--length`, `--damping`, `--voltage`, and `--initial-angle` change the test
conditions. The default episode starts at zero swing angle and zero velocity.
Only initial placement writes the robot's free-body pose.

## Physical construction

The A-frame is fixed to the ground. Two rigid suspension rods, axle bearings,
and a bucket seat rotate on one **unactuated** hinge. The model represents a
swing with rigid hangers; it does not simulate flexible ropes, rope slack,
lateral sway, or an independently pitching seat.

The robot remains a free body. Solid seat, backrest and lap-restraint geometry
support it through contact. There is no weld attaching it to the swing.
The bearing rings have actual holes and clearance around the axles.
The source model's internal CAD collision filtering is retained.

### The seat is cut to fit the duck

All numbers below are millimetres from the trunk origin, x forward and z up.

The duck's legs hang beside its body. They never come closer to the middle than
|y| = 29. Its head never drops below z = +43. So the seat lives in the gap:
every part is inside |y| <= 26 and z <= +35, where only the trunk can reach.

The trunk's underside is a narrow ridge from x -47 to -21. Forward of that the
belly starts only at z = -5. A flat plate under the whole body cannot hold it.

| part | x range | y range | job |
|---|---|---|---|
| seat plate | -50..-12 | \|y\| <= 26 | under the ridge, carries the weight |
| seat lip | -18..-12 | \|y\| <= 26 | stops the round belly rolling forward |
| back panel | -57..-49 | \|y\| <= 100 | 2 mm behind the trunk |
| lap guard | +38..+44 | \|y\| <= 26 | 3 mm in front of the belly; it never touches |
| rear floor | -57..-44 | \|y\| <= 75 | behind the legs, so it can be wide |
| bucket walls | -57..+44 | \|y\| 75..100 | outside every leg |

Two cross bars at z = +28 carry the frame out to the ropes. That is above the
legs, which stop at +17, and below the head, which stops at +43.

The load-carrying plate is only 38 x 52 mm, and the duck hides it. On its own
the swing looked like it had no seat at all. The bucket walls, the wide back
panel and the rear floor fix that. They use the three spaces the duck never
enters: |y| >= 75, behind x = -44, and below z = +43. The legs come out through
the gaps at |y| 26..75, the way they do on a toddler bucket swing. None of
these parts ever touches the duck.

### Contact matches what you can see

Collision group 3 leaves out the parts that meet the seat. The visible thigh
hung 41 mm below any group-3 shape, and the visible belly reached 49 mm further
forward. The seat then touched a shape nobody can see, so the duck looked sunk
into the plate. The trunk, thighs, shins and feet now collide with their visible
meshes. That is 43 collision geoms instead of 11, at 3 ms per control step.

Three seat versions failed before this one:

| version | what went wrong |
|---|---|
| lap guard inside the belly | it pinned the duck; that is the penetration you could see |
| wide plate under everything | shins landed first, underside stayed 36 mm up |
| tall back panel | head struck it, drove 10 mm in, levered the duck out at 13 s |
| narrow plate only | correct physics, but no seat was visible around the duck |

The original robot geometry and joints are driven by BAM XL330 M6 voltage
actuators: 7.4 V nominal, firmware gain 200, 1.75 A current limit, and
load-dependent voltage sag. The 50 Hz controller supplies position targets;
BAM computes motor torques and friction at the 2 ms physics timestep.
The measured maximum nominal motor torque is about 0.394 Nm, below the
0.641 Nm current limit.

Runtime guards reject externally applied forces, collision/opacity changes,
and position or velocity edits by the motor/controller update. No mocap
bodies or equality welds exist in this scene. Camera and lighting changes
only affect the rendered view.

## Pumping expert

This is an optimized **simulation feedback controller**, not a new PPO/ONNX
policy or a hardware-deployment claim. It observes the swing angle and angular
velocity, computes oscillation phase, and synchronizes head and leg motion to
it. The phase is measured rather than driven by a fixed animation clock.
An amplitude regulator reduces pumping near the target and reverses the
energy input if the swing becomes too large.

### How much each joint can pump

Measured on this model. The first number is how far the centre of mass moves.
The second is how much the moment of inertia about the pivot changes.

| joint | CoM shift per rad | inertia change |
|---|---|---|
| neck_pitch | -28.1 mm | 3.0% over +/-0.6 rad |
| head_pitch | +11.8 mm | |
| hip_pitch | -10.7 mm | 2.1% over +/-0.5 rad |
| knee | +1.2 mm | 4.0% over +/-0.6 rad |
| ankle | -0.1 mm | |

The head and neck weigh 251 g. The legs weigh 116 g. So the head has more
authority per radian. The knee cannot move the centre of mass. It changes the
moment of inertia most of all.

### The knee must move once per swing, not twice

A standing person pumps by changing the moment of inertia twice per swing
cycle. That does not work here. Measured with the head held still:

| knee motion | swing amplitude | servo work |
|---|---|---|
| knee at 1x, with the hips | **8.8 deg** | 14.1 J |
| knee at 2x, with the hips | 0.1 deg | 12.4 J |
| knee at 2x, alone | 0.3 deg | 37.6 J |

At 2x the knee cancels the pumping. The 2x-only case spent 37.6 J and moved
the swing 0.3 deg. So `knee_harmonic=1` is the default.

### Rebalancing so the legs do real work

The first version pumped with the head alone. Joint travel in the last 20
seconds, then and now:

| joint | first version | now |
|---|---|---|
| neck_pitch | 70 deg | 61 deg |
| head_pitch | 37 deg | 36 deg |
| hip_pitch | 7 deg | **83 deg** |
| knee | **0 deg** | **68 deg** |

Both versions reach the same swing amplitude. Settings are `hip=-0.85`,
`knee=0.65`, `neck=0.45`, `head=-0.25`. Peak servo torque is 0.528 Nm against
the 0.641 Nm limit, so the motors are not saturating.

The new seat holds the duck much better, so the legs get more grip. The head
now does less work than before and the legs do more.

A physics sweep compared four timing phases and different hip/head motions.
The successful phase pumps approximately in opposition to swing velocity;
the opposite phase damps motion. Timing and amplitude were then validated
with the complete contact and motor model.

Research informing this approach:

- Hirata et al., [Initial phase and frequency modulations of pumping a
  playground swing](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.107.044203),
  Physical Review E 107, 044203 (2023): body-motion timing adapts to swing phase
  and amplitude.
- Koshkin and Jovanovic, [Swinging a playground swing: torque controls for
  inducing sustained oscillations](https://arxiv.org/abs/2206.09579) (2022):
  feedback control of sustained oscillations in underactuated swing models.

### How high the swing can go

The task target is 22 deg. The duck can go much further. Raising
`target_degrees` and running 240 s each:

| target | reached | at t | overlap |
|---|---|---|---|
| 30 deg | 31.6 deg | 74 s | 0.85 mm |
| 40 deg | 41.7 deg | 120 s | 1.33 mm |
| 50 deg | 51.1 deg | 239 s | 1.87 mm |
| 60, 70, 80 deg | 49.8 deg | - | 1.82 mm |

So about **50 deg** is the ceiling. Above 50 the amplitude stops growing and
peak torque drops to 0.367 Nm, so the motors are not the limit.

It is not a phase-tracking problem either. A pendulum slows as it swings wider,
`T = T0 (1 + theta^2/16)`, and the expert uses the fixed small-angle rate.
Correcting that rate for amplitude changed the result by 0.1 deg: 50.1 deg
against 50.2 deg over 300 s. So the correction is not in the code.

The duck stays seated the whole way. At 50 deg the upright cosine is still
0.9998 and the worst contact overlap is 1.9 mm.

### Long video

`videos/swing/duck_swing_2min.mp4` is a two minute clip, from 210 s of running:

```bash
--seconds 210 --target-degrees 80 --dismantle-at 195 --video-speed 2.5
```

`--video-speed` also takes a list of `(until, speed)` pairs in code, so one clip
can change pace. The two minute clip uses `[(15,1),(195,2),(1e9,1)]`:

| video | speed | run time |
|---|---|---|
| 0:00-0:15 | real time | 0-15 s |
| 0:15-1:45 | 2x | 15-195 s |
| 1:45-2:00 | real time | 195-210 s, the seat opens and the duck falls |

The target is set out of reach on purpose, so the amplitude regulator never eases
off and the arc grows the whole way, 20 deg to 53 deg. The camera pulls back as
the arc grows, then follows the duck down. The current speed is written on each
frame. Files are written with `+faststart`, or a player that streams shows
nothing until the whole file is loaded.

Frames are timed in run seconds, not steps. The comparison needs a tolerance:
`2/25` is not exact in binary, so each frame time landed a hair past its tick
and the check waited one step too long. That cost 5 s of a 2 minute clip.

### Getting off the swing

`--dismantle-at` puts the seat floor on a hinge, latched shut by one equality.
The latch lets go, the duck's own weight swings the floor open, and the duck
drops out while the seat stays on the ropes. Nothing is pushed or teleported.

Without the flag the floor is fixed and there is no equality, so the graded task
still reports `root_welds` 0.

Two things had to be right, and both took measuring:

| what | why |
|---|---|
| release at the front of the arc | anywhere else throws the duck into the backrest |
| the flap folds back 160 deg | at 90 deg it hangs down as a hook and the duck catches on it |

For the release, 11 points across one swing period gave 7 clean falls when fired
on a clock. Waiting for the front of the arc, where the swing has just stopped
and started back, gives 11 of 11. The duck is nearly still there, and the tilted
seat drops it away from the backrest.

For the flap, at a 50 deg swing:

| flap opens to | landed | ends up on |
|---|---|---|
| 89 deg | never | the flap itself |
| 126 deg | 2.0 s | the floor |
| 160 deg | 0.3 s | the floor |

Both were found by testing the release the clip actually uses. Shorter runs
released at 27-44 deg, where the duck cleared the flap, so those tests passed
while the clip stayed broken.

After it lands the duck stops bracing and cycles two poses, folding its legs in
and driving them down. **It does not get up.** It stays at about z 0.05 with an
upright cosine near 0.18, which is lying down. Standing from a face-down landing
needs a trained policy, not an open-loop pose cycle.

## Evidence

- Nominal 60-second run: final 20-second amplitude **22.65°**; the 30 measured
  half-cycle amplitudes in that window are **18.60–22.62°**.
- Identical initial setup with fixed joint targets: **0.04°**, compared with
  active pumping's 22.65°. Net motor work is approximately **19.0 J** in the
  active episode and **−0.01 J** in the fixed-pose baseline.
- Legs alone, with the head held still: **18.8°**, against 0.04° for fixed
  joints. So the legs pump on their own. On the old seat this was 8.8°.
- Minimum trunk upright cosine relative to the seat: **0.9998**, against 0.934
  on the old seat. The duck barely tilts. Maximum nominal compliant-contact
  penetration is **0.57 mm**, against 0.92 mm before; the simulation does not
  claim perfectly rigid contact.
- The duck's underside touches the seat plate for **98%** of the run. The lap
  guard, bucket walls and rear floor never touch it.
- Maximum servo torque **0.528 Nm** against the **0.641 Nm** current limit.
- The 90-second suite passes all eight cases: fixed pose, nominal pumping,
  ±3° starting angles, 0.42/0.56 m hangers, 6.5 V battery, and doubled bearing
  damping. All seven pumping cases finish at **22.50–22.75°** amplitude, with
  worst penetration 0.58 mm and worst upright cosine 0.9998.
- The 6.5 V case uses less joint travel: hip 47 deg and knee 35 deg, against
  83 deg and 68 deg at 7.4 V.
- With **all 81** duck geoms made solid, the deepest overlap anywhere over a
  whole run is **0.60 mm**, and only between the trunk and the seat plate,
  lip, back panel and back cross bar. Nothing looks see-through.
- Eleven regression tests pass. Five original ones cover passive suspension,
  initial rest, direction-sensitive feedback, rejection of external
  pushes/collision bypasses, and active versus fixed pose. Three lock in the
  leg use: knees and hips must move, legs alone must pump, and the 2x knee
  harmonic must stay worse than 1x. Two lock in the seat: the duck's underside
  must touch the seat plate, and the parts that meet the seat must collide with
  their visible meshes. An eleventh replays a run with every duck geom solid
  and checks that nothing sinks into the set.
