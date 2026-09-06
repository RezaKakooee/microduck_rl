# Egg on the head — walk without dropping an egg balanced on the crown

**Date:** 2026-09-05.
**Script:** `src/microduck_lab/tasks/objects/egg_on_head.py`.
**Scene:** `src/microduck_lab/models/scene_egg.xml` (egg as a free
body) and `robot_allcollisions_egg.xml` (a copy of `robot_allcollisions.xml`
plus one geom, the head pad).
**Policy:** the pretrained `alpha_walking.onnx`, unchanged. The script adds a
head-level loop that writes the policy's 4-slot head-pose command every tick.
**Videos:** `videos/egg_on_head/egg_stand_stab.mp4`, `egg_stand_nostab.mp4`,
`egg_walk_stab.mp4`, `egg_walk_nostab.mp4` (side view, camera follows the head).

## Result in four lines

- **Standing works.** With the head-level loop the egg stays for the whole
  run (11 s on the head, pad tilt 0.1 deg rms). Without it the egg's own 25 g
  pulls the head 2.5 deg nose-down and the egg is off in 0.4-1.9 s.
- **Walking does not work with this policy.** The egg is thrown off within
  about a second at every speed, with or without the loop (best: 1.04 s and
  0.09 m at 0.25 m/s). Turning in place improves from 0.26 s to 3.2 s.
- The reason is not head tilt. The walking gait swings the head sideways
  with 4.5 m/s2 rms and 10 m/s2 peaks at about 2 Hz, and rolls it +-5 deg.
  The loop removes the mean tilt (7.8 deg nose-up while walking -> 0) but
  cannot follow a 2 Hz swing, and no head or body command reduces the swing.
- A flat pad was needed: on the bare shell the egg rolls off a level,
  standing head in 0.3-2 s, because the crown is a dome.

## Head-top measurement

Measured in the home pose from `duck_sim.make_policy` (trunk at z = 0.125 m),
by transforming every mesh vertex of the head body `jaw_soft` with its geom
`xpos`/`xmat` and taking the highest z.

| | |
|---|---|
| head body `jaw_soft` origin | (-0.009, 0, 0.2376) m; its local x axis points straight up |
| highest collision point | `top_head_shell`, world (0.0731, 0.000, **0.2794**) m, 41.8 mm above the head origin |
| where it is | on the front ridge (the forehead), 73 mm ahead of the trunk centre |
| crown shape, fore-aft | drops 2.4 mm over the 35 mm behind the ridge, 5 mm at 65 mm behind (a 4-5 deg slope) |
| crown shape, sideways | drops 3.4 mm at +-27 mm from the centreline (radius of curvature ~80-110 mm) |
| highest visual geom | also `top_head_shell`, same point (the visual and collision meshes are the same file) |

`top_head_shell` does have a `class="collision"` geom in
`robot_allcollisions.xml`, so the egg can rest on it. The problem is shape,
not collision.

## Was a pad needed? Yes

- Egg on the bare shell, standing, head held level by the loop (0.3 deg):
  the egg rolled backward 22 mm during the 1 s settle and was off at 0.3 s
  (long axis along the walk) or 1.1 s (long axis across). The dome slopes
  down behind the ridge, and the ridge itself is at the edge of the face.
- So `robot_allcollisions_egg.xml` adds `head_pad`: a cylinder, 30 mm
  diameter, 2 mm thick, flat, centred 15 mm behind the ridge (world x = 0.058
  m), top face at z = 0.2804 m in the home pose. Its underside is 0.4 mm above
  the shell's highest point under the footprint. Friction 1.2, condim 6,
  rolling friction 0.003. It has no mass (the head body has an explicit
  inertial element). `group="2"` so it renders.
- `--no-pad` switches the pad's contacts off at runtime for the bare-shell
  test; nothing shared was edited.

## Egg

- Ellipsoid, semi-axes 10 x 10 x 14 mm, 25 g, friction 1.2, condim 6,
  rolling friction 0.003, pale yellow, free joint. It lies on its side.
- `--egg-axis x` (default): long axis along the walk, so it can only roll
  sideways. `--egg-axis y`: long axis across, so it rolls fore-aft. x
  survives longer in every case.
- Rest height with the loop on: egg centre at **0.2836 m**, 9.7 mm above the
  pad top (the head sits 6 mm lower than the home pose once the policy holds
  it). Without the loop 0.2813-0.2817 m, because the head sags.

**Rolling friction, measured in isolation.** MuJoCo's rolling friction is a
creep, not a static hold. An egg on a static pad tilted 0.5 / 1 / 2 deg:

| rolling coefficient | 0.5 deg | 1 deg | 2 deg |
|---|---|---|---|
| 0.0002 (a physical rubber value) | rolls off freely | rolls off | rolls off |
| 0.003 (used) | creeps 3 mm/s | 6.5 mm/s | 21 mm/s |

With 0.0002 the egg leaves even a standing, level head in 2.8 s. 0.003 is the
closest this contact model gets to "an egg on a rubber mat holds about a
degree", and it still creeps. `--rolling` overrides it.

## Protocol

1. The duck stands 1.5 s with the head command active (egg parked on the floor).
2. The egg is set on the pad, 0.5 mm above it, lying on its side, and rests 1.0 s.
3. The walk (or turn) command is stepped in. The egg clock starts here.
4. A fall is the egg centre 15 mm below its rest height (in world or relative
   to the pad) or an egg-floor contact. The run stops 1 s after a fall.
5. Reported: time until the fall, trunk displacement while the egg was on,
   pad tilt (max and rms), and the egg's position in the pad frame every 1 s.

The sim is deterministic; each row is one run.

## The controller

Angles follow the harness convention. For any frame with rotation R whose
level orientation is the identity:

```
g      = R^T · [0, 0, -1]
roll   = atan2( g_y, -g_z )        > 0 : left side down
pitch  = atan2( g_x, -g_z )        > 0 : nose down
```

Trunk: R = trunk rotation (from `policy.get_projected_gravity()`).
Head: R = R_pad · P0^T, where R_pad is the pad geom's rotation and P0 is the
pad rotation in the home pose, so the head angles are zero when the pad is
level. On the real robot they would come from the head IMU instead.

Each control tick (50 Hz), for pitch and for roll separately:

```
I_pitch += pitch_head · dt                          (clipped to +-cap)
head_pitch_cmd = trim_pitch - kp·pitch_trunk - kh·pitch_head - ki·I_pitch
head_roll_cmd  = trim_roll  - kp·roll_trunk  - kh·roll_head  - ki·I_roll
head_pitch_cmd = clip(head_pitch_cmd, -1.1, 1.1)    (training caps)
head_roll_cmd  = clip(head_roll_cmd, -0.31, 0.31)
policy.head_offset = [0, head_pitch_cmd, 0, head_roll_cmd]
```

`policy.head_offset` is the 4-slot head-pose command [neck_pitch, head_pitch,
head_yaw, head_roll] that goes into obs cmd[3:7]; the policy tracks it with
its own neck servos. `--no-stabilise` sends only the trim.

Gains used: **trim_pitch = -0.29, trim_roll = 0, kp = 1, kh = 2, ki = 5 /s**
(`--kp-pitch --kp-roll --kh-pitch --kh-roll --ki-pitch --ki-roll --trim-pitch
--trim-roll`, optional `--tau` low-pass on the trunk angles).

Why these terms, measured:

- At zero head command the policy holds the head **11 deg nose-down**, so
  the trim is needed even to stand. head_pitch cmd -0.3 gives -0.5 deg.
- Plant, standing: head_pitch cmd -> pad pitch 0.64 rad/rad, 63 % in 0.1 s;
  head_roll cmd -> pad roll 0.94 rad/rad, 63 % in about 0.15 s.
- The egg's weight makes the head sag 2.5 deg in 0.6 s. Only the terms on the
  head's OWN tilt (kh, ki) see this; the trunk term (kp) sees nothing because
  the trunk pitches 0.2 deg rms while walking and not at all while standing.
  Standing grid: ki = 0 -> off in 0.6-2.1 s at every kh; ki = 2 -> 0.5-3.7 s;
  ki = 5 -> stays.
- Walking grid (kp 0/1/2, kh 1..4, ki 2..10, tau 0/0.05, both egg axes, 0.25
  and 0.3 m/s): every setting falls between 0.44 and 1.06 s. Higher kh lowers
  the roll swing a little (sd 2.9 -> 2.1 deg at kh_roll = 3) and nothing else.

## Comparison table

Egg long axis along the walk (`--egg-axis x`), rolling friction 0.003, 10 s
runs. "Time" is until the fall, counted from the command; "walked" is the
trunk displacement while the egg was on.

| case | no stabilisation (trim only) | stabilised (kp 1, kh 2, ki 5) |
|---|---|---|
| standing | fell at 1.94 s (pad tilt 2.6 deg rms) | **stayed 10 s** (11 s on the head), tilt 0.1 deg rms, max 0.3 deg |
| walk 0.25 m/s | fell at 0.52 s, walked 0.04 m, tilt max 3.9 deg | fell at **1.04 s**, walked 0.09 m, tilt max 4.6 deg, rms 2.7 |
| walk 0.30 m/s | fell at 0.46 s, walked 0.04 m, tilt max 8.7 deg | fell at 0.70 s, walked 0.08 m, tilt max 4.8 deg, rms 3.0 |
| walk 0.30 + yaw 0.5 rad/s | fell at 0.46 s, 0.04 m, turned 5 deg | fell at 0.60 s, 0.06 m, turned 9 deg |
| turn in place, yaw 1.0 rad/s | fell at 0.76 s, turned 11 deg, tilt max 10.5 deg | fell at **3.18 s**, turned 19 deg, tilt max 7.0 deg, rms 2.1 |

Same with the egg across the walk (`--egg-axis y`, rolls fore-aft):

| case | no stabilisation | stabilised |
|---|---|---|
| standing | 0.40 s | stayed 10 s |
| walk 0.25 | 0.64 s | 0.74 s |
| walk 0.30 | 0.42 s | 0.48 s |
| walk 0.30 + yaw 0.5 | 0.40 s | 0.42 s |
| turn in place, yaw 1.0 | 0.26 s | 1.44 s |

Sensitivity rows (stabilised, egg axis x):

| variant | result |
|---|---|
| standing, bare shell (`--no-pad`) | fell at 0.28 s |
| standing, free-rolling egg (`--rolling 0.0002`) | fell at 2.82 s at 0.1 deg rms tilt |
| walk 0.25, free-rolling egg | fell at 0.52 s |
| egg set on the head 3 s into a steady walk (`--place-at 3`), 0.25 / 0.30 | fell at 0.60 s / 0.38 s |
| walk 0.2 m/s (axis y, kh 1, ki 5) | fell at 1.04 s; the duck steps and moves 0.08 m/s |

Notes on the commands: a yaw command of 0.5 rad/s alone does not move the duck
at all (deadband), so the turn rows use 0.3 m/s + 0.5 rad/s (moves, turns
slowly) and 1.0 rad/s alone (turns in place at about 0.1 rad/s).

## Why walking fails: the head is shaken, not tilted

Measured without the egg, steady walking (2-8 s after the command), pad frame:

| | pad pitch | pad roll | pad sideways acc | pad vertical acc |
|---|---|---|---|---|
| 0.30 m/s, no loop | mean -7.8 deg (nose up), sd 0.7 | mean -1.7, sd 3.6, range -6.9..+3.9 | 4.6 m/s2 rms, 10.4 peak | 2.4 rms, 8.3 peak |
| 0.30 m/s, loop | mean 0.0, sd 1.0 | mean 0.0, sd 2.6, range -4.4..+3.8 | 4.4 rms, 10.6 peak | 2.4 rms, 8.1 peak |
| 0.25 m/s, loop | mean 0.0, sd 1.0 | sd 3.1 | 4.5 rms, 10.1 peak | 2.3 rms, 7.3 peak |
| trunk itself, 0.30 | sd 0.2 | sd 2.4 | sways +-20 mm at ~2 Hz | |

- The loop does its job on the mean: -7.8 deg -> 0.0 deg pitch, -1.7 -> 0.0
  deg roll. It cannot follow the +-5 deg roll swing at 2 Hz through a plant
  with 0.1-0.15 s lag.
- The killer is the sideways acceleration. Sliding friction 1.2 can hold at
  most 11.8 m/s2 with full normal force, and the vertical jerks (8 m/s2) take
  most of the normal force away at the same moments; the egg lifts off the
  pad (no contact for a few ticks) and is thrown sideways. A tray would have
  to bank 25 deg rms / 45 deg peak to cancel it; the roll cap is 18 deg.
- The step-in transient at the walk command is the same story: head pitch
  -4.7 deg and roll -7 / +9.5 deg within 0.5 s, 8 m/s2 sideways, while the
  trunk rolls only +-2 deg. The policy moves the head itself: it uses the
  190 g head as a balance mass.
- Things that did not change the shaking (sideways rms stayed 4.2-4.7 m/s2):
  body-pose command z -30..+20 mm, body pitch +-0.2 rad, neck_pitch command
  -0.8..+0.5 rad, roll feedforward kp_roll 3, roll feedback kh_roll 3, a
  0.05 s low-pass, and 0.2 m/s (the duck steps in place with the same swing).

## Honest limits

- Every number is one deterministic run. No noise, no domain randomisation.
- MuJoCo's rolling friction creeps instead of holding. The chosen 0.003 is a
  compromise; the physical 0.0002 loses the egg even standing. A real egg on
  a rubber pad would hold a standing head with no loop at all, and would still
  be thrown by this gait.
- The pad frame is read from the simulator (`geom_xmat`); on the robot the
  same angles come from the head IMU (`head_imu` site exists in the model).
- The pad is rigid, massless, and floats up to 2.5 mm above the back of the
  crown. The egg-pad contact is MuJoCo's generic convex pair (one contact
  point), which is fine for rolling but loses contact under vertical jerks.
- The head-pose command only asks the policy; the policy tracks it at 0.64
  (pitch) and 0.94 (roll) rad/rad and keeps moving the head with the gait.
  A scripted loop cannot take that away from it.
- The camera in the videos follows the head, not the trunk (`--track head`).

## RL follow-up (design only, not launched)

The scripted loop shows the limit is the gait, so the fix is a policy that
walks with a quiet head. Built on the velocity recipe
(`microduck_velocity_env_cfg.py`):

- **Robot and scene.** Use `robot_allcollisions_egg.xml` (pad included) and
  add the egg as a free body in the env scene, spawned on the pad at reset:
  after the robot pose reset, set the egg 10.5 mm above the pad top with
  +-3 mm random offset, zero velocity. MuJoCo Warp: use a sphere or capsule
  egg if the ellipsoid pair is not supported there.
- **Observations.** Keep the 61-D actor obs unchanged (the egg is not
  observable on the real robot, and the contract must hold). Give the critic
  the egg's position in the pad frame and the pad's tilt as privileged obs.
  Zero-pad the head-pose command slot (sample a tiny range, keep the term):
  in this task the head pose is a means, not a command.
- **Rewards.** Keep the twist tracking terms and the usual regularisers, then:
  1. `egg_on_pad`: +1 per step while the egg touches the pad and its centre is
     within 12 mm of the pad centre.
  2. `head_level`: exp(-(pad tilt / 0.05 rad)^2).
  3. `head_acc`: -|a_head_xy|^2 from the head IMU site's linear acceleration
     (the term the measurements say matters most).
  4. A small penalty on head joint velocity so the neck stops being the
     balance mass.
- **Termination.** Egg centre 20 mm below the pad top, or egg-floor contact,
  plus the usual trunk-fall termination.
- **Curriculum.** Start with the egg welded to the pad (reward 1-4 only),
  release the weld once head_acc rms is under ~2 m/s2; then widen the speed
  command range from 0.15 to 0.3 m/s.
- **Domain randomisation.** Egg mass 15-40 g, rolling friction 0.001-0.005,
  pad friction 0.8-1.5, and the existing robot DR.
- **What to judge it by.** Time until the egg falls and distance walked with
  it on, not the reward: the standing case already scores well on tilt alone.
- Expected outcome: a slower, flatter gait with the head decoupled, the way a
  waiter walks with a tray. Smoke-test first (`--env.scene.num-envs 64
  --agent.max_iterations 5`), as AGENTS.md says.

## Commands

```bash
# standing, walking, turning (CPU)
uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0 --seconds 10
uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0.25
uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0.25 --no-stabilise
uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0 --yaw 1.0
uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0 --no-pad          # bare shell
uv run python -m microduck_lab.tasks.objects.egg_on_head --speed 0 --rolling 0.0002  # free-rolling egg

# videos (GPU node)
  MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg python src/microduck_lab/tasks/objects/egg_on_head.py \
  --speed 0.25 --seconds 4 --azimuth 270 --cam-distance 0.6 --video videos/egg_on_head/egg_walk_stab.mp4
```
