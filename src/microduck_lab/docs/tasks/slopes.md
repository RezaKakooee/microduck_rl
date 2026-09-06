# Slopes and stairs — `Mjlab-Velocity-Slopes-MicroDuck`

**Date:** 2026-09-03/04.
**Goal:** make the duck WALK (normal feet) up and down gentle slopes and up
small steps, with the same 61-obs / 14-action contract as every other policy.

Everything here is measured, not planned, unless it says otherwise.

---

## 1. What was built

| File | What it is |
|---|---|
| `src/microduck_lab/rl/microduck_velocity_terrain_env_cfg.py` | The task cfg. Built on `make_microduck_velocity_env_cfg(rough=True)`, so DR / obs / noise / delays and the rough-terrain NaN guards come for free. Adds its own terrain generator, a planar-faced slope heightfield class, and a terrain curriculum. |
| `src/microduck_lab/rl/mdp_terrain.py` | One curriculum function (`terrain_levels_walk`) and its pure helper. Nothing was added to `mdp.py`. |
| `src/mjlab_microduck/tasks/__init__.py` | One block appended at the end: registers `Mjlab-Velocity-Slopes-MicroDuck`. |
| `src/microduck_lab/src/microduck_lab/tests/test_terrain_cfg.py` | 10 CPU tests (see section 5). |
| `src/microduck_lab/models/scene_slope.xml` | CPU MuJoCo evaluation scene: plane + box ramp up, box ramp down, three risers. |
| `src/microduck_lab/tasks/walking/eval_slope.py` | Runs an ONNX walking policy on that scene and reports how far it got. |
| `videos/eval_slope/slope_baseline.mp4`, `videos/eval_slope/slope_baseline_steps.mp4` | The pretrained walker on the ramp and on the steps. |

## 2. Swing-height measurement (done first)

`alpha_walking.onnx` on flat ground, command 0.3 m/s forward, 15 s, CPU MuJoCo
through `duck_sim.py` (timestep 0.005, current limit 1.75 A, projected
gravity). 36-37 swings per foot.

| Measured at | median peak | p90 | max |
|---|---|---|---|
| `left_foot` site | 16.6 mm | 17.3 mm | 17.6 mm |
| `right_foot` site | 14.9 mm | 15.4 mm | 15.6 mm |
| lowest vertex of the sole mesh, left | 15.0 mm | 15.7 mm | 15.9 mm |
| lowest vertex of the sole mesh, right | 14.8 mm | 15.2 mm | 15.4 mm |

So the gait lifts the sole about **15 mm**. It travelled 1.81 m in 15 s
(0.12 m/s against 0.3 commanded — the known tracking gap).

## 3. Terrain design, and why those numbers

mjlab terrain generator in **curriculum mode**: one column per terrain type,
10 rows of increasing difficulty, 8 m x 8 m tiles, a 2 m flat platform in the
middle of each tile where the robot spawns. Five columns:

| Column | Type | Difficulty 0 -> 1 | Spawn share |
|---|---|---|---|
| `flat` | plain box | — | 15 % |
| `slope_down` | frustum, platform on TOP | 5 deg -> 15 deg | 20 % |
| `slope_up` | inverted frustum, spawn in the PIT | 5 deg -> 15 deg | 25 % |
| `stairs_down` | pyramid stairs, platform on top | 10 mm -> 20 mm risers, 0.15 m treads | 15 % |
| `stairs_up` | inverted pyramid stairs, spawn in the pit | 10 mm -> 20 mm risers, 0.15 m treads | 25 % |

Why these numbers:

- **Steps 10-20 mm.** The sole lifts 15 mm (section 2). 10 mm is below the
  swing, 20 mm is at its top. Going above the swing height would ask the RL
  policy for a new gait, not a better one.
- **Slopes 5-15 deg.** At 15 deg a rubber sole at mu 0.7-1.3 is nowhere near
  slipping (tan 15 deg = 0.27). The difficulty is balance and pushing uphill,
  not grip. The existing rough task tops out at 5.7 deg, so this is the next
  band.
- **Both directions are separate columns.** Spawning on the platform of a
  pyramid means the first metres are always downhill; spawning in a pit means
  they are always uphill. Commands resample with random headings, so an
  episode sees both anyway, but the spawn decides what gets practised first.
  Up-hill gets the larger share because it is the hard part.
- **Planar slope faces.** mjlab's stock `HfPyramidSlopedTerrainCfg` builds
  `z = h * x * y` (bilinear). Probed on CPU, its faces are curved: the angle
  you configure holds only along the two axes, the diagonals are up to 1.4x
  steeper and the tile corners flatten out. The task cfg carries a small
  `HfFrustumSlopeTerrainCfg` (max-norm distance from the centre) instead.
  Probed after building: **5.0-14.7 deg** across the 10 rows, and the
  diagonal gradient equals the axis gradient to 0.01 deg.
- **Risers, probed after building:** 10.6 mm (row 0) to 19.8 mm (row 9).
- **Spawn origins.** Every one of the 50 (row, column) origins was raycast on
  CPU: the terrain surface is within 0.1 mm of the origin z, pit types
  included. So a reset never places the robot under the floor (the reason the
  rough task avoided inverted terrains).
- **Foot swing target 20 -> 25 mm.** The walking recipe asks
  `foot_swing_height` for a 20 mm peak and the trained gait delivers 15-17 mm
  (about 20 % under). A foot that has to clear a 20 mm riser needs to peak
  above it, so the target moves to 25 mm. `foot_clearance` (the drag penalty)
  is unchanged. This is the ONLY reward change.

Kept from the rough branch, on purpose: `_soften_terrain_contacts` (2x softer
terrain solref — box edges otherwise produce impulsive NaN forces when a foot
lands on them), `nconmax = 200`, 30/50 solver iterations. A test checks they
are still there.

### Curriculum

mjlab's own `terrain_levels_vel` demotes an env when it walked less than
`|command| * episode_length * 0.5`. For a 0.3 m/s command and 20 s that is 3 m;
this robot walks 0.12 m/s, i.e. 2.4 m, so it would be demoted on every episode
and never leave level 0. `mdp_terrain.terrain_levels_walk` is tile-relative
instead:

- walked more than 25 % of a tile (2.0 m) -> one level harder;
- walked less than 8 % of a tile (0.64 m: fell at once, or stuck on the
  obstacle) -> one level easier;
- envs commanded to stand or turn in place are not judged.

Envs start on rows 0-3 (5-9 deg, 10-14 mm). Levels are logged per column
(`Curriculum/terrain_levels/slope_up` etc.), so the log shows which obstacle
is being learned.

## 4. Baseline: the pretrained walker on the evaluation scene

`scene_slope.xml` has three lanes; the robot spawns at x = 0 facing +x and is
commanded 0.3 m/s forward for 20 s.

| Lane | What is there | Result |
|---|---|---|
| `up` | 10 deg ramp, 1.5 m long, 0.264 m rise | **stalls 0.21 m up the ramp (14 %)**, 32 mm of height gained, does not fall — it keeps stepping in place on the slope |
| `steps` | risers of 10, 15, 20 mm, 0.3 m treads | **does not clear the 10 mm riser**, does not fall — it walks against the step and stays there |
| `down` | 10 deg ramp down from a raised platform | **reaches the bottom** (1.5 m of ramp) and walks on, 3.78 m total, no fall |

A single-riser sweep on the same lane: clears **3 mm**, stalls at **5 mm, 8 mm
and 10 mm**. So although the sole peaks at 15 mm, the swing is low at the
moment the toe passes the riser and the current gait cannot climb even a 5 mm
edge. Downhill costs it nothing. That is the gap the training run has to
close.

Commands:

```bash
P=$POLICIES/alpha_walking.onnx
uv run python -m microduck_lab.tasks.walking.eval_slope --walking $P --lane up    --lin-vel-x 0.3 --seconds 20
uv run python -m microduck_lab.tasks.walking.eval_slope --walking $P --lane steps --lin-vel-x 0.3 --seconds 20
uv run python -m microduck_lab.tasks.walking.eval_slope --walking $P --lane down  --lin-vel-x 0.3 --seconds 20
# other ramp angles: --ramp-deg 5 / 15 (the script re-poses the ramp boxes at load)
# video (GPU node only):
MUJOCO_GL=egl uv run python -m microduck_lab.tasks.walking.eval_slope --walking $P --lane up --video videos/eval_slope/slope_baseline.mp4
```

## 5. Tests

`uv run --with pytest pytest src/microduck_lab/src/microduck_lab/tests/test_terrain_cfg.py` — 10 tests, CPU:
terrain types and curriculum mode; slope angles and inverted flags; step
heights and tread; swing target above the tallest riser; rough physics
guards kept; curriculum wiring; the promotion masks; play mode is a random
mix on its own copy; the 61D obs contract is identical to the walking task;
and the geometry test — builds a 2-level grid on CPU, raycasts every origin,
measures every slope face (axis and diagonal) and every riser.

## 6. Smoke test

RTX 3080, 64 envs, 5 iterations . Read off the tensorboard events
of the first run:

- terrain grid built (1174 terrain geoms incl. 20 heightfields on CPU; the GPU
  run reports the same spec_fn softening), 64 envs stepped;
- **no NaN**: `Episode_Termination/nan_state = 0`, all 53 logged scalars
  finite;
- all 17 reward terms compute; every penalty term is <= 0
  (`action_rate_l2 -0.10`, `foot_swing_height -0.010`, `body_ang_vel -0.027`,
  `foot_slip`, `foot_clearance`, `self_collisions`, `dof_pos_limits` all
  negative, `angular_momentum` -0.000);
- the terrain curriculum logs per column
  (`Curriculum/terrain_levels/{flat,slope_down,slope_up,stairs_down,stairs_up}`),
  mean level 0.31 after 5 iterations;
- the run auto-exported an ONNX: input `obs [1, 61]`, output `actions [1, 14]`,
  finite actions on a zero observation.

As expected for 5 iterations the policy falls at once
(`Episode_Termination/fell_over`, mean episode length 35 steps = 0.7 s).

## 7. The training run

```bash
uv run train Mjlab-Velocity-Slopes-MicroDuck --env.scene.num-envs 4096 \
  --agent.max-iterations 8000 --agent.logger tensorboard --agent.run-name slopes-v1
```

- **The training run**: 4096 envs, 8000 iterations
  (about 4-5 h on an RTX 3080 at ~2 s/iter).
- Console log: `the training log`.
- Checkpoints and tensorboard: `logs/rsl_rl/velocity_slopes/<date>_slopes-v1/`
  (`model_XXXX.pt` every 250 iterations).
- Watch: `Curriculum/terrain_levels/slope_up` and `.../stairs_up` (are the
  hard columns being promoted?), `Metrics/twist/error_vel_xy`, `Metrics/peak_height_mean`
  (the swing peak the policy settles on; the target is 25 mm), and every
  `Episode_Reward/<penalty>` staying <= 0.
- Resume: `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.

The run was launched and NOT watched: there is no result to report yet.

## 8. How to evaluate a checkpoint

```bash
# 1. export (bakes the obs normaliser — the only valid path)
uv run scripts/export.py Mjlab-Velocity-Slopes-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity_slopes/<run>/model_XXXX.pt
# 2. the three lanes, CPU
uv run python -m microduck_lab.tasks.walking.eval_slope --walking <exported>.onnx --lane up
uv run python -m microduck_lab.tasks.walking.eval_slope --walking <exported>.onnx --lane steps
uv run python -m microduck_lab.tasks.walking.eval_slope --walking <exported>.onnx --lane down
# 3. in-sim play on the training terrain mix (GPU)
uv run play Mjlab-Velocity-Slopes-MicroDuck --checkpoint-file ...
```

Judge it by the rollout: metres up the ramp, which risers it clears, and
whether it falls. The ice runs taught that `air_time` and the reward curve
can look healthy while the policy shuffles in place. The numbers to compare
against the baseline in section 4: 0.21 m up a 10 deg ramp, 0 risers cleared,
100 % down.

## 9. Honest limits

- **No policy has been evaluated yet** beyond the pretrained baseline. The
  training run was launched, not watched.
- **The policy is blind.** It has the same 61D proprioceptive obs as the
  walker — no terrain scan. It can only react to a step or a slope once a foot
  or the trunk feels it. That is the same bet the rough task makes.
- **Stairs at 10 mm may be too hard for level 0.** The baseline gait stalls at
  5 mm. The swing target at 25 mm and the tracking reward push against that,
  but if `Curriculum/terrain_levels/stairs_up` stays at 0 for thousands of
  iterations, the first knob is `STEP_HEIGHT_RANGE[0]` (5 mm).
- **The curriculum judges by distance, whatever the direction.** An env that
  walks 2 m along a flat contour of a pyramid is promoted like one that
  climbed.
- **Heightfield contacts on GPU** (MJWarp `HFIELD x MESH` through the convex
  path) are used for the slopes. The rough task already used them; the smoke
  test is the only NaN check so far.
- **The evaluation scene is boxes.** A real ramp has an edge at the top; the
  scene's ramp meets its platform flush (same thickness trick as
  `slope_terrain.py`).
