# Skate blades — `Mjlab-Velocity-Blades-MicroDuck`

Date: 2026-09-04. Status: built, smoke-tested, training launched (job
). Not yet evaluated on a trained checkpoint.

## What this is

The ice task (`docs/project_journey/03_ice_skating_rl.md`) used isotropic
friction. At low mu the duck had nothing to push against in any direction, so
it learned to stand still or shuffle. A real skate blade is anisotropic:
slippery along the blade, grippy across it. This task gives the duck's soles
that property with MuJoCo contact pairs, and trains the same velocity task on
it.

## Files

| File | What |
|---|---|
| `src/microduck_lab/rl/microduck_velocity_blades_env_cfg.py` | The task cfg. Builds on the walking recipe. Adds the pairs in a `spec_fn`, the DR event, the per-step projection event, the curriculum, and the heading hold. |
| `src/microduck_lab/rl/mdp_blades.py` | All new MDP code: the projection maths, the two events, the curriculum. Nothing was added to `mdp.py`. |
| `src/mjlab_microduck/tasks/__init__.py` | One block appended: registers `Mjlab-Velocity-Blades-MicroDuck`. |
| `src/microduck_lab/src/microduck_lab/tests/test_blades_cfg.py` | 19 CPU tests: the tangent mapping, the projection maths, the pairs on the real mjlab scene, the events writing the model, the cfg wiring, the 61D obs contract. |
| `src/microduck_lab/models/scene_blades.xml` | CPU playback scene with two static `<pair>` entries. |
| `src/microduck_lab/tasks/skating/blades_rollout.py` | CPU rollout that applies the same per-step projection as training. Use this to judge a checkpoint. |
| `src/microduck_lab/tools/blades_warp_check.py` | GPU-node check: mujoco_warp honours per-world anisotropic pairs, and the projection is live in the built env. |

## The measurement: the pair frame is world-fixed

A `<pair>` has five friction terms `[t1, t2, spin, roll1, roll2]`. `t1` and
`t2` act along the two tangents of the contact frame. I measured which tangent
is which:

| Test | Result |
|---|---|
| Box, pair friction `[0.02, 1.2]`, shoved along world X | slid 0.19 m |
| Box, pair friction `[0.02, 1.2]`, shoved along world Y | slid 5.12 m |
| Box, pair friction `[1.2, 0.02]` | swapped: X 5.12 m, Y 0.19 m |
| Contact frame read from `data.contact[i].frame` (box AND both duck feet) | normal `(0,0,1)`, tangent1 `(0,1,0)`, tangent2 `(-1,0,0)` |
| Duck standing at yaw 0, pair `[1.2, 0.02]`, trunk shoved +X / +Y | +X slid 0.35 m, +Y slid 0.27 m |
| Duck standing at yaw 0, pair `[0.02, 1.2]`, trunk shoved +X / +Y | +X slid 0.16 m, +Y slid 0.67 m |
| Sole mesh extents in world at yaw 0 | 54 mm along X, 41 mm along Y, 13 mm tall |

So: **slot 0 is world Y, slot 1 is world X, for any contact with the +Z
plane, whatever the foot's yaw.** MuJoCo (C and Warp) builds the tangent
frame from the plane normal alone. The sole's long axis is world X when the
duck faces +X. This means a static pair is a blade only while the foot points
along X. It cannot follow a turned foot.

`src/microduck_lab/src/microduck_lab/tests/test_blades_cfg.py::test_pair_slot_0_is_world_y_and_slot_1_is_world_x`
and `..._contact_frame_is_world_fixed_for_the_duck_foot` lock this in.

## How the blade follows the foot

The real blade is a friction ellipse in the foot frame: radius `mu_along`
on the sole's long axis, `mu_across` perpendicular. MuJoCo can only hold an
axis-aligned ellipse (world X and Y). So a step-mode event
(`mdp_blades.project_blade_friction`) rewrites `pair_friction` for every env
and foot at every control step (50 Hz). Two rules exist; `BLADE_RULE` picks
one.

**`load` (used for training).** For a foot that carries a tangential
contact force, both slots get the true ellipse's radius along the direction
of that force:

```
r(d) = 1 / sqrt( (d.u)^2 / mu_along^2 + (d.v)^2 / mu_across^2 )
```

Pushing perpendicular to the blade gives `mu_across` (grip). Sliding along it
gives `mu_along` (glide). A foot turned out 30 degrees and shoved straight
back gets 0.09 (with 0.08 / 1.0), like a real blade. A foot with no load gets
the ellipse's X and Y intercepts. The force comes from the existing
`feet_ground_contact` sensor (net force, world frame), refreshed every physics
substep.

**`bbox` (rejected, kept for comparison).** The bounding box of the rotated
ellipse. Exact at 0 and 90 degrees, but at 30 degrees a straight-back push
already gets 0.5. Measured on the pretrained walker: its gait turns the feet
out 29-37 degrees and this rule hands it mu_x 0.5-0.6 for free. A policy would
just walk turned-out and never skate.

## Domain randomisation and curriculum

Pairs override geom friction, so the base `foot_friction` geom event would no
longer reach the physics. It is **deleted**. In its place:

- `blade_friction` (reset mode): samples `mu_along` and `mu_across` per env
  from two ranges, into buffers on the env. Both feet of one env share the
  sample. Values are absolute, so nothing accumulates.
- `blade_projection` (step mode): writes the per-env `pair_friction` as
  above. Both events declare `pair_friction` via `requires_model_fields`, so
  mjlab expands it per world. Verified on a GPU node with
  `src/microduck_lab/tools/blades_warp_check.py --env`: mujoco_warp slides the box 0.19 m
  along X and 5.12 m along Y for `[0.02, 1.2]` (same as CPU) and honours
  per-world values (two worlds with swapped values swap the result); in the
  built task `pair_friction` has shape `(num_envs, 2, 5)`, every env carries
  its own `mu_along` / `mu_across`, the values changed on 40 of 40 steps, the
  `[blades] spec_fn` print appears, and spin/roll stay at the defaults.
- `blade_friction` curriculum: rewrites the two ranges on the live event
  term at stage steps. Stage 0 is isotropic walking grip, so the first 1000
  iterations are the walking task. Then `mu_along` ramps down while
  `mu_across` stays grippy:

| Iteration | along | across |
|---|---|---|
| 0 | 0.70-1.30 | 0.70-1.30 |
| 1000 | 0.40-0.90 | 0.70-1.30 |
| 2000 | 0.25-0.60 | 0.80-1.30 |
| 3000 | 0.15-0.40 | 0.80-1.30 |
| 4000 | 0.08-0.25 | 0.80-1.30 |
| 5000 | 0.04-0.15 | 0.80-1.30 |

`Curriculum/blade_friction` in the log is the midpoint of the along range.

## Commands: heading hold

The projection is exact at foot yaw 0 and 90 degrees and weakest near 45. The
hip-yaw joints allow about +-30 degrees, so the body heading must stay near
zero to keep foot yaw small. mjlab's `heading_command` is on: the yaw-rate
slot is `2.0 * (target heading - heading)`, clipped to +-1.0, with the target
sampled in +-0.2 rad and the reset yaw in +-0.2 rad. Turn-in-place practice is
off. Linear commands: forward -0.3..0.5 m/s, lateral +-0.1 m/s. The obs
contract is unchanged (61D; slot 2 is still a yaw-rate command).

Other changes from the walking recipe: `foot_slip` weight 0 (slip along the
blade is the medium); `air_time` keeps weight 3.0 with the ice window
0.10-0.40 s. `rough=True` raises `NotImplementedError` (pairs are built
against the single plane).

## Smoke test (debug node, RTX 3080)

```
uv run train Mjlab-Velocity-Blades-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard --agent.run-name blades-smoke
```

5 iterations, 1600 steps/s. `nan_state` 0.0. Every reward term computed,
every penalty <= 0, `foot_slip` 0, `Curriculum/blade_friction` 1.0 (stage
0). Random-policy falls as expected. Log dir:
`logs/rsl_rl/velocity_blades/2026-09-04_23-16-57_blades-smoke`.

## Training run

```
uv run train Mjlab-Velocity-Blades-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 8000 --agent.logger tensorboard --agent.run-name blades-v1
```

Log: `the training log`.
Checkpoints land in `logs/rsl_rl/velocity_blades/<date>_blades-v1/`.

What to watch. `Metrics/twist/error_vel_xy` is the judge, not `air_time`
(the ice v3 run kept a healthy air time while travelling 0.00 m). The
interesting stages start at iteration 3000. If the metric steps down exactly
at a stage boundary, the stage is too early. If it stays flat while the
policy still walks with feet straight, the blade is not biting.

## How to evaluate

1. Export a checkpoint (the normalizer must be baked in):

```
uv run scripts/export.py Mjlab-Velocity-Blades-MicroDuck --checkpoint-file logs/rsl_rl/velocity_blades/<run>/model_XXXX.pt
```

2. Roll it out with the training physics (per-step projection, `load` rule):

```
uv run python -m microduck_lab.tasks.skating.blades_rollout --walking <policy.onnx> --mu-along 0.08 --mu-across 1.0 --lin-vel-x 0.3 --seconds 15
```

It prints travelled distance, body speed, yaw drift, whether it fell, the
mean foot yaw, and how often each foot was loaded and at what effective mu.
Skating shows as: forward speed near the command, feet mostly straight, the
pushing foot loaded at high mu (perpendicular push), the gliding foot at low
mu. Try `--mu-along 0.04` and `0.15` (the final range) and `--rule bbox` to
see the difference.

3. The plain static scene (exact only while the duck faces +X):

```
uv run python -m microduck_lab.tools.headless_rollout --walking <policy.onnx> --new-cmd-obs --xml src/microduck_lab/models/scene_blades.xml --lin-vel-x 0.3 --seconds 12
```

`--foot-friction` does nothing here: the pair overrides geom friction.
Change the two numbers in the XML (`friction="across along ..."`) instead.

4. Video: add `--video out.mp4` to either script on a GPU node with
`MUJOCO_GL=egl`.

## Baseline: the pretrained walker on blades

`$POLICIES/alpha_walking.onnx`,
commanded 0.3 m/s for 12 s. It never fell.

| Physics | Forward speed | Travelled | Notes |
|---|---|---|---|
| Plain floor (control) | 0.131 m/s | dx 1.47 m | the known tracking gap |
| Static pair, XML values (`headless_rollout.py`) | 0.167 m/s | dx 0.35, dy 1.95 m | the world-fixed grippy Y axis steered it 1.4 rad off course |
| `bbox` rule, along 0.08 | 0.140 m/s | dx 1.38, dy 0.86 m | feet 29-37 deg out, effective mu_x 0.5-0.6: walking on a normal floor |
| `load` rule, along 0.08 (training physics) | 0.154 m/s | dx 1.79, dy 0.32 m | feet 4-10 deg out; loaded 48-50% of steps at mu 0.40-0.45 |

The last row matters. The walker's stance loads are mostly lateral (it
waddles), which is across the blade where the ellipse is wide, and the small
fore-aft push rides along. So blades do not stop this robot from walking. The
question the run has to answer is whether RL finds a glide-and-push gait
that tracks the command better than walking does at along 0.04-0.15.

## Honest limits

- **MuJoCo cannot rotate a friction ellipse.** Everything above is an
  axis-aligned stand-in, rewritten at 50 Hz. The `load` rule is exact along
  the current push direction and wrong off it. Off-axis loads are permitted at
  the same magnitude in every direction until the next step.
- **One step of lag.** The friction written at step k is based on the force
  measured after step k's physics and is used in step k+1. A push that flips
  direction gets one 20 ms step at the old radius. A pulsed push could exploit
  this; `action_rate` taxes it and it needs a precise 25 Hz pattern, but it is
  possible.
- **The walker already moves on blades.** With `mu_along` 0.08 the pretrained
  walker still makes 0.15 m/s. Blades are not a wall here the way isotropic
  ice was, so "the policy walks" would not by itself be a failure of the
  physics. Judge by the foot-load pattern in `blades_rollout.py`, not only
  speed.
- **Heading is held, not commanded.** This policy is trained to skate roughly
  straight. Yaw commands larger than the heading controller's small
  corrections are out of distribution.
- **Not evaluated yet.** No trained checkpoint has been rolled out. The
  numbers above are the physics baseline.
