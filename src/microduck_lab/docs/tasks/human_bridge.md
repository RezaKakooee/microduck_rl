# Human bridge — a brother lies across a gap, his sister walks over his back

**Dates:** 2026-09-25 to 2026-09-27.
**Status:** works in simulation, not every time. She WALKS across the real,
physical brother under the strict rules (section 4). With the best policy
(beam-v1, checkpoint 2000) she crosses 2 of 3 times from the normal start.
Every failure happens at an edge of his body.

Based on the viral clip of a boy lying across a gap in the street so his little
sister can cross over his back. The clip is in the repo as
`videos/human_bridge/human-bridge-original.mp4` (source unknown).

Blog post: https://rezakakooee.github.io/microduck-human-bridge/

## 1. Result

| What | Where |
|---|---|
| Best policy (ONNX, 61 in / 14 out) | `local_storage/hb_dev/rl_policy/s6_eval/beam_v1_iter2000.onnx` |
| Its checkpoint | `logs/rsl_rl/velocity_sideways_beam/2026-09-27_09-40-07_sideways-beam-v1/model_2000.pt` |
| Best film (1080p): he lies down, she walks across 1 s later | `videos/human_bridge/rl_policy/clean_15_i2000_settle1.0_x-0.23_smoothcam_1080p.mp4` |
| Original clip on top, our run below (twice), no sound | `videos/human_bridge/rl_policy/human_bridge_inspiration_vs_microduck_v3_hq.mp4` |

Crossings under the strict rules, from the normal start (x = -0.25 m, small jitter):

| Checkpoint | Fixed copy of his body (replica) | The real, moving brother (story) |
|---|---|---|
| 1000 | 0 / 5 | 0 / 3 |
| 2000 | 3 / 5 | 2 / 3 |
| 2999 (last) | 2 / 5 | 2 / 3 |

**How she fails:** her foot lands tilted (17-25 deg), and her ankle bracket
(6 mm above the sole) clips a ledge corner: at the 30 mm entry hole between her
ledge and his feet, or at the far pit. The policy is blind: it cannot see edges.
The result depends on where her feet happen to meet the edge: over 23 start
positions, checkpoint 1000 crossed about 65% of the time.

## 2. Folders (one per policy)

| Policy | Code | RL task | Scratch | Videos |
|---|---|---|---|---|
| RL (this doc) | `tasks/human_bridge/rl_policy/` | `rl/rl_policy/` | `local_storage/hb_dev/rl_policy/` | `videos/human_bridge/rl_policy/` |
| Scripted | `tasks/human_bridge/scripted_policy/` | `rl/scripted_policy/` | `local_storage/hb_dev/scripted_policy/` | `videos/human_bridge/scripted_policy/` |
| Crawl | `tasks/human_bridge/crawl_policy/` | `rl/crawl_policy/` | `local_storage/hb_dev/crawl_policy/` | `videos/human_bridge/crawl_policy/` |

Each policy uses only its own folders. `tasks/human_bridge/rl_policy/` holds
the world, set, brother, story, rules and bake that this doc describes.

## 3. The world and the set (`tasks/human_bridge/rl_policy/`)

| Part | File | Notes |
|---|---|---|
| Two ducks, BAM XL330 motors, physics-only guard | `world.py` | After `World.start()` only motor targets change the world. `step()` refuses external forces and changes to guarded model fields. |
| The set (layout "F1w") | `scene.py` | Her ledge 335 mm (level with his shins), edge at x = -76 mm. Near step 300 mm. Gap 100 mm. Far shelf 256 mm under his head. Far ledge 322.5 mm. |
| He lies down | `brother.py`: `LieDown`, `PLANK` | A 2 s joint-target blend. He falls forward; soles stay on the step; his neck servo rests on the far shelf. |
| He holds himself straight | `brother.py`: `HoldStraight`, `SETTLE_S = 8` | An integral on his joint targets. Needs about 5 s to settle. With 757 g on his back: at most 2.7 mm sag. Lands well from 15 of 15 starts. |
| Frozen copy of him for RL | `bake.py` -> `bridge_pose.json` | Baked after `SETTLE_S`, 39 solid meshes. Test `Baked` checks it sits on the real brother. |
| The story and its rules | `story.py`: `Story`, `Judge` | See section 4. |

**PLANK:** legs straight, ankles -/+1.57, head_pitch -1.2, hip roll -/+0.38
(legs pressed together).

**His back, as the physics sees it** (sole-rest height, F1w): her ledge 335 ->
a 30 mm hole (35 mm deep) -> his feet about 16 mm lower -> shin servos 335
(two rails, a hole between his legs) -> +4.7 mm onto his trunk -> head top 338 ->
his head slopes down about 17 deg -> a 12-15 mm far pit -> far ledge 322.5.
Where a sole can rest: 31-55 mm wide on his legs, 30-54 on his trunk, 22-31 on
his head.

## 4. The rules (`story.Judge`) — read this before claiming success

A crossing counts only if, from the moment she starts walking:

- her trunk stays upright (up >= 0.9, tilt at most about 25 deg);
- only her feet touch anything (soles, and the foot shells 2 mm above them);
- her feet touch only his body, her ledge (`near_top`) and the far ledge
  (`far_top`) — side ledges, notch floors and the floor fail;
- her trunk stays above `DROP_Z`, and he stays lying;
- she really steps: at least 3 landings per foot on him, each after at least
  2 ticks in the air and 5 mm of lift;
- all of her ends past the far edge, then 3 s of standing with the rules on.

The same rules score the fast replica test and the real story. Look at the
video frames yourself before calling anything a success.

## 5. The walker that works: `Mjlab-Velocity-SidewaysBeam-MicroDuck` (`rl/rl_policy/`)

Why a new skill: measuring showed the old sideways walker (v4 iter 1000) plus a
simple steering law crossed FLAT beams down to 28 mm wide. It failed on almost
any step or ramp on a beam 48 mm wide or narrower. His back is exactly that.

| Part | Choice |
|---|---|
| Terrain (`beam_terrain.py`) | 10 kinds x 10 rows of narrow beams (150 -> 28 mm) with steps, ramps, gaps, bumps, rounded tops, and a copy of his back. Harder rows only after success. |
| Command (`mdp_beam.py`) | The deployment steering law every step: vy 0.08-0.14, vx = clip(-0.055 + 5 y, +-0.3), wz = clip(0.06 - 2 heading_err, +-0.4). Never zero. |
| Rewards | Pay for new ground only (a ratchet), capped at 1.25 x the commanded speed. Other positive terms are gated on progress. Offset and heading costs saturate. A failure costs -10 after dt scaling. |
| Terminations | The story rules: tilt, any non-foot contact, a foot off the beam, dropped. |
| Motors | The story's motor model (no action or obs delay, 7.4 V). With the recipe's 15-30 ms delay, the old walker failed a flat 28 mm beam in training (8/32 vs 32/32). So this policy is for the simulator, not yet for the real robot. |
| Training | Warm start from sideways v4 iter 1000; 3000 iterations, 4096 envs, about 3 h 12 min on one RTX A4500. |

## 6. Hard limits of this robot

1. **She cannot stand on one foot** (no ankle roll; hip roll stops at 0.38 rad).
   She must keep walking.
2. **Forward walking does not fit his back** (her feet are 84 mm apart).
   Sideways, her feet go one behind the other along his body.
3. **The foot site is at the sole bottom** (0.05 mm), not 10.3 mm above it as an
   older note said.

## 7. What was tried before, and why it failed

| Approach | Result | Why |
|---|---|---|
| Static stepping (IK + gravity feedforward) | Abandoned | She cannot stand on one foot |
| `alpha_walking` + steering, forward | Fell 6/6 | Too wide for his back |
| Belly crawl, about 3000 searched gaits | Best 14 of 39 cm | Nothing to push against on his narrow, smooth back |
| RL on his baked body, `Mjlab-Bridge-Sideways-MicroDuck` v1-v11b | Never crossed | Reward farms (lean and dive, march in place); scrambled bake in v1-v8 |
| Scripted look-ahead controller (`scripted_policy/`) | 0 of 103 pass the strict rules | Searched commands never kept her upright past his hips |
| Old walker v4 + steering, after reshaping the set | 0 of 21 on the replica | Height changes on a narrow path |

## 8. Bugs and lessons

- **Collision filter (fixed 2026-09-26).** Collision bits were set after compile,
  so MuJoCo's per-body filter kept the thighs and neck of both ducks from ever
  colliding. His neck rested 16.8 mm inside the far shelf. Now `world.arm()` sets
  the bits on the spec before compile; test `Armed` locks it. Servos, thigh
  plates and ankle brackets were also made solid.
- **Bake frames (fixed 2026-09-26).** The first bake saved compiled mesh frames,
  so RL runs v1-v8 trained on a scrambled brother. Test `Baked` locks it.
- **Rewards that invite cheating.** Paying for world +x speed paid leaning,
  rocking and diving. Event rewards are scaled by dt (a -5 fall cost was
  really -0.1). Marching in place earned more than crossing. The fixes are in
  section 5.
- **Motor delay mismatch.** See section 5.
- **Film camera.** A camera that switched its target when she started walking
  made the video jump. The clean-film camera now always aims halfway between her
  and the middle of the bridge.

## 9. Next steps

- A v2 run: keep 30-50% of the his-back envs at full difficulty without
  demotion, and reward a flat sole at touchdown. This targets the edge failures.
- Then sim-to-real: train with the robot's motor delays again.

## 10. How to run

```bash
# tests
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_human_bridge
OPENBLAS_NUM_THREADS=1 uv run --with pytest pytest src/microduck_lab/tests/test_sideways_beam_cfg.py src/microduck_lab/tests/test_sideways_cfg.py -q

# rebake his body after changing PLANK or the set
.venv/bin/python -m microduck_lab.tasks.human_bridge.rl_policy.bake

# train the beam walker (A4500; writes logs/rsl_rl/velocity_sideways_beam/<date>_sideways-beam-v1)
CPUS=4 bash local_storage/hb_dev/rl_policy/s4_fix/launch_beam.sh performance 4096

# export a checkpoint (use --checkpoint-file: --checkpoint N picks the warm-start folder)
.venv/bin/python scripts/export.py Mjlab-Velocity-SidewaysBeam-MicroDuck \
    --checkpoint-file logs/rsl_rl/velocity_sideways_beam/<run>/model_2000.pt --onnx-file out.onnx

# score it: 5 replica + 3 real-story attempts, strict rules, a clip each (GPU debug node)
sbatch -M cluster -p debug --gres=gpu:1 --cpus-per-task=16 --time=00:30:00 \
    local_storage/hb_dev/rl_policy/s6_eval/eval_ckpt.sh 2000

# one scored attempt by hand (always pass --steer BEAM_LAW for beam-task policies)
MUJOCO_GL=egl .venv/bin/python local_storage/hb_dev/rl_policy/bench/story_steer.py \
    --bake src/microduck_lab/tasks/human_bridge/rl_policy/bridge_pose.json --seeds 3 \
    --policy out.onnx --steer BEAM_LAW --video 'videos/human_bridge/rl_policy/try_{n:02d}_s{seed}.mp4' --out res.json

# the clean film (she starts 1 s after he lands, smooth camera, 1080p)
MUJOCO_GL=egl .venv/bin/python local_storage/hb_dev/rl_policy/s6_eval/clean/clean_film.py \
    --policy out.onnx --settle 1.0 --start-x -0.23 --video videos/human_bridge/rl_policy/<new>.mp4 --out res.json
```

Videos are never overwritten: every attempt gets a new numbered name.
The bench tools are documented in `local_storage/hb_dev/rl_policy/bench/README.md`.
