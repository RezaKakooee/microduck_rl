# Human bridge — a brother lies across a gap, his sister crosses on his back

**Dates:** 2026-09-25 to 2026-09-26.
**Status:** NOT solved. He lies down as a bridge reliably. She has not yet
**walked** across him. The one "success" (`scripted_try96`) was a fall across,
not a walk.

Based on the viral clip of a boy lying across a gap outside an apartment so his
little sister can walk over him.

## 1. What works

| Part | Where | Result |
|---|---|---|
| Two-duck world, BAM XL330 motors, physics-only guard | `tasks/human_bridge/world.py` | `World.step` refuses external forces and model edits after `start()` |
| The set | `tasks/human_bridge/scene.py` | Her ledge, a 35 mm lower notch where he stands, a 100 mm gap (a real drop), a notch shelf for his head, the far ledge |
| He lies down | `brother.py`: `LieDown`, `PLANK` | One joint-target blend. He falls forward; soles stay on the step; head lands on the far shelf |
| He holds her weight | `brother.py`: `HoldStraight` | Integral on his joint targets. A 757 g load anywhere on his back: at most 5.5 mm sag |
| The story + video | `story.py` | Phases wait → he_lies → cross → across. Strict "walked" checks (section 4) |
| Frozen copy of his body for RL | `bake.py` → `bridge_pose.json` | Matches the simulated brother to 0.00 mm (test `Baked`) |

Video of him lying down: `videos/human_bridge/rl_policy/brother_lies_down.mp4`.

**PLANK (his bridge pose):** legs straight, ankles -/+1.57, head_pitch -1.2
(head folds back like a person in a plank; without it a 60 mm dip over the
neck), hip roll -/+0.38 (legs pressed together; otherwise a 64 mm gap between
his legs on his centre line).

**Heights on his centre line (the corrected bake):** her ledge 0.329 m, his legs
and trunk rise to 0.349, head 0.343, far ledge 0.315 (a 28 mm step down).

## 2. Hard limits measured on this robot

1. **She cannot stand on one foot.** Legs have no ankle roll; hip roll stops at
   0.38 rad. Over one flat foot her centre of mass gets 1-5 mm inside the sole
   at best. So slow foot-by-foot stepping is impossible; she only walks
   dynamically.
2. **Forward walking cannot work on him.** `alpha_walking` needs a path at least
   115-120 mm wide and climbs at most 15 mm. His trunk is 64 mm wide. 6 of 6
   runs fell off.
3. **Sideways is the better fit.** Her feet go one behind the other along his
   length; each 54 mm sole lies across his back. She stands facing -y, so her
   left (+vy) points across the gap (+x).

## 3. What was tried for her, and why it failed

| Approach | Result | Why |
|---|---|---|
| Static stepping expert (IK + gravity feedforward) | Abandoned | Limit 1 above |
| `alpha_walking` + steering | Fell 6/6 | Limit 2 |
| Belly crawl, CEM-searched gaits (3 families, ~3000 tries) | Best 140 of 390 mm | No grip on his narrow smooth back |
| RL fine-tune `Mjlab-Bridge-Sideways-MicroDuck`, v1-v11b | Never walked across | See below |
| Scripted look-ahead controller (other agents, `tasks/human_bridge_scripted/`) | try96 "success" = a fall across | Old checks too loose |

### The RL runs (`rl/microduck_bridge_sideways_env_cfg.py`)

Fine-tunes a sideways walker on a frozen copy of his lying body. 61D obs; the 6
body-pose command slots carry bridge state (position along him, offset,
heading error, height ahead, sink) — a simulation-only expert.

| Run | Change | What she did |
|---|---|---|
| v1 | first version | Learned to stand still on her ledge |
| v2 | pay for +x speed; route spawns | Unlearned walking in 250 iterations |
| v3 | progress weight 40, finish bonus | Rushed, fell off his side |
| v4 | bridge state in the body-pose slots | Walked 22 s on him, then fell |
| v5 | from stepping sideways v4 | Stopped 50-90 mm before her ledge edge |
| v6-v8 | ledges raised to "ankle servo" height | Put him in a HOLE. Wrong: see bug |
| v9 | corrected bake | Marched in place at the edge (64/64 time-outs) |
| v10 | per-step "alive" pay removed | Still marched in place |
| v11/v11b | fresh start, full exploration noise | Steps onto his legs, falls there (64/64), flat from iter 2750 to 5000 |

**Big bug, fixed:** until the fix, `bake.py` saved each mesh's COMPILED frame;
MuJoCo re-centres meshes again when it compiles the training scene, so every
bridge run v1-v8 trained on a scrambled brother (thighs 56 mm off, soles 50 mm
high, neck 77 mm low). That created a fake "48 mm ankle-servo wall". Fixed by
saving the authored frame (`mesh_offset`); test `Baked` locks it.

## 4. The success check (read this before claiming success)

`story.py` now fails a crossing unless she WALKS:

- trunk upright >= 0.9 (tilt <= ~25 deg) during the whole crossing;
- only her soles touch him or the ledges (knees, body, head = fail);
- all of her ends past the far ledge edge (`rearmost`), standing.

The old checks (fall = tilt past 60 deg) let `scripted_try96` pass: she tipped
over onto him at 6.4 s, lay across his head and the far ledge, and got up
there. The scripted agents' runner (`human_bridge_scripted/run.py`) has its own
loose checks and was not changed.

## 5. The sideways walking skill (made for this task)

`Mjlab-Velocity-Sideways-MicroDuck`, `rl/microduck_velocity_sideways_env_cfg.py`.
Eval: `tasks/walking/sideways.py` (infer_policy path: scene.xml + BAM).

| Version | Result |
|---|---|
| v1 | Strafes 170 mm/s both ways, but turns 40-110 deg per 6 s |
| v2 | Turns less, but SLIDES its feet |
| v3 | Still slides: swing peaks 1.7-3.4 mm of real lift |
| v4 | Steps: swing peaks 10-20 mm; still turns 20-80 deg per 6 s |

Lessons: the recipe never learned to strafe because hip-roll std 0.05 rad
punished side steps and its loose velocity reward paid 67% for ignoring vy.
The foot site sits 10.3 mm above a flat sole, so "swing height" targets must add
that. The recipe's slip cost is squared, so slow slides were free.

## 6. Ideas not yet tried

- Make the gap wider so falling across is impossible (his length allows maybe
  150-180 mm; check his support and load again).
- Scripted controller with the strict checks as its objective.
- RL with the strict checks as termination (tilt > 25 deg ends the episode).
- Curriculum on the bridge: a wide flat plank first, then his real body.

## 7. How to run

```bash
# the story (needs a bridge ONNX; exported by scripts/export.py)
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.human_bridge.rl_policy.story --policy <bridge.onnx>
sbatch -M cluster local_storage/hb_dev/rl_policy/from_first_chat/video.sbatch -m microduck_lab.tasks.human_bridge.rl_policy.story \
    --policy <bridge.onnx> --video videos/human_bridge/<new_name>.mp4
# rebake his body after changing PLANK or the set
.venv/bin/python -m microduck_lab.tasks.human_bridge.rl_policy.bake
# tests
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_human_bridge
OPENBLAS_NUM_THREADS=1 uv run --with pytest pytest src/microduck_lab/tests/test_sideways_cfg.py -q
```

Dev helpers (not in git) in `local_storage/hb_dev/rl_policy/from_first_chat/`: `train.sbatch` (training),
`video.sbatch`, `evalbridge2.sh` (export + story + training-sim rollout, with
videos), `train_sim_rollout.py` (64 envs from her ledge in the training sim),
`evalside.sh` + `feet.sh` (sideways foot lift / slip). Checkpoints:
`logs/rsl_rl/bridge_sideways/`, `logs/rsl_rl/velocity_sideways/`.

Videos never overwrite: every attempt gets a new name in `videos/human_bridge/`
or `videos/sideways/`.
