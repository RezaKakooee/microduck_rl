# 03 — Training the duck to skate on ice

**Date:** 2026-08-29 to 2026-08-30. Three training runs, about 8 GPU-hours.
**Goal:** the same velocity-tracking task as walking, on a surface with almost
no grip.

The task is `Mjlab-Velocity-Ice-MicroDuck` in
`src/microduck_lab/rl/microduck_velocity_ice_env_cfg.py`. It builds on the
walking recipe so that the whole domain-randomisation and observation stack
stays in sync.

---

## 1. How MuJoCo combines friction

Before changing anything we checked a physics assumption, and it was worth it.

**Question:** if the floor is icy, is the contact icy?

**Test:** slide a 1 kg box at 2 m/s for 3 s.

| Floor μ | Box μ | Slid |
|---|---|---|
| 1.0 | 1.0 | 0.25 m |
| 0.02 | 1.0 | 0.25 m |
| 1.0 | 0.02 | 0.25 m |
| 0.02 | 0.02 | **5.12 m** |

MuJoCo uses the **maximum** of the two geoms' friction coefficients:

```
μ_contact = max( μ_geom1 , μ_geom2 )
```

So lowering the floor alone does nothing against the default μ = 1.0 feet.
Both must be lowered. The task pins the floor at 0.02 and randomises the feet,
so that the foot value is always the effective one. This rule is in
`src/microduck_lab/src/microduck_lab/tests/test_ice_cfg.py` as a test, so if MuJoCo ever changes it the test says so.

## 2. Two reward changes

**`foot_slip` → 0.** The walking recipe prices slip at −0.1. On ice slip is
not an escapable error; it is the medium. The repo's own rule (AGENTS.md) is to
price only the escapable part.

**`air_time`.** This term pays for feet in flight inside a window
`[t_min, t_max]`. It is what makes the walking gait step. We first softened it
from weight 3.0 to 1.0, reasoning that skating glides. That was wrong — see v1.

## 3. Run v1: it learned to stand still

Straight onto full ice, friction 0.04–0.18. 6000 iterations.

| Metric at the end | Value |
|---|---|
| peak foot swing height | 0.2 mm |
| velocity error | 0.46 m/s (commands are at most 0.3) |
| falls | 4 % |
| positive reward from `upright`+`pose`+`head_pose` | ≈ 5.5 of ≈ 7.2 |

Standing banked most of the positive reward for free, with zero fall risk. Any
attempt to push risked a fall and paid the action-rate tax. "Do nothing" was the
argmax. AGENTS.md describes exactly this: a tax on a skill that does not exist
yet makes doing nothing win.

In simulation the policy did not move at any command, on ice or on grip.

## 4. Run v2: a friction curriculum

**Change 1:** start at walking grip (0.7–1.3) and ramp down to ice over 4200
iterations. The curriculum function is a step function of the training step:

```
ranges(step) = ranges_k   for the largest k with  stage_k.step < step
```

It rewrites the `foot_friction` event's range. One catch: mjlab samples that
event once, at startup. It had to move to reset mode, or the curriculum would
write ranges that nothing ever read.

**Change 2:** `air_time` back to 3.0. Softening it removed the only term that
pays for moving a foot at all.

**What happened:**

| Iterations | Foot friction | Air time (mean) | Air-time reward |
|---|---|---|---|
| 500–2000 | 0.7–1.3 | 0.12 | 1.38 |
| 3200–4200 | 0.10–0.30 | 0.095 | 0.80 |
| 4200–7500 | **0.04–0.18** | 0.095 → 0.037 | 0.80 → 0.07 |

It walked fine down to friction 0.1, then **unlearned** stepping over the
3000 iterations on full ice, and ended where v1 did.

## 5. The checkpoint that skates

The final policy stands still. The checkpoint from **iteration 4000**, taken
just before the last friction stage, skates:

| Foot friction | Speed (0.3 commanded) | Fell |
|---|---|---|
| 0.30 | 0.275 m/s | no |
| 0.20 | 0.270 m/s | no |
| 0.12 | 0.216 m/s | no |
| 0.05 | 0.207 m/s | no |

That is better tracking than the vendored walking policy manages on dry ground
(0.13 m/s). The walking policy on the same ice falls at 1.9 s.

`ice_v2_iter4000.onnx` is the deliverable. Clips: `videos/ice_*.mp4`.

## 6. Run v3: it marched in place

We raised the floor of the ramp to 0.10–0.25, where v2 had still stepped.
The training curve looked healthy: air time stayed at 0.094. The exported
policy travelled **0.00 m**. It lifted its feet 6 mm and did not move.

Measured on the ONNX, a forward command shifts v2@4000's actions by 0.30 and
v3@7999's by 0.05. It had gone command-deaf. Air time is a misleading metric
for this task; a shuffle earns it without propelling.

**Lesson:** judge an ice run by velocity error and a rollout, never by air time.

## 7. Why full ice is physically hard

Real skating works because a blade is **anisotropic**: slippery along its
length, grippy across it. The skater pushes sideways while gliding forward. The
duck has flat rubber soles with the same friction in every direction. At
μ = 0.04 there is nothing to push against, so a non-propulsive shuffle is
genuinely the optimum. More training finds it more reliably.

MuJoCo can model a blade. A contact pair with five friction terms
`(slide1, slide2, spin, roll1, roll2)` gives directional sliding friction:

| Direction | Slid |
|---|---|
| along the blade | 0.17 m |
| across the blade | 3.61 m |

We did not build the skate version. Two things to know if someone does: the
pair's low-friction direction follows the contact frame, not world axes (the
first test came out inverted), and a `<pair>` overrides `geom_friction`, so the
friction curriculum would have to move to randomising pair friction.

---

## Appendix — numbers and commands

**Friction rule test** — in `src/microduck_lab/src/microduck_lab/tests/test_ice_cfg.py`,
`test_mujoco_combines_friction_by_maximum`.

**Training**

```bash
uv run train Mjlab-Velocity-Ice-MicroDuck --env.scene.num-envs 4096 \
  --agent.max-iterations 8000 --agent.logger tensorboard --agent.run-name ice-v2
# logs: the training log (v2),  (v3)
```

**Export a mid-run checkpoint**

```bash
uv run scripts/export.py Mjlab-Velocity-Ice-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity_ice/<run>/model_4000.pt
```

**Run on ice in CPU MuJoCo** — both surfaces must be slippery:

```bash
uv run python -m microduck_lab.tools.headless_rollout --walking ice_v2_iter4000.onnx --new-cmd-obs \
  --xml src/microduck_lab/models/scene_ice.xml --foot-friction 0.12 \
  --lin-vel-x 0.3 --seconds 15
```

**Command sensitivity of an ONNX** — feed a zero observation with and without a
forward command, compare the actions:

| Policy | max |Δaction| |
|---|---|
| ice_v2_iter4000 | 0.30 |
| ice_v3_iter7999 | 0.05 |
