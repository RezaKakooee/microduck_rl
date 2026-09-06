# 04 — Six RL attempts at a one-legged spiral

**Date:** 2026-08-31 to 2026-09-03. Six runs on rollers, about 30 GPU-hours.
**Goal:** the figure-skating spiral. Glide on one leg with the other leg
extended straight out behind, like the photo of the skater.

The task is `Mjlab-Spiral-Flat-MicroDuck` in
`src/microduck_lab/rl/microduck_spiral_env_cfg.py`, built on the roller
recipe. Every version is still in that file, with the reason it changed.

The short version: all six failed, and the reasons were different each time.
File 05 shows the failure is physical. This file is about the reward-design
mistakes, because they are the educational part.

---

## 1. The sign of the hip

Before writing a reward we measured which way the hip extends the leg. The
legs mirror each other and both extend *backward* opposite to their home sign:

| Leg | hip_pitch | Foot x (forward) | Foot z |
|---|---|---|---|
| left | −0.458 (home) | +21 mm | −133 mm |
| left | **+1.2** | **−113 mm** | −25 mm |
| right | +0.458 (home) | +21 mm | −133 mm |
| right | **−1.2** | **−113 mm** | −25 mm |

So the target is +1.1 for a free left leg and −1.1 for a free right leg. Get
that backwards and you train a knee-to-chest tuck that scores the same on every
other term. A test asserts it against the real model.

## 2. Run v1: a walk that looked like a spiral

Rewards added: `spiral_free_leg` (free leg extended) and `single_support_hold`
(a ramp on time with one foot up). `skating_air_time` set to 0, because it
pays per swing and drives swing *frequency*.

The training curve looked like success. The rollout measured:

| Metric | Value |
|---|---|
| single support | 92 % of the time |
| free-leg hip | +0.99 rad against +1.10 target |
| mean hold on one foot | **0.11 s** |
| support-foot swaps in 13 s | **108** |

It was a walk. `LLLLLRRRRRRLLLLLL…`.

**Metric trap 1:** "single support 92 %" means nothing. A *walk* is single
support most of the time, because it alternates. The number that matters is
how long it holds the *same* foot.

**Metric trap 2:** the free-hip number scored well because a walking swing
passes through a backward-extended hip. The pose term was paying for walking.

**And a term was fighting the task:** the roller recipe's `gait_symmetry`
penalises "lopsided left/right foot usage" at −1.0. That is the definition of a
one-legged spiral.

## 3. Run v2: the topple exploit that was not one

Near the end of v1 the pose reward jumped from 0.06 to 2.14 while wheel speed
went to zero and falls rose 11×. We read it as an exploit — the forward gate
`_forward_progress_gate` measures *trunk* velocity, and a duck toppling forward
has plenty — and added two gates: wheels actually turning, and upright.

The gates were reasonable. The diagnosis was wrong. The spike was the walk
appearing, and we had not rolled the policy out before deciding. Lesson:
never diagnose from the reward curve alone.

## 4. Run v3: a latch nothing could reach

To exclude the walk we put a **latch** on both terms: below a minimum hold
time on one wheel, they pay exactly zero. Ramped 0.30 → 1.5 s.

Result: both terms read **0.0000 for all 10000 iterations**. The policy never
once held one wheel for 0.30 s, so it never saw the reward. The terms were dead
weight. It trained the roller stack only and ended with both wheels down.

v1 was too easy (a walk satisfied it). v3 was too hard (nothing reached it).
Both are the same mistake: a conjunction of hard gates is binary. There is no
gradient between farmable and unreachable.

## 5. Run v4: hand it the pose

The standard fix for "the frontier is never sampled" is a reverse curriculum:
spawn some episodes already in the pose. 70 % at first, decaying to 0. And we
measured the latch bound properly: held open-loop, the pose survives 0.38 s
before the free foot drops, so the latch started at 0.20 s.

Result: pose term still 0.0. Hold term peaked at 0.008. Even handed the pose
for free, it could not exploit it.

## 6. Run v5: two-layer pose tracking

Two reference repos changed the approach. `jonathanhawkins/microduck-lab` uses
a DeepMimic-style imitation reward with a **two-layer Gaussian**, and its
comments record our exact failures ("the joint angles of a run cycle are just
as matchable lying face-down on the floor"). `Vottivott/microduck-playground`
trained stilt walking with a curriculum on the *physics*, not the reward.

The reward became a two-layer match to a static reference pose `q_ref`:

```
d²    = Σ_j ( q_j - q_ref,j )²                 sum over the 14 joints
r     = ½ · exp( -d² / σ_wide² ) + ½ · exp( -d² / σ_tight² )
```

Best-of over the two mirror variants, so either leg can be the free one. The
wide layer keeps a gradient when the body is nowhere near the pose; the tight
one pays for precision.

**What happened:** it never lifted a foot. We had set `σ_wide = 1.8` by eye.
Computed afterwards: the squared error between *standing* and the pose is
2.43, so standing scored `½·e^(−2.43/3.24) ≈ 0.237` of a maximum 1.000. Only
4.2× worse than a perfect spiral, for zero fall risk. Doing nothing won again.

## 7. Run v6: the std, computed this time

| σ_wide | σ_tight | standing | in pose | ratio |
|---|---|---|---|---|
| 1.8 | 0.60 | 0.237 | 1.000 | 4× |
| 1.2 | 0.40 | 0.093 | 1.000 | 11× |
| **0.9** | **0.30** | **0.025** | **1.000** | **40×** |
| 0.7 | 0.23 | 0.004 | 1.000 | 283× |

At 0.9 / 0.30 the climb is still smooth: 0.025 → 0.093 → 0.237 → 0.507 →
0.867 → 1.000 at 0, 25, 50, 75, 90, 100 % of the way. A test now asserts the
ratio is above 20× and the climb is monotonic.

v6 was stopped before finishing, because the scripted-controller work in file
05 showed the target pose itself was wrong.

## 8. What the six runs teach

1. Roll the policy out before believing any metric. Twice we diagnosed from
   the curve and were wrong.
2. Measure the thing that distinguishes success from the nearest cheat. Here
   it was hold duration on the *same* foot, not single-support fraction.
3. A conjunction of hard gates is either farmable or unreachable. Use a smooth
   shaped reward with a gradient everywhere.
4. Compute the reward landscape before training. One line of numpy would have
   saved run v5.
5. Check that the target state is physically reachable. That is file 05.

---

## Appendix — numbers and commands

**Hold-duration measurement** — the metric that exposed v1. Run 15 s, label
each control step `L`, `R`, or `-` by which foot is up (site z > 15 mm), then
take run lengths:

```
v1  : swaps 108 | mean hold 0.11 s | longest 0.12 s
v3  : swaps   0 | both feet down
v5  : swaps   0 | both feet down, travel -0.14 m
```

Foot sites sit at 3–13 mm on the roller model, not tens of mm. A 45 mm
threshold reports 0 % single support for every policy.

**Reward landscape** — the numpy check that should have preceded v5 is in
`src/microduck_lab/src/microduck_lab/tests/test_spiral_cfg.py`, `test_standing_still_scores_far_below_the_pose`.

**Training**

```bash
uv run train Mjlab-Spiral-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 --agent.logger tensorboard --agent.run-name spiral-v5
# logs: duck-rl- (v1),  (v2),  (v3),  (v4),  (v5),  (v6)
```

**Moving a pending job to the RTX partition**

```bash
```
