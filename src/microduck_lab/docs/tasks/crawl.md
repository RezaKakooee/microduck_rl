# Two crawls

Both use the same flat-grass scene, BAM XL330 M6 motors at 7.4 V, 50 Hz
commands, 2 ms physics steps, and 25 cm non-colliding distance stripes.
Distances below are **net horizontal trunk displacement**, not accumulated
back-and-forth motion. Both clips run for 20 seconds at real speed.

| Gait | Leg motion | Distance | Speed | Trunk mean / max |
|---|---|---:|---:|---:|
| Original `expert.py` | Sines, legs half a cycle apart | 3,371 mm | 168.6 mm/s | 46.9 / 86.7 mm |
| Baby `baby.py` | Both legs tuck and drive together | 537 mm | 26.8 mm/s | 60.7 / 88.5 mm |

Baby clip: `videos/crawl/duck_crawl_baby.mp4` — 1280×720, 25 fps, H.264,
2.5 MB, `+faststart`. Original: `videos/crawl/duck_crawl.mp4`.
The baby crawl is slower but preserves the synchronized fold/extend motion
seen after 1:46 in `videos/swing/duck_swing_2min.mp4`.
It advances 534 mm along negative world x, with 60 mm lateral drift. Its
second-half speed is 29.8 mm/s, so the result is sustained travel, not just
initial settling. It rocks back during parts of each cycle.

The unmodified swing poses, tested from the same flat-ground start, achieve
only 55 mm in 20 seconds (2.7 mm/s; second half 0.35 mm/s). This is a different
initial state from the swing landing's previously measured roughly 1 mm/s.

## Search and selected motion

Seed 7, cross-entropy: **8 rounds × 40 candidates × 8 seconds**, ranked on
net distance. Search varies each pose's hip, knee, ankle, neck and head,
and each pose's dwell time. Pose bounds preserve tucked versus extended legs.
The best eight valid candidates update each round's distribution. The top
12 individual candidates are then measured for the full 20-second clip;
the best sustained result is saved, rather than an untested distribution mean.

The selected cycle holds tuck for 0.865 s and drive for 0.612 s:

| Joint (radians) | Tuck | Drive |
|---|---:|---:|
| Left hip pitch | -0.394 | -1.454 |
| Left knee | 1.393 | -1.471 |
| Left ankle | 0.263 | 0.005 |
| Neck pitch | -0.650 | 0.194 |
| Head pitch | -0.556 | -0.368 |

Right leg targets are always the negative of the left, with no phase offset.
Every physics step checks trunk height ≤200 mm, absolute upright cosine ≤0.70,
and lateral-axis vertical component ≤0.60 to reject standing and side rolls.
The rendered run's maximum absolute upright cosine is 0.247. Peak motor torque
is 0.641 Nm, at the existing current limit. No external forces are added.

## Reproduce

From `microduck_rl`, search and render on an EGL GPU node:

```bash
sbatch -M cluster src/microduck_lab/tasks/crawl/baby.sbatch
```

Evaluate the saved `crawl_baby_gait.npy` without rendering, or run the physics
regressions (including actual leg synchronization and posture rejection):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.crawl.baby
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_crawl_baby -v
```

The search history, full-duration finalist results, render report, trajectory,
and contact sheet are `videos/crawl/crawl_baby_*` and
`videos/crawl/duck_crawl_baby_sheet.jpg`. Search, evaluation and rendering
share one rollout function. The original `crawl.py`, `expert.py`,
`crawl_gait.npy`, and `duck_crawl.mp4` were verified byte-for-byte unchanged.
