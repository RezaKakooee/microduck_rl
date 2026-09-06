# Delivery task — pick up a cube, carry it to a target, drop it there

**Date:** 2026-09-03.
**Script:** `src/microduck_lab/tasks/objects/delivery.py`. **Scene:** `src/microduck_lab/models/scene_delivery.xml`.

## What it does

1. Walk to the cube and pick it up in the beak. This part is `task_pick_up.py`,
   reused as-is: same stand-off (81 mm), same stop window, same 3 s settle,
   same weld grip. Nothing about the pick was re-derived.
2. Stand up with the cube and steer to the target zone. The duck cannot strafe,
   so it aims at the spot where it must *stand* for the held cube to hang over
   the target centre. The held cube rides ~105 mm ahead of the trunk
   (measured, see below), so the stand spot is target minus 105 mm forward.
3. Stop when the held cube is within 60 mm of the target centre. Settle 2 s.
4. Re-check. If the cube is still inside 0.7 × radius: weld off, mouth open.
   The cube falls from the beak (~229 mm up). If not: back off 2 s and retry.
5. Wait until the cube is on the floor and still, then stand clear.

Success = cube at rest on the floor (centre z < 30 mm, speed < 20 mm/s),
within `--radius` (default 0.15 m) of the target centre, and the trunk never
below 70 mm (the walking fall height from `duck_sim.FALL_HEIGHT`).

Everything the summary prints is measured from the simulator state: the cube's
final position, its distance to the target, its height, its speed, and the
lowest trunk height over the whole run.

## Run it

```bash
cd $REPO
uv run python -m microduck_lab.tasks.objects.delivery --cube 1.0 0.5 --target 1.0 -0.5
# with video, on a GPU node:
MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg src/microduck_lab/tasks/objects/delivery.py --cube 1.0 0.5 --target 1.0 -0.5 --video videos/delivery/delivery_a.mp4
```

Options: `--cube X Y`, `--target X Y`, `--radius` (0.15), `--seconds` (90),
`--video`, `--cam-distance` (1.2), `--debug` (trace the held cube every 0.5 s).
`--walking` and `--ground-pick` default to the ONNX files in
`$POLICIES`. Exit code 0 on success, 1 on fail.

## Results (CPU, all with radius 0.15 m unless noted)

| Layout | Cube | Target | Picked at | Dropped at | Cube to target | Fell | Result |
|---|---|---|---|---|---|---|---|
| A: cube ahead-left, target ahead-right | (1.0, 0.5) | (1.0, -0.5) | 12.1 s | 22.9 s | **18 mm** | no | SUCCESS |
| B: cube behind, target ahead | (-0.9, 0.4) | (0.8, 0.3) | 40.6 s | 55.6 s | **26 mm** | no | SUCCESS |
| C: cube ahead, target behind | (1.0, 0.3) | (-0.8, -0.3) | 11.3 s | 29.7 s | **18 mm** | no | SUCCESS |
| D: cube straight ahead, radius 0.10 | (0.8, 0.0) | (1.8, 0.0) | never | never | 1.00 m | no | FAIL |

Details per layout:

- A: cube released 40 mm from centre, came to rest 18 mm from centre 0.3 s
  later. Final cube (+1.011, -0.486, z 15 mm). Min trunk z 74 mm.
- B: the pick took 6 tries (see limits). After that the carry went straight:
  1.68 m to the target, released at 47 mm, rest at 26 mm. Final cube
  (+0.780, +0.317, z 15 mm). Min trunk z 76 mm.
- C: the duck turned round holding the cube, walked 1.90 m, released at 23 mm,
  rest at 18 mm. Final cube (-0.804, -0.283, z 15 mm). Min trunk z 76 mm.
- D: the duck never got the pick to fire, so nothing was delivered. See limits.

The carry leg has not failed in any run so far: in A, B, C the first stop was
57-60 mm from centre, stopping and settling moved the cube by 11-37 mm
(mostly the coast after the zero command), and the drop itself moved it
another 5-22 mm. The 150 mm radius has a lot of margin; the
delivery was 18-26 mm in the three runs.

Videos: `videos/delivery/delivery_a.mp4` (layout A, 656 frames), `videos/delivery/delivery_c.mp4`
(layout C, 826 frames). Rendered on a GPU node; the numbers matched the CPU
runs exactly. Note: the shared `duck_sim.Recorder` captures at 25 fps and
writes at 30 fps, so every clip in `videos/` plays about 1.2x fast.

## Numbers measured while building this

- Held cube offset: while carrying, the cube sits 103-109 mm ahead of the
  trunk and bobs ±15 mm sideways with the gait. Standing still: 105 mm ahead,
  0-4 mm to the side, 227-229 mm up. `HOLD_X = 0.105`.
- Drop: from 229 mm up, the cube lands and stops within 0.3-0.4 s and moves
  5-22 mm horizontally while doing so.
- The 60 mm stop rule works because the duck walks in at ~2.6 mm per control
  tick, so it cannot step over a 60 mm window. Stopping and settling moves the
  cube by 11-37 mm (the duck coasts after the zero command, most in layout C).
- Backward walking: a -0.3 m/s command moves the duck about 1 cm in 2 s. The
  "stand clear" step is mostly standing still. It is harmless and the feet
  never touch the dropped cube (the cube is ~130 mm ahead of the trunk).
- Min trunk height in a successful run is 74-76 mm. That happens during the
  crouch of the ground pick, not during the carry. The fall threshold is 70 mm,
  so the margin there is only 4-6 mm. A real fall puts the trunk far lower, so
  the check still separates the two, but do not tighten the threshold.

## What failed, and honest limits

- **Layout D (cube straight ahead at 0.8 m) never picks.** The duck stops at
  89-90 mm from the cube, then rocks back to 95-98 mm while settling. The pick
  pocket ends at 95 mm. So it settles 0-3 mm out, backs off, and repeats: 10
  times in 60 s. `task_pick_up.py` does exactly the same at this cube position
  (checked, same log). This is a limit of the reused stand-off, not of the new
  code; fixing it means re-tuning the pick, which this task deliberately did
  not do. Layout B hit the same loop 5 times before it fired on the 6th.
- **The drop is a fall, not a placement.** The weld is switched off and the
  cube falls 21 cm. A gentler version would crouch with the ground-pick policy
  and release near the floor. Not done: the falling cube already lands within
  25 mm and the task did not need it.
- **The grip is still a weld** (from the pick-up task). No contact forces hold
  the cube; it cannot slip out during the carry, so "carried it without
  dropping" is not a test of anything.
- **Four layouts is not a success rate.** The carry leg was only tuned on
  layout A; B and C worked as-is. A sweep over cube and target positions is
  the honest next step. Expect more failures from the pick (layout D) than
  from the carry.
- **The duck reads the true cube and target positions** from the simulator.
  There is no perception.
- The delivery accuracy limit is the walking policy's deadband: it cannot make
  a correction smaller than about 6 cm, so anything under `--radius 0.06` is
  a matter of luck. All three successes would still pass at radius 0.05, with
  24-32 mm to spare.

## Files

- `src/microduck_lab/tasks/objects/delivery.py` — new.
- `src/microduck_lab/models/scene_delivery.xml` — new; a copy of
  `scene_pickup.xml` plus two non-colliding disc geoms (`target_zone`,
  `target_centre`). The script moves them to `--target` and sets the big
  disc's radius to `--radius`.
- `docs/tasks/delivery.md` — this file.
- `videos/delivery/delivery_a.mp4`, `videos/delivery/delivery_c.mp4`.

No existing file was changed.
