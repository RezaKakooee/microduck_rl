# Duck love story: physical mechanisms and actuator control

The corrected film is `videos/love_story/duck_love_story_physics.mp4`. The supplied
`videos/love_story/duck_love_story.mp4` is preserved as the original reference.

```bash
uv run python -m microduck_lab.tasks.love_story.dispenser --dry --trace
bash src/microduck_lab/render/render.sh film
OPENBLAS_NUM_THREADS=1 uv run --with pytest pytest src/microduck_lab/src/microduck_lab/tests/test_love_story_physics.py
```

The adults meet, bow and nod, kiss beak-to-face through actual head contact,
walk around the nest, and take turns sitting beside a mechanical egg
dispenser. Its two trays withdraw, letting preloaded eggs drop onto the nest
under gravity. While she sits with the eggs he comes to her side and rests his
head on hers (measured contact). Motors open the shells; the ducklings uncurl
through servo torques. Both parents then watch them.

Two beats were checked against the physics before being written:

- **The ring stays on the dais.** Carrying it in the beak was tried: the jaw's
  convex collision hull is bulkier than the ring's bore, and the ring slides
  off within two seconds even standing still. There is no grip to hand it over
  with.
- **The kiss is his jaw on her face, not beak tip to beak tip.** Every lean
  that was tried — symmetric, one high and one low, necks stretched — has the
  top head shells meet first, with the beak tips 29 mm apart at best. The
  closest real contact is his jaw on her head shell at 17 cm, which is what
  the nuzzle now does (he goes low and down, she high and up).
- **The head kiss is closed-loop on the contact.** He is routed along her side
  so he arrives facing her head; if the lean is at full stretch and still
  short, he takes one short step in and leans again (never needed in the
  recorded run). A duck stalled within 12 cm of its mark now stops there:
  backing it off blindly once reversed him into the nest and onto an egg.

## What changed

The original director used ONNX locomotion but also teleported the ring and
eggs, repeatedly reset hidden ducklings, disabled collisions, and kicked shell
lids with explicit velocity changes. The earlier claim that everything in
that film was physical was incorrect.

The replacement uses:

- **Actuator controls only after initialization.** Four MuJoCo steps per
  50 Hz controller tick, with gravity, inertia, friction and contact response.
  Guards reject controller edits to positions or velocities, changes to
  collision masks or opacity, and externally applied forces. No mocap bodies
  or equality welds exist in the new model.
- **Existing ONNX adult skills.** Walking, standing, bowing and sitting retain
  the policy observation and joint conventions. Adult servo torque is capped
  at the existing XL330 current limit. Scaled ducklings use bounded joint
  servos; they do not have a newly trained walking policy.
- **A visible mechanical laying device.** Two sliding trays, rail carriages,
  guide sleeves and a fixed frame are modeled geometry. Prismatic joints
  represent the rail bearings, with 0.5 mm carriage clearance. Each tray
  motor is limited to 2 N. The eggs are free bodies throughout; sleeves keep
  them upright while the support slides away.
- **Hollow, collidable mechanical eggs.** Each shell half consists of 64 thin
  convex panels. The ducklings exist inside from the initial state, opaque
  and collidable. A visible hinge and a motor limited to 0.0025 Nm open each
  lid. No egg spawning, hidden duckling placement, opacity fade, collision
  enabling, imposed rocking torque, or lid velocity kick occurs.
- **Solid scenery and a real gift.** The nest has a thin solid mat and an
  open entrance. The parents navigate around its hardware. The ring rests
  on a presentation stand; it is not transferred onto a neck without a
  gripper. Floating hearts and moving mocap decorations are removed.
- **Measured completion.** Head contact (the kiss and the head kiss), actual
  tray travel, at least 10 mm of egg descent, actual lid angles, and upright
  actors gate completion.
  A timeout or fall fails the run and returns a nonzero exit code.

MuJoCo normally uses convex hulls for mesh collision. A single mesh shaped
like a bowl therefore does not provide its visible hollow interior. The
compound wall panels follow the approach described in the official
[MuJoCo collision documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#collision-detection).

## Files and evidence

- `src/microduck_lab/tasks/love_story/dispenser.py`: state-based director, runtime guards,
  cameras, rendering and JSON report.
- `src/microduck_lab/film/physical_stage.py`: physical props, mechanisms and actor
  attachment. Reuses the source robot CAD and coloring helpers.
- `src/microduck_lab/src/microduck_lab/tests/test_love_story_physics.py`: real release/hatching regression,
  permanent collision checks, teleport rejection and timeout failure.
- `videos/love_story/duck_love_story_physics.json`: completion events, descent, lid
  angles, upright measurements and worst contact penetration for the video.

`render_duckfilm.sh` selects the corrected film by default. Its `legacy`,
`posed`, and `puppet` modes explicitly select the earlier implementations.
The new director reuses only `Agent` and static pose helpers from the original
module; its old story logic and prop updater are never run.

## Validation of the recorded scene

The complete run finishes all 15 phases in 111.9 simulated seconds. Both
adults and both ducklings remain upright. Nuzzle contact is measured at 12.8 s
and the head kiss at 56.2 s (he arrives 126 mm from her head). The two eggs
drop 11.87 and 11.66 mm, and the final lid angles are -2.03 and -2.06 radians.
All four CPU regression tests pass. The GPU recording reproduces the same
completion events and physical measurements as the headless run.

The largest reported soft-contact penetration is 3.2 mm during a foot-floor
impact. This is a measured limitation of the compliant contact model, not a
collision bypass. These results validate the recorded layout; they are not a
success-rate estimate over randomized stories.

## Physical scope

This is a simulated robot story with a mechanical nest dispenser and hinged
capsules, not biological egg laying or fracture simulation. The chicks remain
supported inside their open shells. The affection scene is a head nuzzle;
head contact is not labeled as beak-tip contact — the beak tips cannot pass the
head shells on this robot, as measured above.

Bodies are placed once to initialize the simulation. After that, camera
motion changes only the view. Fixed scenery and rail joints represent
anchored structures. The original CAD collision representation, including
filtering between overlapping parts within an individual robot, is retained;
all external actors and props collide. MuJoCo uses compliant contacts, so the
report records penetration rather than claiming infinitely rigid contact.
The scaled chicks and custom dispenser have not been validated on hardware.
