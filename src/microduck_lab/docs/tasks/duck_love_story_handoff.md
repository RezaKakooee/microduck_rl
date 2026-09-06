# Duck love story — handoff

Written 2026-09-05. Three films exist and **all three must be kept**. Two are
finished and rendered. The third is finished except for its last beat, which is
described in "Where it stopped" below, with the measurement that blocks it and
three ways to finish.

## The three films

| film | script | state |
|---|---|---|
| `videos/love_story/duck_love_story.mp4` (66 MB) | `src/microduck_lab/tasks/love_story/original.py` | the original policy-driven film. Rendered, kept as reference. Its `Agent` class is the shared duck driver used by all three. |
| `videos/love_story/duck_love_story_physics.mp4` (58 MB) | `src/microduck_lab/tasks/love_story/dispenser.py` + `film/physical_stage.py` | actuator-only film with the mechanical egg dispenser. **Rendered, complete: 15/15 phases, 111.9 s, success.** Report `videos/love_story/duck_love_story_physics.json`. Its 4 tests pass. |
| `videos/love_story/duck_love_story_v2.mp4` (103 MB) | `src/microduck_lab/tasks/love_story/v2.py` + `film/scene_v2.py` | the first film's story and scene on the physics rules, **17/17 phases, 150 s, rendered and complete**, ending with both parents at the nest over their chicks. |

Older experiments kept but not maintained: `film/film.py` (posed rig),
`film/physics.py` (puppeteered physics).

```bash
uv run python -m microduck_lab.tasks.love_story.v2 --dry --trace          # no GPU, ~3 min
bash src/microduck_lab/render/render.sh film-v2
bash src/microduck_lab/render/render.sh film   # the dispenser film
OPENBLAS_NUM_THREADS=1 uv run --with pytest pytest src/microduck_lab/src/microduck_lab/tests/test_love_story_physics.py
```

Always prefix with `OPENBLAS_NUM_THREADS=1`. A dry run is ~3 minutes, a GPU
render ~4 minutes.

## The rule all three obey

After initialisation the director may write **nothing but `data.ctrl`**. A
guard in `Story.step` raises if any position, velocity, collision mask, colour
or applied force changes another way, and the model has no mocap bodies and no
equality constraints. Everything that moves is an actuator on a joint. The
adults are driven by the pretrained ONNX policies (walking / standing /
sit-stand / ground-pick) closed loop, exactly as `task_kick_ball.py` and
`task_pick_up.py` do.

## The ending, and what it does and does not show

The last three beats are in: `gather` (both parents come to either side of the
nest), `raise_cradle` (a narrow cradle inside the nest lift carries the two
open shells 100 mm up to the parents' heads) and `kiss_the_kids` / `finale`
(both lower their heads over their own chick while the chicks wriggle and
peep). The recorded run completes all 17 phases in 150 s with nobody falling.

**The beaks do not touch the chicks.** `duckling_gap_mm` in the report is the
closest each beak got: 71.5 and 81.3 mm. That number is in the report on
purpose. Getting the last 70 mm was tried five ways and each one is written
into the code comments where it was tried:

* bowing to them (ground-pick): the beak lands at the right height, but the
  bow shuffles the duck 20-100 mm in a direction that varies;
* leaning to full stretch: the head meets the folded lid or the shell and
  presses the whole egg over, tipping the chick inside (0.99 -> 0.69 upright);
* standing closer than 105 mm: a chest touches the egg, same result;
* folding the lid to its stop (-2.79 rad): held hard against the stop it has
  nowhere to give, goes past its limit as the cradle accelerates and levers
  its own shell. It is held at -2.45 now;
* a chick raising its head to meet them: a passive chick that lifts its head
  tips itself over, with nothing near it. The curled pose is the only one it
  holds.

To close it properly the shell would have to fold away completely, or the chick
would need a policy of its own.

## What the policies and MuJoCo forced — do not re-derive these

Every one cost a run. They are the reason the scripts look the way they do.

**Positioning.** These policies cannot be placed accurately.

- A turn command under ~1.0 does nothing; a proportional controller parks the
  duck in that deadband forever. Turns are bang-bang with a floor (`turn_cmd`).
- Turning on the spot shuffles the duck 5-13 cm. Every approach is routed to
  arrive already facing the right way.
- Told to stop at walking speed, a duck coasts 30-90 mm, and the amount varies.
  Stop commands go out well before the mark.
- Timed forward bursts are worse in both directions: 0.5 s barely moves it (the
  gait takes that long to start), 1.0 s carried one 130 mm across the nest.
- Reverse walking is slow (2-3 cm/s), drifts ~40% sideways, and needs its
  heading held; it lands within ~50 mm. It is how a parent gets onto the nest.
- `Agent.drive_to` has an opt-in `loose_aim` (v2 sets it): near a target the
  aim angle is mostly noise, which sent a duck into a 20 s turn-shuffle loop.
  **It is off by default so the two earlier films keep the approach they were
  recorded with — do not turn it on globally.**
- A bare `parent.vel(0.3)` does not steer. After a route both parents were
  facing north and walked off the set. Use `drive_to`.

**The ducks cannot step over anything.** A 3 mm nest rim, an 8 mm dais and one
straw standing 1.5 mm proud each put a duck on the floor. Everything they walk
near is flush or off their line.

**Reach.** Standing, a beak bottoms out at z = 0.15; leaning changes it by only
a few mm. A chick in its shell sits at z = 0.04 and cannot lift its head (the
shell holds it at 0.036 whatever the neck servo is told). That gap is why the
cradle exists.

**The kiss between the adults** is beak-on-face, not tip-to-tip: every lean
that was tried has the top head shells meet first, tips 29 mm apart at best.

**The proposal is a sit, not a kneel.** No policy on this robot puts one knee
down and nothing may be posed by hand. The ground-pick bow was used first and
reads as a collapse: the beak goes to the floor and the body folds. The
sit-stand policy's sit is a trained, deliberate motion that lands him low in
front of her with his head up.

**The ring stays on its dais.** The jaw's convex collision hull is bulkier than
the ring's bore; carried, it slides off within two seconds.

**Ground-pick (the bow)** puts the beak at z = 0.03-0.06, 81 mm ahead — the
right place to kiss a chick at nest level — but it shuffles the duck 20-100 mm
in a direction that varies, so contact is a coin toss. Repeated bows do not
converge.

**MuJoCo traps, both silent.** A mesh only gets a convex hull, and a static
geom only enters the world's broadphase tree, if it can collide *at compile
time*; switch `contype` on later and the geom is simply not there (planes
bypass the tree, which hides it). And the two halves of one egg overlap along
the zigzag, so they need `<contact><exclude>` or the weld fires the egg out of
the nest.

**Masses.** Eggs are 60 g bottom-weighted: at 8 g a parent nudging one knocked
it out of the nest, at 200 g the lifts could not raise it. Balloon hearts (free
bodies with gravity compensation) went unstable within a second every time and
are now scenery on stakes.

## Scene layout (`film/scene_v2.py`)

Ground is four box slabs around a 115 mm pocket (a plane cannot have a hole).
Nest lift raises the eggs from the pocket to nest level; a narrow cradle inside
it raises them another 110 mm at the end. Guide fingers and columns ride with
the lifts. Rims are sunk flush. The dais, the ring and four heart stakes are
off every walking line. Her bow and his bow tie are geoms fixed to the robot
bodies at build time.

`scene_v2.py` was rewritten from scratch once after a bad `str.replace` in a
patch script corrupted it — when editing these files, anchor replacements on
unique strings and check `ast.parse` after.

## Camera

`Story.camera` in each film. v2's was tuned twice: the ducks are 25 cm tall and
the first pass read as two dots on a lawn. Shots are 0.42-0.95 m out at -4 to
-16 degrees. Two specific fixes worth keeping: the hatch is shot over the
father's shoulder from above (58 az, -38 el) because he sits between the eggs
and a southern camera, and her exit is shot from the south-east because it runs
straight at a southern camera and fills the lens.
