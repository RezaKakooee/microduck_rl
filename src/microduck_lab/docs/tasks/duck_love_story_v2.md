# Duck love story v2 — the first film's story, on the physics film's rules

`videos/love_story/duck_love_story_v2.mp4`. Two and a half minutes, seventeen beats, and the same
guarantee as `duck_love_story_physics.mp4`: after initialisation the director
may write **nothing but `data.ctrl`**. A guard raises if a position, a
velocity, a collision mask, a colour or an applied force changes any other way.

```bash
uv run python -m microduck_lab.tasks.love_story.v2 --dry --trace                 # no GPU
bash src/microduck_lab/render/render.sh film-v2
```

Both earlier films are untouched and still build and run:

| film | script | what it is |
|---|---|---|
| `duck_love_story_physics.mp4` | `task_love_story_physics.py` | the mechanical dispenser film |
| `duck_love_story_v2.mp4` | `task_love_story_v2.py` | this one: the first film's story and scene |
| `duck_love_story.mp4` | `task_love_story.py` | the original, kept as the reference |

## The story, and the mechanism behind each beat

| beat | driven by |
|---|---|
| he walks up to her | walking policy, drive-to-target |
| he sits down in front of her, beside the ring; she nods yes | sit-stand policy; head-pose command |
| they kiss, beak tip to beak tip, measured contact | standing policy, offset stance and turned heads |
| she loops round the nest and backs onto her spot | walking policy, forward then reverse |
| she settles | sit-stand policy |
| two eggs rise out of a pocket behind her | one lift motor |
| he comes to her side and rests his head on hers | standing policy, leaned in until contact |
| she broods; she leaves; he backs on and broods | walking (reverse) + sit-stand |
| the ducklings stir, then the shells open | duckling servos; two lid motors |
| he steps back and looks at them | walking policy |
| both parents walk in, one to each side of the nest | walking policy, drive-to-target |
| a cradle carries the two open shells up to head height | one cradle motor |
| each parent lowers its head over its own chick | head-pose commands, both sides |

## What each faked thing in the first film is made of here

* **The eggs** are in the world from t = 0, standing on a motorised lift in a
  115 mm pocket sunk into the ground behind where she sits. The lift raises
  them 102 mm to nest level. They are the physics film's hollow capsules: 64
  convex wall panels each, so the interior is really hollow, with the
  ducklings inside from the start and a motor on each hinged lid.
* **The hearts** are scenery on stakes. Balloon hearts were built first — free
  bodies with gravity compensation, in a box with a motorised lid. Two grams
  of thin extruded mesh sheared by a lid, and by each other, went unstable
  inside a second, every time.
* **The ring** rests on a dais. The beak cannot carry it: the jaw's convex
  collision hull is bulkier than the ring's bore and it slides off in two
  seconds, measured.
* **Her bow and his bow tie** are geoms fixed to the head and trunk bodies at
  build time, not props flown along by hand.
* **The nest** is a lining, two woven arcs of twigs, and scattered straw, all
  solid.

## What the mouth kiss cost everywhere else

Worth recording, because it is the real lesson of this film. Moving one beat
re-rolled the dice on every beat after it, and each of these was a separate
failed run:

| what broke | why | fix |
|---|---|---|
| she fell into the egg pocket standing off the nest | sitting down carries a duck 73 mm backwards, and backwards is the pocket; she sat 31 mm from a 12 mm hole | `SIT_BACKS_UP`: back onto a mark 40 mm short of the spot |
| he shuffled 240 mm from his mark for 45 s, touching nothing | the head kiss leaves his head yawed hard and his body leaning, and nothing reset either; he walked in a slow circle | `wake_walking`: clear the pose, walk forward for 1.2 s |
| he stalled turning to a waypoint 60 degrees behind him | he now finishes the head kiss beside her, not west of the nest | first waypoint straight south of wherever he is |
| a chick tipped as the cradle rose | the pose stepped 0.65 rad in one tick on a body that is only balanced | ease it in over half a second |
| a chick leaned over through the whole ending | +-0.12 rad of wriggle on a passive body | `KID_WRIGGLE` 0.05 |
| both chicks tipped, twice, with nothing near them | a stiffer nest lift (kp 200 -> 600) to close the pocket step. It closed it, and shook the eggs | reverted; the pocket is handled at the sit mark instead |

The last one is the one to remember. The stiffer lift looked like a clean fix
for a real problem -- the pocket was a 12 mm hole because the loaded lift sagged
-- and it worked. It also cost two chicks, twice, in a beat 60 seconds later.

## What the set had to learn about these ducks

Every one of these cost a run, and they are all the same lesson: **the ducks
cannot step over anything, and nothing may start inside anything.**

- A 3 mm nest rim caught a foot and put her over. The rims are now sunk flush,
  top at z = 0.
- One straw capsule standing 1.5 mm proud jammed the first design's sliding
  cover with 208 N and stopped a 4 N motor dead. Straw lies flat now, and
  the cover is gone.
- An 8 mm dais in the middle of the meeting spot tripped him. The dais and the
  heart stakes are off every walking line.
- The egg pocket lies between the meeting spot and where she sits, so walking
  straight there she went through it. She loops east and **backs** onto her
  spot instead — the same move the father uses.
- Guide fingers hold each egg upright as it rises (unguided: upright 1.00 →
  0.28). They stop at the egg's equator: taller ones stand in the arc the lid
  sweeps and the lid cannot open.
- The lift motor started at 300 N/m against an 8 N limit and launched the eggs
  100 mm into the air. It is 60 N/m against 1.2 N.
- Two hearts 30 mm apart, 32 mm wide, started 2 mm inside each other.

One shared change to `task_love_story.py`: `Agent.drive_to` now takes an
opt-in `loose_aim`. The aim angle to a nearby point is mostly noise — 40 mm of
lateral error at 150 mm out reads as 0.4 rad, sends the duck into a turn, the
turn shuffles it, and it never arrives (20 s of oscillation, measured). With
`loose_aim` the heading tolerance opens up inside 180 mm. It is **off by
default**, so the two earlier films keep the approach they were recorded with;
this one opts in.

## Measured

From `videos/love_story/duck_love_story_v2.json`, the recorded run:

```
success            true, 17/17 phases, 146.0 s
beak kiss          measured, jaw on jaw, tips 5.4 mm apart
head kiss contact  measured (he arrives 105 mm from her head)
duckling kiss      NOT reached: beaks stop 92.5 mm and 110.6 mm short
egg rise           99.5 mm and 99.5 mm; cradle travel 99.8 mm
lid angles         -2.494 and -2.494 rad
minimum upright    0.98 she, 0.98 he, 0.93 and 0.99 the ducklings
mocap bodies       0        equality constraints  0
worst penetration  5.6 mm
```

The 5.6 mm is the compliant contact model, not a collision bypass. These
numbers validate the recorded layout; they are not a success rate over
randomised stories.

## The last kiss is close, not contact

`duckling_gap_mm` is reported because the beat is not fully achieved. The
cradle lifts the open shells 100 mm so the chicks reach the parents' head
height, and both parents lower their heads over them, but the beaks stop
about 70-80 mm away. Five ways to close it were tried and each one broke
something:

* bowing (ground-pick) puts the beak at the right height but shuffles the
  duck 20-100 mm in a direction that varies;
* leaning to full stretch presses the head onto the folded lid or the shell
  and tips the whole egg, chick included (upright 0.99 -> 0.69);
* standing closer than 105 mm puts a chest against the egg, same result;
* holding a lid hard on its stop at -2.79 rad gives it nowhere to give: it
  goes past its limit as the cradle accelerates and levers its own shell.
  Lids are held at -2.45 instead;
* a chick raising its head to meet them tips itself over with nothing near
  it. The curled pose is the only one a passive chick holds.

Closing the gap needs a shell that folds fully away, or a policy for the
chicks.

## The kiss is on the mouth, and here is why that was hard

Straight on it is impossible, and the geometry says so plainly: her top head
shell reaches **6.4 mm further forward than her beak tip**, on the centreline.
Two ducks nose to nose touch shells 13 mm before the beaks meet. Every
pitch-only lean was tried and 29 mm apart was the best of them.

The way through is the way people do it — stand a little to one side and turn
your head. Yawed and offset, the head shells pass each other instead of
meeting. A contact-checked search over both head poses (10 parameters) found
**3.2 mm of tip gap with jaw touching jaw and no other pair of geoms in contact
anywhere on either duck**. The recorded run measures 5.4 mm.

Four things had to be right, and each cost a run:

* **Pose first, then walk to it.** Walking to a mark and then turning the heads
  fails: head yaw drags the beak tip 44 mm backwards, so by the time both had
  turned they were 113 mm apart with every head command at its limit.
* **The stand mark is solved from the tips each tick**, not fixed in advance.
  The policy tracks a head command loosely -- a yaw command of 0.2 rad moves
  the joint 0.05 -- so a mark computed from the intended pose is wrong by
  however much the head fell short.
* **The mark is heading-free.** The first version assumed he would face
  straight down the axis. He arrived within 10 mm of it facing 98 degrees off,
  because `drive_to` leaves a duck pointing wherever it came from.
* **He is not posed until he has stopped walking.** `lean` switches a duck to
  the standing policy, and a standing duck ignores a turn command. He sat 43
  degrees off her and never corrected, because `face` was talking to a policy
  that was not listening.

Jaw contact alone is not the test, either. The jaw geom runs the width of the
head, and the first version reported success with the tips 92 mm apart -- they
had met side to side. The check is jaw-on-jaw **and** tips within 30 mm.

## The proposal is a sit, not a kneel

The first version used the ground-pick policy. On screen it reads as a
collapse: the beak goes to the floor and the body folds forward. No policy on
this robot puts one knee down, and nothing may be posed by hand, so the beat
now uses the sit-stand policy's sit. It is a trained, deliberate motion, and
it lands him low in front of her with his head still up.

## Two rendering bugs worth remembering

The first was the nest.
The four nest-rim boxes lie inside the ground slabs' footprint, and both had
their top face at exactly z = 0. Physically that is fine and no test failed.
On screen the depth buffer could not order two coplanar surfaces, so the whole
rim rectangle flashed brown-green-brown as the camera moved.

The rim now stands `RIM_PROUD = 0.2 mm` above the grass: far more than the
depth buffer needs to tell the faces apart, and far less than anything that has
tripped a duck here (the straw that caused trouble stood 1.5 mm proud).

The second was the ducks themselves, and it looked like the same thing but was
not. The shadow map is spread over `shadowclip * extent` metres, and the extent
here is set by the four 5 m ground slabs, not by the 50 mm robot anyone is
looking at: 12 m of world over 4096 texels, so **3 mm per texel** on a head
50 mm across. Self-shadow landed in blocks a sixth of a head wide, and those
blocks crawled over her face as she moved. On screen that is a stripe flashing
under her bow and small rectangles appearing and vanishing on the shell.

It is now `shadowclip 0.5` with an 8192 map: 0.5 mm per texel, six times finer,
and the ground shadows measurably survive (31.8k shadowed pixels against 31.0k
before).

Worth writing down, because the first guess was wrong. Coincident faces on her
head *were* found, and shrinking them changed nothing. Freezing her and moving
only the camera settled it: 176 pixels flipped back and forth, out of 218,400.
There is no z-fighting on the robot. The 7,555 "flickering" pixels in the first
measurement were her head turning 6 degrees while the test ran.

`src/microduck_lab/tests/test_no_render_flicker.py` checks both: no coplanar
visible faces, and under a millimetre per shadow texel.

## Scope

A simulated robot story: a mechanical nest lift and hinged capsules, not
biological laying or shell fracture. The kiss is his beak on her face — the
beak tips cannot pass the head shells on this robot, measured every way the
lean can be arranged. The ducklings have no policy at a fifth scale; they sit
in the one pose a passive duckling stays upright in, and their servos move
their heads and beaks. The scaled ducklings and the lift have not been
validated on hardware.
