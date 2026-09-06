# Two ducks: a proposal, a kiss, two eggs, two ducklings

A 67-second rendered short film in six scenes, made from the real microduck
MJCF.

```bash
src/microduck_lab/film/legacy_story_two_ducks.py --check          # geometry checks, no GPU needed
bash src/microduck_lab/render/render.sh story
```

- Film: [`videos/love_story/story/story.mp4`](../../videos/love_story/story/story.mp4) — 1080x720, 30 fps, 39 MB
- Small copy: [`videos/love_story/story/story_small.mp4`](../../videos/love_story/story/story_small.mp4) — 960x640, 2.6 MB
- Contact sheet: [`videos/love_story/story/story_sheet.png`](../../videos/love_story/story/story_sheet.png)
- One clip per scene: `videos/love_story/story/1_meet.mp4` … `6_hatch.mp4`
- Code: [`src/microduck_lab/film/legacy_story_two_ducks.py`](../../src/microduck_lab/film/legacy_story_two_ducks.py), [`bash src/microduck_lab/render/render.sh story`](../../bash src/microduck_lab/render/render.sh story)

The whole film renders in about 30 s on a debug-partition GPU.

## This is animation, not simulation

`mj_step` is never called. No policy, no reward, no servo torque and no contact
force takes part. Every frame writes `qpos` directly and calls `mj_kinematics`,
the way a keyframe animator poses a rig. **The film says nothing about what the
robot can do** — there is no policy in this repo that kneels, kisses or lays an
egg, and a physics rollout cannot be directed beat by beat anyway.

What is still true of the real robot, because it cost nothing to keep:

| | |
|---|---|
| the model | `robot_allcollisions_mouth.xml` — the real MJCF with the 15th servo, so the beak opens and the ring hangs off the real `mouth_tip` site |
| joint order | left leg, neck/head, right leg, with the model's own left/right sign mirror; poses are written in the left leg's convention and mirrored on the way in |
| the poses | STAND and SIT come from `scene.xml`'s keyframes |
| height | no trunk z is ever typed. Each frame the lowest corner of the duck's visible geometry is measured and the trunk dropped onto the floor, so the walk bob, the kneel and the sit all get their height from the same rule |
| the ducklings | the same MJCF scaled to a third — `scale_spec` scales link offsets, geom sizes, joint anchors and inertias as well as meshes, because scaling meshes alone shrinks the parts and leaves the skeleton full size |

## The four things that were solved rather than tuned

`--check` prints all of them:

```
stage      he x -121 mm, she x +121 mm
kiss       beaks 4.0 mm apart, height 166 vs 166 mm
kneel      trunk 86 mm, folded leg +0.2 mm, planted foot -0.0 mm
head kiss  beak lands 1.4 mm from the aim point (her crown 223 mm), he stands at x -188 mm
nuzzle     beak lands 0.5 mm from a duckling crown at 96 mm, she stands at x -274 mm
duckling   trunk 40 mm vs a parent's 119 mm
```

1. **Where they stand for the kiss.** Each duck is posed at the origin in the
   kiss pose, its beak's own x offset read off, and the two positions set so the
   tips meet with a 4 mm gap. Nothing is eyeballed and nothing intersects.
2. **The kneel.** A grid search over the folded leg, the planted leg and the
   trunk pitch for the pose whose folded leg and planted foot both touch the
   floor with the trunk as low as it will go. It drops him from 119 to 86 mm.
3. **The kiss on her head.** Two unknowns solved per frame: his trunk's xy comes
   from the beak offset of his pose, and the height from bending his neck, whose
   beak travels about 69 mm per radian. Four passes land the beak within 1.5 mm
   of the aim point, which is her crown or his own 216 mm ceiling, whichever is
   lower.
4. **Her laying pose.** Sitting with her head *up* puts her crown 24 mm behind
   the front of her own trunk at 247 mm, which no standing duck can reach past
   her body — the highest a duck can put its beak while still pointing it down
   is 213 mm. Bowed over the nest the crown comes 85 mm forward and drops to
   223 mm, and he can stand 150 mm clear of her and touch the top of her head.
   The pose exists for that reason, not for the look.

## How it is put together

**Four ducks in one world.** `MjSpec.attach` puts the duck MJCF in four times
with a name prefix each; the two ducklings go through `scale_spec` first.

**Colour.** Every geom in the CAD export carries a material, but at runtime
`geom_rgba` wins over the material, so recolouring is one array write. Parts are
sorted by their material's own colour: already-orange (beak, feet) and dark
(servos, lens) are left alone, and everything light is a shell and takes the
duck's colour scaled by how light it was.

**Props.** The ring is a `make_supertorus` mesh, 32 mm across — the one size that
reads as a ring in a 30 mm beak and still goes round the 22 mm stalk under her
head, which is where she wears it. The eggs are a procedural surface of
revolution split along a zigzag into two watertight thin shells, so they
interlock while the egg is whole and read as a cracked shell when they part. The
hearts are a puffed extrusion of the sin-cubed heart curve, so they survive the
camera swinging round.

**Motion.** Every scene is a function of scene-local time that writes the
absolute state of everyone, so any moment can be rendered on its own
(`--frames 21.5`). The gait is not keyframed: the phase is the distance
travelled divided by the step length, so the feet cannot skate, and the ankle
takes back what the hip and knee did — the relation the STAND keyframe already
has — so the soles stay flat. Walks between the stage and the nest bend through
a waypoint, because a straight line crosses the nest.

## Beats

| t (s) | |
|---|---|
| 0–7 | he walks up the lawn, a ring in his beak; she notices him |
| 7–17.5 | he kneels and offers it; she is startled, nods yes, and takes it |
| 17.5–26 | he stands, the ring goes to her throat, and they kiss beak to beak for 3.5 s |
| 26–39 | they walk to the nest; she sits and lays two eggs while he bows to her head |
| 39–52 | she sits on the eggs; they swap; he sits on them |
| 52–67 | both eggs rock, crack, the lids fly off, and a blue and a pink duckling climb out and hop to their parents |

## Limits

- The kneel is the deepest pose the search found with both legs down. On a 25 cm
  biped with 5 cm shins it reads as a deep lunge, not a human genuflection.
- Nothing is supported by contact. A duck sitting on an egg is a pose that
  happens to intersect nothing; the eggs are 46 mm tall and a brooding parent
  clears them by a 21 mm lift written into the pose.
- The ducklings are hidden inside their shells by alpha, not by geometry.
- Her wading into the nest clips the straw. Real ducks do that too, but this is
  not why it happens here.

## Not to be confused with

`src/microduck_lab/film/` and [`duck_love_story.md`](duck_love_story.md) are a
separate, longer (105 s) film of the same story, written in parallel by another
session. The two share no code and no output files.
