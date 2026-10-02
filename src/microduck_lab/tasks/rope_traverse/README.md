# Rope traverse: hang by the legs, inch sideways

**Status (2026-10-01): works in simulation.** The duck hangs upside down from
the rope by its two legs. It moves sideways along the rope with an inchworm gait.

Two filmed trials pass the 15 cm traverse rule:

| trial | gait | cycles | travel | time | speed |
|---|---|---|---|---|---|
| `rope_try62` | reference (hand-tuned) | 13 | 165 mm | 95 s | 1.7 mm/s |
| `rope_try63` | fast (searched) | 6 | 159 mm | 30 s | 5.2 mm/s |

In both trials the legs carried at least half the weight at every tick.
Nothing but the two legs touched the rope, and nothing touched the floor.

Physics only: after the start, only motor targets move the robot. There are
no welds, no support forces, and no rope state in the controller. Not tested
on hardware.

This replaces the earlier mouth-and-feet prototype. That prototype never held
on: trials 1-60 all fell to the floor (see "Earlier prototype").

**Post to post** (`rope_try64.mp4`, wide view of the whole rope): the fast gait
starts 0.235 m left of centre and runs 16 cycles. It crosses 45 cm of rope in
79 s and stops 0.22 m right of centre, close to the right post.

- No contact with the posts or the floor, and no other body part on the rope.
- One 20 ms tick near the right end had only 47% leg support. So the strict
  judge does not pass this run.
- Running further hits the post: 22 cycles from x = -0.26 m end against the
  right post (`rope_try65.mp4`). In a CPU run of the same settings it also fell.

Videos: `videos/rope_traverse/rope_try62.mp4`,
`rope_try63.mp4`. Each has a `.json` (settings, judge result, source hashes)
and a `.npz` (trace).

## How it works

**The hold** (`arch.py`). Each leg folds into a hook over the rope:

- the thigh on one side of the rope,
- the shin across the top,
- the foot on the other side.

The robot hangs head down below the rope. The rope runs along the robot's
left-right axis, so the robot moves sideways.

A hook needs no friction: the rope sits under the shin, between two walls.
Folding the knee further squeezes the hook around the rope. That is the clamp.

**The gait** (`inchworm.py`). One cycle has three moves. Both legs stay on
the rope all the time:

1. The trailing leg clamps. The leading leg opens a little, and its hip roll
   slides it out along the rope.
2. The leading leg clamps. The trailing leg opens and slides in.
3. Both hooks relax. Both hip rolls turn back together. This carries the
   body forward between the hooks.

It is open-loop: joint targets only, in smooth steps at 50 Hz.
`direction=-1` mirrors it to move the other way.

**Pass rules** (`traverse.judge`):

- Hold: 2 s in a row carried by the two leg hooks alone. No floor or post
  contact, no other body part on the rope, and the legs carry at least half
  the weight.
- Traverse: the centre of mass moves 15 cm along the rope. The hold rule is
  never broken after the start. Both hooks carry load at the end.

## Robustness

Each gait was run in 16 changed conditions (`search.py check`). Travel is
measured along the rope. A "dip" is a 20 ms tick in which the legs carry
less than half the weight; the judge counts every dip as a failure.

| change | reference gait, 4 cycles (28.8 s) | fast gait, 5 cycles (24.2 s) |
|---|---|---|
| nominal | 47 mm | 134 mm |
| friction x0.7 | 47 mm | 140 mm (dip: 1 tick, 48%) |
| friction x1.3 | 46 mm | 125 mm |
| mass x1.05 | 48 mm | 140 mm |
| mass x0.95 | 44 mm | 122 mm |
| kp 400 | 49 mm | 126 mm |
| kp 200 | 38 mm | 64 mm |
| battery 7.0 V | 47 mm | 132 mm (dip: 1 tick, 46%) |
| rope sag 20 mm | 45 mm | 133 mm |
| rope sag 6 mm | 46 mm | 132 mm |
| rope 10 mm thick | 62 mm (dip: 2 ticks, 42%) | 144 mm (dip: 7 ticks, 39%) |
| rope 15 mm thick | 34 mm | 121 mm |
| start 10 cm left of centre | 49 mm | 135 mm |
| start 10 cm right of centre | 43 mm | 126 mm |
| moving the other way | 44 mm | 120 mm |
| rope 60 cm high | 50 mm | 132 mm |

- No run fell, touched the floor, or put another body part on the rope.
- The reference gait is the safer one: one dip, on the thinner rope only.
- The fast gait swings the body more. A 10 mm rope is the hardest case for
  both gaits: the hooks are shaped for 12 mm.
- At the walking-policy servo gain (kp 200), both gaits still move, at about
  half speed.

The fast gait comes from a cross-entropy search (`search.py search`, 12
generations, then all step times x1.4 to remove most dips). Faster samples
reached 7-9 mm/s, but they dipped in 3-4 of the 16 conditions.

## Why legs only, not mouth and feet

- The old mouth-and-feet grips fell in every trial (1-60). I replayed the
  reference grip (trial 59's pose) with both fixes below. It still fell to
  the floor in all 8 cases (noslip on/off, pinch 0.25/0.4, kp 200/800).
- The head is 92 mm wide. The gap between the two leg hooks on the rope is
  about 56 mm. So the mouth cannot reach the rope between the legs.
- The sideways joints are small: hip roll +-0.38 rad, head roll +-0.44 rad.
  The head and legs can shift the centre of mass only +-7 mm sideways. So the
  gait does not shift weight. It uses a clamped hook against an open hook.
- Hanging on one leg made the body tilt about 20 deg and turn up to 35 deg
  about the vertical. So the gait never lifts a leg off the rope.

## Two simulation settings that matter

| setting | value | why |
|---|---|---|
| MuJoCo `noslip_iterations` | 10 (new field `Scene.noslip`) | Without it, a clamped hook crept 3-11 mm along the rope per step. The side force was below 0.6 N, while the friction limit was over 6 N. A real rope does not do this. The creep cancelled every step: zero net travel. With noslip, a clamped hook moves less than 1 mm. |
| XL330 firmware P gain | 800 (`arch.KP`) | At 200 (0.55 Nm/rad), a hip roll stalled at 0.06 rad of a 0.3 rad target. At 800 (2.2 Nm/rad) it reaches about 0.3 rad. The runtime already uses about 730 in standby. Stall torque (0.96 Nm) does not change. |

## Limits

- It moves sideways only, and slowly: 1.7 mm/s (reference) or 5 mm/s (fast).
- It starts already hanging on the rope. Getting onto the rope is not done.
- Usable rope: about 0.235 m each side of the centre. The swinging head and
  legs reach the post beyond that. The run is open loop, so the cycle count
  must be chosen to stop before the post.
- Open loop: nothing corrects a missed hook. In all runs above, no hook missed.
- The real robot is untested. Open points for hardware: the higher servo gain,
  rope friction, and the rigid convex collision shapes of the legs.
- The mouth is not used.

## Reproduce

From the repository root:

```bash
# CPU tests (about 30 s): hold, one cycle, mirror, judge, earlier prototype
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/microduck_lab/tasks/rope_traverse/

# film a traverse on a GPU node (reserves a new rope_try number)
sbatch -M cluster local_storage/hb_dev/scripted_policy/rope_traverse/video.sbatch \
  -m microduck_lab.tasks.rope_traverse.traverse \
  --gait src/microduck_lab/tasks/rope_traverse/fast_inchworm.json --cycles 6

# post to post, wide view
sbatch -M cluster local_storage/hb_dev/scripted_policy/rope_traverse/video.sbatch \
  -m microduck_lab.tasks.rope_traverse.traverse \
  --gait src/microduck_lab/tasks/rope_traverse/fast_inchworm.json \
  --cycles 16 --x -0.235 --wide

# no video, on CPU
.venv/bin/python -m microduck_lab.tasks.rope_traverse.traverse --cycles 2 --no-film

# robustness table, or a new search (CPU, many workers)
.venv/bin/python -m microduck_lab.tasks.rope_traverse.search check \
  --gait src/microduck_lab/tasks/rope_traverse/fast_inchworm.json --workers 16
```

## Files

| file | what |
|---|---|
| `scene.py` | the rope between two posts; now with `noslip` |
| `arch.py` | leg-hook pose, start placement, rope loads |
| `inchworm.py` | the gait (open-loop joint targets) |
| `traverse.py` | trial runner, judge, video |
| `search.py` | robustness table and cross-entropy search (CPU) |
| `reference_inchworm.json` | hand-tuned gait (trial 62) |
| `fast_inchworm.json` | searched gait (trial 63) |
| `test_arch.py` | CPU checks for the hold, the gait and the judge |
| `fit.py`, `run.py`, `reference_grip*.json`, `test_rope.py` | earlier mouth-and-feet prototype |

## Earlier prototype (mouth and feet)

Kept for reference; it never held on.

- Trials 1-6: the start pose overlapped the rope; no dynamics were run.
- Trials 7-9: unstable rope mounts with MuJoCo resets. Their old
  `error: null` fields are not evidence of valid physics.
- Trials 10-60: corrected mounts and smaller physics steps. Several hanging
  poses, bites, jaw torque limits (0.25 and 0.5 Nm), COM placement and ankle
  pressure were tried. No trial held for 2 s; most fell within 0.3 s.

The jaw torque limit (0.25 Nm) is provisional, not a hardware value. To rerun
the old hold test: `run.py --poses reference_grip.json --count 1 --pinches .25`
through `local_storage/hb_dev/scripted_policy/video.sbatch`.
