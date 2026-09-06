# 02 — Two closed-loop tasks on top of the pretrained policies

**Date:** 2026-08-28.
**Goal:** make the duck *do a task*, not replay a behaviour. Find an object,
walk to it, act on it.

Both tasks run on CPU MuJoCo with the pretrained ONNX policies. They read the
object's true position from the simulator. A real version would use a camera
(the `microduck-tracking` repo does exactly that). The point here was to learn
how the policies behave when you chain them.

---

## 1. Fetch and kick a ball

`src/microduck_lab/tasks/objects/kick_ball.py`. The duck steers toward a ball, stops, settles,
and fires the kick policy. Two policies swap mid-episode behind the shared
61-dim observation, which is what the runtime does on the real robot.

The kick policy expects the ball in a fixed pocket in the duck's own frame:
90 mm ahead, 42 mm to the kicking side. Four things had to be right, and each
one silently gave a miss.

### 1.1 How much placement error the kick tolerates

We measured it by teleporting the ball off-pocket and reading how far it flew.

| Forward error | Lateral error | Ball travel |
|---|---|---|
| 0 | 0 | 3.93 m |
| 0 | ±15 mm | 3.3–3.8 m |
| 0 | +30 mm | 1.45 m |
| +15 mm | 0 | 1.78 m |
| +30 mm | 0 | **0.00 m** |

Forward error is the one that kills it. Lateral is forgiving. So the approach
has to nail the forward distance and can be sloppy sideways.

### 1.2 It must fire from a standstill

The kick policies were trained to start and end standing. Fired mid-stride,
the swing misses even with the ball perfectly placed. So the task stops,
waits 3 s for the gait to settle, re-checks the pocket, then fires.

### 1.3 The duck cannot strafe

From file 01: a sideways command gives zero sideways motion. So the duck cannot
sidestep into alignment at the end. It has to aim off during the approach. The
steering target is not the ball; it is the spot where the duck must *stand*:

```
rel   = R(-yaw) · (ball_xy - duck_xy)          ball in the duck's frame
err   = ( rel.x - 0.090 ,  rel.y - (±0.042) )   distance to the standing spot
aim   = atan2( err.y , err.x )
turn  = clip( 2.5 · aim , -1.5 , 1.5 )
walk  = 0.30 if |aim| < 0.25 rad else 0.0     turn on the spot if badly aimed
```

### 1.4 Stop inside the pocket, not short of it

The policy does not move below ~0.25 m/s commanded, so it cannot take a small
step. Parked 40 mm short, it stays there forever. The stop window has to sit
inside the pocket. And the duck rocks back about 7 mm while settling, so the
stop window is shifted ~10 mm nearer than the pocket to absorb it.

A control-flow bug made this hard to see: a duck that settled *out* of the
pocket fell through to "walk forward", shinned the ball 7.7 m across the floor,
and chased it. Now it backs off and retries.

**Results:** ball ahead-left 1.72 m, ahead-right 1.53 m, ball behind the duck
(it has to turn around first) 1.45 m. Three layouts is not a success rate; the
windows were tuned on the first.

## 2. Pick up a cube in the beak

`src/microduck_lab/tasks/objects/pick_up.py`. Walk to a cube, run the ground-pick policy (beak to
the floor), grip, stand up, carry it away.

### 2.1 There is no mouth

The 14 servos are legs and neck. On the real robot the mouth is the **15th
servo**, driven by `robotd` and excluded from every policy
(`duck-control/src/model.rs`: "neck, head, mouth"). The RL export fuses the
whole head into one rigid body, so the beak cannot open in simulation.

**What we built:** `add_mouth.py`, in the style of the repo's own
`add_backlash.py`. It lifts the `jaw` and `jaw_soft` meshes onto a hinge and
appends a position actuator as **index 14**, after the 14 servos. So the policy
still maps straight onto `ctrl[0:14]` and the task drives the mouth itself —
the same split the real robot uses.

Finding the hinge took four renders. The first swung the jaw up through the
head (wrong sign). The second see-sawed about the middle (pivot too far
forward). The third had the duck's facing backwards; we settled it by parking
the cube on the `mouth_tip` site and rendering. The head body's quaternion is a
−90° rotation about y, so body-y maps to world-y and a hinge on `0 1 0` with
positive angle drops the beak tip. Including `bottom_head_shell` in the moving
set is wrong: it spans the head and see-saws.

One trap: `PolicyInference` sizes its joint list from `model.nu`. A 15th
actuator pushes the mouth into the observation and makes it 64-dim. The shared
setup trims it back to 14.

### 2.2 The grip is a weld, not contact

There is no closing force to simulate. `scene_pickup.xml` adds an equality
constraint between the jaw body and the cube, inactive until the beak arrives.
MuJoCo's weld holds body 2 at a fixed pose relative to body 1:

```
eq_data = [ anchor(3) , relpos(3) , relquat(4) , torquescale(1) ]
relpos  = R(q_jaw)^-1 · (p_cube - p_jaw)
relquat = q_jaw^-1 ⊗ q_cube
```

We first welded the cube wherever it lay. That looked fine on the ground and
wrong the moment the duck stood: the head pitched up and carried the cube into
its own face. Now the cube is seated on the beak tip, 20 mm out along the
head's forward axis, and welded there.

### 2.3 The duck shuffles backward while crouching

The beak bottoms out at z = 20 mm, 108 mm ahead of the trunk, in a standing
start trace. In the task it bottoms out at 117 mm — and the cube that was
110 mm away at trigger time is 146 mm away when the beak is down. **The duck
moves back about 36 mm as it crouches.** So it has to stand ~81 mm from the
cube and let the drift carry the cube to the beak. The cube's top face sits at
30 mm, which is exactly beak height.

The first "success" gripped from 38 mm clear of the cube because the grab
radius was 55 mm. Tightening it to 30 mm (the cube's size) exposed the miss.

**Results:** all three layouts succeed. Beak reaches 29–30 mm from the cube
centre, lifts it to 235–250 mm, carries it 0.85–0.99 m.

---

## Appendix — numbers and commands

**Render the kick task, three layouts**

```bash
bash src/microduck_lab/render/render.sh kick
```

**Render the pickup, with a close camera**

```bash
bash src/microduck_lab/render/render.sh pickup
MUJOCO_GL=egl uv run python -m microduck_lab.tasks.objects.pick_up --walking ... --ground-pick ... \
  --cube 1.0 0.5 --cam-distance 0.45 --video videos/pick_up/pick_closeup.mp4
```

**Regenerate the mouth variant**

```bash
cd src/mjlab_microduck/robot/microduck
python3 add_mouth.py robot_allcollisions.xml -o robot_allcollisions_mouth.xml \
  --meshes jaw,jaw_soft --pivot -0.0045 0 -0.040 --axis 0 1 0 --range 0 0.55
```

**Kick tolerance sweep** — teleport the ball to `(0.092 + dx, -0.042 + dy)` in
the duck frame at trigger time, run 8 s, read the ball displacement. The table
in section 1.1 is that sweep.
