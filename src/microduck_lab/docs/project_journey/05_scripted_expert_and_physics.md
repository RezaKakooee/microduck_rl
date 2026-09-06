# 05 — A scripted controller, and the physics that ends the spiral

**Date:** 2026-09-03 to 2026-09-04.
**Goal:** stop tuning rewards and ask the prior question. Can this robot hold
the pose at all? If a hand-written controller cannot, no reward function will.

The answer is no, and the reason is one missing joint.

---

## 1. The scripted controller

`src/microduck_lab/tasks/skating/expert_spiral.py`. It starts the duck already in the pose, moving,
and balances with a PD controller on the support leg. Roll and pitch come from
projected gravity (file 01, section 5).

```
e_roll  = roll  - roll_target
e_pitch = pitch - pitch_target
ctrl[support hip_roll] = target + k_p,roll  · e_roll  + k_d,roll  · d(roll)/dt
ctrl[support ankle]    = target + k_p,pitch · e_pitch + k_d,pitch · d(pitch)/dt
```

All other joints hold the reference pose. The rates are finite differences at
50 Hz. A grid search over the gains measures the hold time.

## 2. Rollers: a knife edge

| Setting | Best hold |
|---|---|
| spiral pose, straight line, any gains | 0.30 s |
| spiral pose, banked turn (lean + steer) | 0.30 s |
| one wheel, free foot lifted only 0.15 rad | 0.76 s |

Every setting gave the same number, which is the sign that control had no
effect. The trace showed why: roll went 0.8° → 64° in 0.32 s. A roller is a
line contact fore-aft and a knife edge sideways. There is no lateral support
width and no restoring force.

We tried a banked turn, because a skater's spiral is skated on a curve. The
lean angle a turn holds up is:

```
tan(θ) = v · ω / g        v forward speed, ω yaw rate
```

It made no difference: the duck fell before it could turn.

Two setup bugs surfaced here. Setting the pose without adjusting trunk height
left the support foot 15.7 mm *below* the floor, and the contact impulse threw
the duck sideways. A binary search on trunk height for the lowest contact-free
position fixed it. And the first grounding used `geom_size` as a radius; mesh
geoms have no meaningful size, so it silently did nothing.

## 3. Normal feet: the missing weight shift

Rollers were abandoned. On flat feet the result was 0.30 s again.

**What we found:** at t = 0 the centre of mass was **42.8 mm to the side** of
the support foot. The duck was already falling before the controller acted.

The cause was the reference pose. It only extended the free leg. The support
leg stayed in its home stance, which is offset sideways, so the body never
moved over the support foot. A person shifts their weight across *first*.

Sweeping the support hip_roll:

| hip_roll shift | CoM offset from support foot |
|---|---|
| 0 | +42.5 mm |
| +0.30 rad | +13.7 mm |
| +0.40 rad | +4.1 mm |
| +0.50 rad | −5.4 mm |

With the shift, the hold improved from 0.30 s to 0.90 s. The gain search kept
asking for more shift, which usually means a target is being clipped.

## 4. The joint limit and the foot

| Joint | Range |
|---|---|
| hip_roll | **±0.384 rad (±22°)** |
| hip_pitch, knee, ankle | ±1.571 rad |

So shift 0.40 was impossible. The best reachable CoM offset is ~13.7 mm.

The foot's footprint in world coordinates, standing normally: 54 mm long,
**41 mm wide, half-width 20.5 mm**. A CoM 13.7 mm off centre is inside that.
So we concluded the pose was statically feasible with 7 mm of margin, built a
"balanced" reference with both hip_rolls at +0.384, and verified it:

```
CoM y = +2.0 mm,  foot mesh spans y = -34.1 .. +6.3 mm   -> inside
```

The scripted controller held it for 0.96 s. We launched RL run
`Mjlab-Arabesque-Flat-MicroDuck` against this pose.

That check was wrong, and the next section is why.

## 5. The finding: the sole is on its edge

A panel of scripted controllers was run on ice (section 7). Two of them, using
different methods, reported the same thing, and we then verified it directly:

```
support foot mesh, y range:  -34.1 .. +6.3 mm
CoM y:                        +2.0 mm
floor contacts on the support foot:  ONE, at y = -26.6 mm
```

The foot mesh *spans* +6.3 mm, but only its **outer edge** touches the floor.
The CoM is **28.6 mm outside the only contact line**.

The mechanism: the duck's leg has five joints — hip yaw, hip roll, hip pitch,
knee, ankle pitch. **There is no ankle-roll joint.** When the hip rolls to put
the body over one foot, the sole rolls with it. Tilted 22° about the fore-aft
axis, the sole touches the floor along one edge. The comparison in section 4
treated the foot as flat on the floor. It was not.

The static condition for standing is that the CoM lies inside the support
polygon — the convex hull of the contact points. Here the support polygon is a
line, 28.6 mm from the CoM. Gravity produces a torque about that line:

```
τ_gravity = m · g · d_lateral ≈ 0.737 · 9.81 · 0.0286 ≈ 0.21 N·m
```

and there is nothing to oppose it. This is a knife edge, the same situation
as the roller wheel, reached by a different route. It is why every version of
the task fell in 0.2–0.4 s, why the gains never mattered, and why "more hip
shift" always looked better right up to the joint limit.

To stand on one foot, a robot needs to lean the leg inward *and* keep the sole
flat. That needs an ankle-roll joint. The duck does not have one.

## 6. Why the head does not help

The head is ~38 % of the mass, and we expected it to be a strong balance lever.
The CoM-tracking controller measured the lateral CoM Jacobian by finite
differences — how far the CoM moves per radian of each joint:

| Joint | ∂(CoM_y)/∂q |
|---|---|
| support hip_roll | **−95 mm/rad** (and pinned at its limit) |
| free hip_yaw | +6.7 mm/rad |
| support knee | −6.3 mm/rad |
| support hip_yaw | −4.5 mm/rad |
| head_yaw | +3.7 mm/rad |
| head_roll | **+1.2 mm/rad** |

The head's roll axis passes about 3 mm from its own centre of mass. Rolling the
head barely moves it. Mass is not a lever; moment arm is.

Could the head work as a reaction wheel? The angular momentum it can supply
by swinging is ~0.22 mN·m·s per rad/s. Undoing the 28 mm offset needs
~30 mN·m·s. About 100× short.

Could the ground help? On ice the lateral force is capped at
`μ · m · g = 0.1 · 0.737 · 9.81 ≈ 0.72 N`, about 0.1 N·m at the CoM against
the 0.2 N·m gravity torque. Not enough even at full friction utilisation.

## 7. The controller panel, and the methods

Seven strategies were designed in parallel against one shared harness,
`src/microduck_lab/tasks/skating/ice_experts/harness.py`, which scores every controller by the same
rule: seconds until the free foot drops below 15 mm, the trunk below 70 mm,
or the free leg tucks below 0.6 rad. A session rate limit killed five of the
seven agents and both verifiers. Two finished. Neither beat the 0.22 s
baseline honestly. Both are worth reading for the methods.

### 7.1 Inverted-pendulum LQR (`lqr_pendulum.py`)

About the support contact the robot is an inverted pendulum. With CoM height
`h = 0.155 m`, `ω₀² = g / h ≈ 63 s⁻²`. Each axis, linearised, with `x` the
lean and `u` the joint command through a lever `L` (m/rad):

```
ẋ = A x + B u
A = [[ 0 , 1 ],         B = [ 0 ,  -ω₀² · L / h ]ᵀ
     [ ω₀², 0 ]]
```

Levers measured from the model: support hip_roll 0.103 m/rad, support
hip_pitch 0.091, ankle 0.023. The LQR gain minimises
`∫ (xᵀ Q x + uᵀ R u) dt` and is

```
P  solves  Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0     (continuous Riccati)
K  = R⁻¹ Bᵀ P
u  = −K x
```

The code calls `scipy.linalg.solve_continuous_are`. Q weights lean and lean
rate; R weights the command. A slew-rate limit on the command keeps the XL330s
out of current saturation. The gains then seed a random search on the harness.

Honest result: 0.22 s. The search found 0.78 s and 9.98 s, and both were
cheats: the duck squatted to a trunk height of 72 mm (limit 70) and rested its
tilted free foot on the floor while the foot *site* stayed 33 mm up. The agent
added a watchdog that cuts the hold at any floor contact by a non-support
body, and reported 0.22 s.

### 7.2 Whole-body CoM tracking (`com_tracking.py`)

Each control step, move the CoM back over the real support point using all
joints at once.

1. Support point: the mean of the floor contacts on the support foot (not the
   site — the site is 15 mm from the edge that touches).
2. Error `e = CoM_xy − support_xy`, and its rate by finite difference.
3. Jacobian `J = ∂CoM_xy / ∂q` (2 × n) by finite differences: perturb each
   joint by `ε = 10⁻³` on a scratch copy of the state, run forward kinematics,
   read the CoM.
4. Damped least squares, with a joint weight matrix `W`:

```
Δq = W Jᵀ ( J W Jᵀ + λ I )⁻¹ · ( −K_p e − K_d ė )
```

   Joints that would be pushed past a limit get weight 0 and the system is
   re-solved, so the pinned hip_roll does not swallow the whole correction.
   `Δq` is rate-limited and added to the targets.

Honest result: 0.32 s, and the agent was clear that this was a slowed
collapse, not a hold. The strong lever is pinned. The other joints together
move the CoM under 10 mm.

### 7.3 The harness can be gamed

Both agents found the same exploit: lie down with the trunk propped at 72 mm
and the free foot's site above 15 mm while its mesh touches the floor. The
harness checks a site height, not contacts. Anyone reusing it should add a
trunk-tilt limit and a "support foot only" contact check. This is the same
class of failure as the reward traps in file 04: the metric admitted a state
that was not the behaviour.

## 8. Conclusion

The one-legged spiral is not a reward-design problem for this robot. It is a
morphology problem. Without an ankle-roll joint the duck cannot lean over one
foot and keep the sole flat, so a single-foot stance is always an edge contact
with the centre of mass outside it. Rollers fail for the same reason from the
other direction.

What would change the answer:

1. An ankle-roll joint, or a hip_roll range well past 0.6 rad. Changes the
   robot.
2. A rounded sole that keeps contact under the CoM as the leg tilts — the
   `microduck-playground` stilts used a rounded tip. Changes the model only,
   and would not transfer to hardware.
3. Dynamic balance on the edge, like a bicycle. Needs lateral ground force to
   steer the contact under the CoM, which ice does not provide and the 0.64 N·m
   servos are too slow for.

The RL runs were stopped. The ice-skating policy (file 03) and the two tasks
(file 02) stand.

---

## Appendix — numbers and commands

**Scripted controller, rollers vs normal feet**

```bash
uv run python -m microduck_lab.tasks.skating.expert_spiral --feet normal --speed 0 --hip-roll-shift 1.4 \
  --kp-roll 0 --kd-roll 1.5 --kp-pitch 4 --kd-pitch 0.3 --seconds 10
# held 0.96 s   (shift past the limit clips to 0.384 and holds the hip there)
uv run python -m microduck_lab.tasks.skating.expert_spiral --feet normal --ice-mu 0.1 --speed 0.4 ...
# held 0.36 s
```

**Verify the edge contact** — build the "balanced" pose with
`src/microduck_lab/tasks/skating/ice_experts/harness.make()`, step once, list `data.contact`:

```
right_foot_collision / floor   y = -26.6 mm      (one contact)
foot mesh y range              -34.1 .. +6.3 mm
CoM y                          +2.0 mm
```

**Joint ranges and footprint** — read `model.jnt_range` for `*_hip_roll`;
transform `right_foot_collision` mesh vertices by `geom_xmat`/`geom_xpos`.

**Shared harness baseline**

```bash
uv run python src/microduck_lab/tasks/skating/ice_experts/harness.py     # baseline PD: 0.22 s
uv run python src/microduck_lab/tasks/skating/ice_experts/lqr_pendulum.py   # HOLD 0.22
uv run python src/microduck_lab/tasks/skating/ice_experts/com_tracking.py   # HOLD 0.32
```

**Pendulum numbers** — m = 0.737 kg, h = 0.155 m, ω₀² = 63 s⁻²,
roll inertia about the contact edge ≈ 0.021 kg·m², servo torque limit
≈ 0.64 N·m (current-limited XL330).
