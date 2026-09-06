# Microduck project journey

**Dates:** 2026-08-28 to 2026-09-04.
**Goal:** get the Microduck running in simulation, train it with RL, and try to
teach it new skills — skating on ice, and a one-legged figure-skating spiral.

This is the story of what we did, what went wrong, what the numbers were, and
what each method actually computes. It is written to be read by someone who
wants to learn the methods, not only to record the results. Every section that
uses a method gives the equation that the code uses.

| File | What it covers |
|---|---|
| [01_setup_and_first_sims.md](01_setup_and_first_sims.md) | The two repos, the cluster, headless rendering, and the three traps that make a good policy fall over. |
| [02_closed_loop_tasks.md](02_closed_loop_tasks.md) | Two scripted tasks: fetch-and-kick a ball, pick up a cube in the beak. The mouth joint. |
| [03_ice_skating_rl.md](03_ice_skating_rl.md) | Training on ice. How MuJoCo combines friction. Why the policy learned to stand still, and what worked. |
| [04_spiral_rl.md](04_spiral_rl.md) | Six RL attempts at a one-legged spiral on rollers. Three metric traps. Two-layer pose rewards. |
| [05_scripted_expert_and_physics.md](05_scripted_expert_and_physics.md) | A hand-written controller, and the measurements that show why the spiral cannot be held by this robot. LQR and CoM-Jacobian equations. |

Evidence for everything:

| Source | Holds |
|---|---|
| `videos/` | every rendered clip named in the text |
| `src/microduck_lab/src/microduck_lab/tests/test_ice_cfg.py`, `src/microduck_lab/src/microduck_lab/tests/test_spiral_cfg.py` | the assertions that pin each measured number |
| `the training log` | every training log |
| `src/microduck_lab/tasks/skating/ice_experts/` | the scripted controllers and the shared harness |
| `HANDOFF.md` | the short operational summary |

## The one-paragraph version

We installed both repos, found that training needs a GPU and rendering
needs a GPU node, and rendered the seven pretrained behaviours. We built two
closed-loop tasks on top of the pretrained policies. We trained an ice-skating
policy: the best one came from a mid-training checkpoint, and we learned that
MuJoCo takes the maximum of two friction values. We then spent six RL runs and
one scripted-controller panel on a one-legged spiral, and failed every time.
The failure is physical. The duck has no ankle-roll joint, so when it leans its
hip to put its weight over one foot, the sole tilts with the leg and only the
sole's outer edge touches the floor. The centre of mass ends up 28 mm outside
the only contact line. That is a knife edge, the same as a roller wheel, and
no controller can balance on it. See section 5 of file 05.
