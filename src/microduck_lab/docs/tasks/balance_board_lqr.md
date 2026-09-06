# Balance board solved in simulation with whole-body LQR

**User review changed the behavior target:** holding still was insufficient;
the duck should repeatedly rock the board. Use the [active rocking expert](balance_board_rocking.md)
and [new video](../../videos/balance_board/board_rocking.mp4). This report describes the static
stabilizer retained underneath that controller.

**Verified 2026-09-05:** the original orientation-A task now passes. The duck
holds the original free roller for **60 seconds** and passes **33/33** tested
episodes. The previous differential-leg expert reproduces its **1.80 s** hold.
The robot, roller size, plank size, contact geometry, servo strength, and
one-second setup are unchanged.

- Controller: [`src/microduck_lab/tasks/balance_board/lqr.py`](../../src/microduck_lab/tasks/balance_board/lqr.py)
- Saved expert: [`videos/balance_board/board_lqr_policy.npz`](../../videos/balance_board/board_lqr_policy.npz)
- Video: [`videos/balance_board/board_lqr_push.mp4`](../../videos/balance_board/board_lqr_push.mp4), 30 s with a +0.05 m/s lateral velocity kick at 2 s
- Results: [`videos/balance_board/board_lqr_validation.json`](../../videos/balance_board/board_lqr_validation.json)
- Video measurements: [`videos/balance_board/board_lqr_push.json`](../../videos/balance_board/board_lqr_push.json) and [CSV](../../videos/balance_board/board_lqr_push.csv)
- Regression tests: [`src/microduck_lab/src/microduck_lab/tests/test_balance_board_lqr.py`](../../src/microduck_lab/src/microduck_lab/tests/test_balance_board_lqr.py)

## Run it

From `$REPO`:

```bash
# Design locally (about one second) and evaluate the original 10-second task.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.lqr

# Replay the saved expert, with a push, without identifying dynamics again.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.lqr \
  --load-policy videos/balance_board/board_lqr_policy.npz --seconds 30 --push .05

# Reproduce all 33 episodes and regenerate the saved policy/results.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.lqr \
  --suite --save-policy videos/balance_board/board_lqr_policy.npz \
  --output videos/balance_board/board_lqr_validation.json

# CPU regression tests, no pytest installation needed.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p test_balance_board_lqr.py -v

# Rendering needs EGL on a GPU node, as in HANDOFF.md.
  MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.balance_board.lqr --load-policy videos/balance_board/board_lqr_policy.npz --seconds 30 --push .05 --video videos/balance_board/board_lqr_push.mp4
```

The script exits nonzero if the requested rollout fails or any suite episode
fails. `--trace` writes measurements to CSV; `--output` writes JSON. Video
frames play at their actual 25 Hz capture rate. The video starts at release;
the one-second held setup is excluded from the clock.

## What changed

The old controllers constrained the robot to one or two prescribed joint
combinations. Their inability to stabilize the task did not establish that
the complete 14-actuator robot was uncontrollable. No ankle-roll joint was
added. Instead, the expert coordinates all the existing joints using the
coupled robot/plank/roller dynamics.

1. Run the original one-second held-board setup to obtain nominal joint
   targets `u0` and configuration `q0`.
2. In a separate scratch `MjData`, numerically differentiate the actual
   **20 ms** simulator transition, including four 5 ms physics steps. This
   captures contacts, servo response and the moving support together.
3. Form `x = [differentiatePos(q0, qpos), qvel]` (64 dimensions for this model),
   then approximate `x_next = A x + B (u - u0)` locally.
4. Penalize body/platform displacement and tilt, with a small velocity cost
   and actuator-effort cost. Run 1,000 finite-horizon Riccati steps to get `K`.
5. At 50 Hz, return `u = clip(u0 - K x, actuator_ctrlrange)`.

The policy is a constant matrix multiplication at runtime. It never changes
positions, velocities, forces, body masses or constraints. It only returns
14 position targets. The original actuator torque clamp remains
**±0.6405 Nm**. Position targets can overshoot joint travel, as in the shared
simulator; the physical joint limits and torque saturation remain active.

The model has a strong unstable discrete mode, magnitude **1.1184 per 20 ms**.
Its roller also has nearly neutral coordinates. The finite-horizon design
leaves a largest local eigenvalue around **1.000016**, so these results are
empirical finite-duration holds, not a proof of indefinite stability.

## Validation

All episodes use the original 30 mm radius, 60 g loose roller and
300 × 120 × 12 mm, 70 g plank. The robot mass is 737.24 g. Neither support is
fixed after the setup phase. The evaluator checks contacts at every 5 ms
physics step and refreshes MuJoCo's derived geometry before scoring it.

| Test group | Episodes | Outcome |
|---|---:|---|
| Nominal, original geometry | 1 × 60 s | Passed |
| Lateral velocity kicks ±0.01, ±0.02, ±0.05 m/s at 2 s | 6 × 10 s | All passed |
| Random initial velocities, seeds 0–19 | 20 × 10 s | All passed |
| Contact friction 0.8, 1.0, 1.2, 1.8, with +0.01 m/s kick | 4 × 10 s | All passed |
| Robot mass/inertia ×0.9 and ×1.1, with +0.01 m/s kick | 2 × 10 s | Both passed |
| Separate rendered +0.05 m/s kick run | 1 × 30 s | Passed; inspected frames |

Random starts perturb base linear velocity by up to 0.01 m/s per axis, base
angular velocity by 0.05 rad/s per axis, and joint velocities by 0.02 rad/s.
The mass/friction variants use the same nominal gain, without redesign.
These are a specified test battery, not a population-wide success-rate claim.

Across the 33-episode battery:

- No plank-floor, foot-floor or non-foot robot contact.
- No loss of plank-roller contact.
- Worst plank clearance: **43.79 mm**; worst tilt: **6.40°**.
- Longest foot contact interruption: **25 ms**, within the original 100 ms grace.
- Nominal 60 s run: maximum plank tilt **0.0174°**, trunk roll **0.0485°**,
  and continuous contact of both feet.

The 30 s video shows the duck lean and change its leg configuration just after
the push, then recover upright on the plank. The board remains clear of the
floor throughout. The capsule's round end is visible under the board because
the camera looks along the roller axis.

Ablations confirm this remains an unstable balance problem:

| Controller on the same setup | Hold |
|---|---:|
| Constant calibrated targets, feedback removed | 1.70 s, plank-floor contact |
| Feedback with all plank/roller state columns zeroed | 2.11 s, trunk falls |
| Full feedback | 60 s |

Removing velocity feedback alone happened to pass the nominal 10 s case;
that ablation was not validated for push recovery. Raw results are in
[`board_lqr_ablations.json`](../../videos/balance_board/board_lqr_ablations.json).

## Scope and research

This is a **simulation expert with privileged state**, not a deployable
61→14 ONNX policy. It reads exact positions, orientations and velocities of
the robot, plank and roller. `LQRExpert(state, dt)` requires `state['qpos']`
and `state['qvel']`, provided by this script's evaluator. The original
`tb.run()` observation dictionary lacks those fields; passing this controller
there directly is unsupported. The existing shared scripts are untouched.

Real deployment needs a state estimator (including the moving support),
sensor-noise/delay tests, and a deployment integration. No hardware success,
one-legged balance success, or orientation-B result is claimed. The existing
box sole/capsule contact approximations from the original task are retained.

The web search found direct prior work: Baltes, Iverach-Brereton and Anderson,
[“Jimmy DARwIn Rocks the Bongo Board” (2013)](https://www.humanoidsoccer.org/ws13/papers/HSR13_Baltes.pdf).
They investigate PD, predictive feedback and deliberate swaying to handle
hardware limitations. Their paper also reports floor strikes, so it is useful
precedent rather than a ready-made solution to our stricter criterion.

The supplied [Greatist balance-board article](https://greatist.com/fitness/balance-board-exercises)
describes weight shifting and an upright starting stance. The robot's
restricted joints make a controller derived from its own dynamics more useful
than copying the human stance literally. This implementation uses local LQR,
not a copied trained policy or the paper's sway controller.
