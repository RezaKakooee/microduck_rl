# 01 — Setup, and the first simulations

**Date:** 2026-08-28, one session.
**Goal:** install, run the duck in simulation, train something, and render it.

---

## 1. Two repos, and which one is which

The request was "install this and run the duck in simulation". The repo we
were in, `microduck`, is not a simulator. It is the robot's brain: Rust daemons
that run on the RK3566 board. Its own design doc says simulation was deferred,
and `robotd --fake` describes itself as "a robot made of nothing".

Simulation and RL live in the sibling repo `microduck_rl`: mjlab, MuJoCo Warp,
PPO, the MJCF models, and the ONNX export.

**Decision:** install both. Work in `microduck_rl`. Keep `microduck` for the
pretrained policies it vendors in `policies/` and as the deployment target.

The contract between them is **61 observations in, 14 actions out**. `robotd`
refuses anything else at load time.

## 2. What the training needs

A GPU. Training speed at 4096 environments is about 1.15 s per iteration on an
H200 and about 1.9 s on an RTX 3080. The README budgets 1-2 hours for a usable
walking gait, so a 10 GB card is enough for the flat tasks.

## 3. Rendering without a screen

A machine with no display and no OpenGL library cannot run the interactive
viewer in `scripts/infer_policy.py` at all.

**What we found:** `libEGL` comes with the NVIDIA driver. With `MUJOCO_GL=egl`
on a GPU, MuJoCo renders offscreen. No root needed.

**What we built:** `src/microduck_lab/tools/headless_rollout.py`. It drives the same
`PolicyInference` class as the viewer script, with no window, and writes an mp4
with a camera that tracks the trunk. `src/microduck_lab/sim/duck_sim.py` holds the shared
setup so the scripts cannot drift apart (see section 5 for why that matters).

Ignore the `EGLError` traceback after a successful render. It comes from
teardown, after the frames are written.

## 4. What sudo would and would not buy

We had no root. Two Rust crates (`padd`, `mediad`) need `libudev-dev` and
GStreamer dev packages, so `cargo test --workspace` cannot run. The other 17
crates build and pass 321 tests.

An offer came to move the project to a server with an A100 and sudo.

**What we found:** the A100 buys nothing — the cluster already has H200s. Sudo
would buy Docker, which `scripts/board-test.sh` needs (60 assertions against
the real board image), and the two blocked crates. Rendering, which we thought
needed sudo, does not.

**Decision:** stay on the cluster.

## 5. Three traps that make a good policy fall over

The first time we ran the pretrained walking policy headless, it fell within
two seconds. It looked like a physics problem. It was three setup mistakes.

**Trap 1: the timestep.** The scene XML says 0.002 s. `infer_policy.py:882`
overrides it to 0.005 s so that 4 physics steps per control step gives 50 Hz,
which is what the policies were trained at. Without the override the policy
runs at 125 Hz against a model it never saw.

**Trap 2: projected gravity.** `PolicyInference.__init__` defaults
`use_projected_gravity=False`. The viewer's `main()` passes `True`. At the
default, the policy reads the raw accelerometer where it expects projected
gravity, and falls over. This was the worst trap. It cost the most time and it
looked exactly like physics.

Projected gravity is the world's down-vector expressed in the trunk frame.
With trunk quaternion `q` and rotation matrix `R(q)`:

```
g_body = R(q)^T · [0, 0, -1]

roll  = atan2( g_body.y, -g_body.z )      0 when upright
pitch = atan2( g_body.x, -g_body.z )      + = nose down
```

Every controller in this project reads roll and pitch this way.

**Trap 3: the start pose.** The scene's `STAND` keyframe is an older pose. The
policies treat `DEFAULT_POSE` (the `HOME_FRAME` in `microduck_constants.py`) as
their zero. Start there, at trunk height 0.125 m, with the servo current limit
applied. The current limit matters: the XL330 saturates at 1.75 A, and torque
is `k_t · I`, so the actuator force is clipped at `k_t · 1.75` N·m.

After the three fixes the policy stood and walked:

| Command | Achieved | Fell |
|---|---|---|
| 0.30 m/s forward | 0.13 m/s | no |
| 1.0 rad/s turn | 0.38 rad/s | no |
| 0.15 m/s forward | 0.00 m/s | no |
| 0.2 m/s sideways | 0.000 m/s | no |

Two things in that table shaped everything later. The walking policy does not
move at all below about 0.25 m/s commanded, and it **cannot strafe** — a
sideways command produces exactly zero sideways motion. The roller policy is
the only one that tracks its command well (0.42 achieved for 0.40 asked).

## 6. First training, end to end

We trained the walking task for 10 iterations on an RTX 3080, exported the
checkpoint with `scripts/export.py --checkpoint-file`, and ran the ONNX in the
headless script. The 10-iteration policy fell over at once, as it should. The
point was that the loop closes: train → export → 61-in 14-out ONNX → sim.

PPO is what mjlab trains with. The objective it maximises, per update, is the
clipped surrogate:

```
r_t(θ)   = π_θ(a_t | s_t) / π_old(a_t | s_t)
L_clip   = E_t [ min( r_t · Â_t ,  clip(r_t, 1-ε, 1+ε) · Â_t ) ]
```

`Â_t` is the advantage estimate, `ε` the clip range. The clip keeps each
update close to the previous policy. Nothing in this project changed PPO; every
change was to the reward, the curriculum, or the physics.

## 7. Seven behaviour clips

With the setup right, we rendered one clip per pretrained policy on a GPU node.
`bash src/microduck_lab/render/render.sh clips` regenerates them.

| Clip | Result |
|---|---|
| walk, 0.3 m/s | 1.47 m in 12 s |
| turn, 1.0 rad/s | 3.84 rad |
| roulade | 14.4 rad tumble, lands |
| ground pick | trunk dips 0.116 → 0.084 m |
| sit then stand | dips to 0.059 m, back to 0.116 m |
| ball kick | duck stays put |
| rollers, 0.4 m/s | 4.6 m, tracks 0.42 m/s |

---

## Appendix — numbers and commands

**Install and verify**

```bash
cd $REPO && uv sync
uv run python -c "import mujoco, onnxruntime, torch; print(mujoco.__version__)"
# mujoco 3.10.0, onnxruntime 1.24.4, torch 2.9.1+cu128
```

**GPU test on a node**

```bash
  uv run python -c "import warp as wp; wp.init(); import mujoco_warp, mjlab"
```

**Training smoke test, then a real run**

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

**Headless rollout of the walking policy**

```bash
uv run python -m microduck_lab.tools.headless_rollout \
  --walking ../microduck/policies/alpha_walking.onnx --new-cmd-obs \
  --lin-vel-x 0.3 --seconds 15
```

**Rust side**

```bash
cd $MICRODUCK && cargo test -p robotd -p duck-control -p kinematics ...
# 321 tests pass across 17 crates. padd and mediad need libudev-dev + GStreamer.
ORT_DYLIB_PATH=.../microduck_rl/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4 \
  ./target/debug/robotd --fake --socket /tmp/rd.sock --params robotd-local.toml
# reports healthy, 50.0 Hz, 0 missed ticks, all seven policies loaded
```
