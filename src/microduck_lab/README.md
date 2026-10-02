# microduck_lab

Everything we built on top of the upstream `mjlab-microduck` repo, in one
folder. Upstream code is untouched except for two lines; see "The seam" below.

Import it as `microduck_lab` from anywhere — the venv puts `src/` on the Python
path, so no `sys.path` juggling and no "run it from the repo root".

## Layout

| folder | what is in it |
|---|---|
| `paths.py` | every path the code needs, worked out once |
| `sim/` | `duck_sim.py`, the shared MuJoCo + ONNX policy setup, and `upstream.py` |
| `film/` | the duck-film set builders: `stage`, `geometry`, `scene_v2`, `physical_stage` |
| `rl/` | our RL tasks, one folder per skill (see below), and `register.py` |
| `models/` | our robot variants and scene XML, one folder per skill (see below) |
| `tasks/` | the scripted tasks, one folder per skill |
| `tools/` | `headless_rollout.py`, `blades_warp_check.py` |
| `render/` | `render.sh`, one target per set of clips |
| `tests/` | our tests (upstream's stay in the repo's `tests/`) |
| `docs/` | task write-ups, the project journey, handoffs |

### tasks/

| folder | scripts |
|---|---|
| `common/` | `world` and `runtime`, shared by the scripted skills below (a copy of the human bridge world) |
| `balance_board/` | `board`, `lqr`, `rocking`, `rocking_asym`, `pattern`, `expert` |
| `bridge/` | `bridge` (loose catenary suspension bridge with soft contact compliance); [doc](docs/tasks/suspension_bridge.md) |
| `crawl/` | `crawl`, `expert` (alternating), `baby` (synchronized tuck/drive); [comparison](docs/tasks/crawl.md) |
| `human_bridge/` | one folder per policy: `rl_policy/` (`world`, `scene`, `brother`, `bake`, `story` with the strict rules; she walks across in simulation, 2 of 3), `scripted_policy/`, `crawl_policy/`: [status](docs/tasks/human_bridge.md) |
| `jump/` | scripted jump: `world`, `video` |
| `love_story/` | `original`, `dispenser`, `v2` |
| `objects/` | `pick_up`, `delivery`, `kick_ball`, `egg_on_head` |
| `painting/` | `painting` (chair, brush, easel), `expert` (flower); [notes](docs/tasks/painting.md) |
| `rope_traverse/` | hang from a rope by the legs and inch sideways; [README](tasks/rope_traverse/README.md) |
| `salmon_jump/` | roll up from the back, hop and land; [README](tasks/salmon_jump/README.md) |
| `skating/` | `expert_spiral`, `blades_rollout`, `ice_experts/` |
| `swing/` | `swing` (physical set), `expert` (phase-feedback pumping) |
| `trampoline/` | scripted bouncing and flips on a trampoline |
| `walking/` | `balance_beam`, `eval_slope`, `sideways` (sideways-walk eval: speed, turn, foot lift, slip) |

### rl/

| folder | RL tasks |
|---|---|
| `skating/` | ice, blades, spiral and arabesque (`Mjlab-Velocity-Ice`, `-Blades`, `Mjlab-Spiral-Flat`, `Mjlab-Arabesque-Flat`) |
| `walking/` | sideways walking and slopes (`Mjlab-Velocity-Sideways`, `-Slopes`) |
| `human_bridge/` | the crossing (`Mjlab-Bridge-Sideways`, `warmstart.py`) and one folder per policy: `rl_policy/` (the narrow-beam walker), `scripted_policy/`, `crawl_policy/` |
| top level | `register.py`; the trampoline files (`*trampoline*`) still sit here and move to `trampoline/` later |

### models/

| folder | files |
|---|---|
| `robot/` | robot variants shared by many skills (`mouth`, `egg`) and `add_mouth.py` |
| `bridge/` | `scene_beam.xml`, the bridge template, and the bridge test scenes |
| `objects/` | `scene_pickup`, `scene_delivery`, `scene_egg` |
| `skating/` | `scene_ice`, `scene_blades` |
| `walking/` | `scene_slope` |
| top level | the board-feet robot, the balance-board scenes and the trampoline scenes; they move to their folders together later |

A scene and the robot file it includes must sit at the same folder depth:
MuJoCo reads the robot's `meshdir` relative to the scene file.

## Paths in the docs

Commands in `docs/` use three variables so they work on any machine:

```bash
export REPO=/path/to/microduck_rl          # this clone
export MICRODUCK=/path/to/microduck        # the robot's Rust repo, next to it
export POLICIES=$MICRODUCK/policies        # the pretrained ONNX policies
```

`microduck_lab.paths` reads `POLICIES` from the environment and works out the
rest from its own location, so the Python needs nothing set.

## Running things

Every task is a module, so `-m`, not a file path:

```bash
uv run python -m microduck_lab.tasks.love_story.v2 --dry --trace
uv run python -m microduck_lab.tasks.balance_board.lqr --suite
OPENBLAS_NUM_THREADS=1 uv run --with pytest pytest src/microduck_lab/tests -q
bash src/microduck_lab/render/render.sh film-v2
```

`render.sh` targets: `film`, `film-v2`, `film-dry`, `film-legacy`, `film-posed`,
`film-puppet`, `story`, `clips`, `ice`, `pickup`, `kick`, `swing`.

## The seam with upstream

Two things reach across, and both are deliberate.

1. **`mjlab_microduck/tasks/__init__.py` gains two lines** at the end: it
   imports `microduck_lab.rl.register` and calls `register_all` with its runner
   class. mjlab finds tasks through that entry point, so registration has to be
   triggered from there. The runner is passed in rather than imported back,
   which would make a cycle. Before this refactor the same job took 71 lines
   there plus 399 in `mdp.py`; both files are otherwise byte-identical to
   upstream now.

2. **`sim/upstream.py`** is the one place with a `sys.path.insert`. Upstream
   `scripts/infer_policy.py` is not part of any package and cannot be imported
   normally. There used to be 32 copies of that line across our scripts.

`models/` also reads upstream meshes: our robot XML points `meshdir` back at
`src/mjlab_microduck/robot/microduck/assets` rather than copying 26 MB.

## Outputs

Videos, sheets and reports go to `videos/<task>/`. `videos/` at the repo root is
a symlink to `local_storage/videos/`, and both are gitignored.
