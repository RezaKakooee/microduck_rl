"""Microduck velocity environment — slopes and stairs variant.

Same task as the walking recipe (track a commanded twist; head / body pose
slots zero-padded into the shared 61D obs) on a terrain grid that mixes flat
ground, gentle slopes (up and down) and low stairs (up and down). Normal feet,
not rollers.

Built on ``make_microduck_velocity_env_cfg(rough=True)`` so the DR / obs /
noise / delay stack AND the rough-terrain physics guards stay in sync for free:
the ``_soften_terrain_contacts`` spec_fn (2x softer terrain solref — box edges
otherwise produce impulsive NaN forces when a foot lands on them), nconmax=200
and the 30/50 solver iterations all come from that branch. Only the terrain
generator, the terrain curriculum and one foot-swing target change here.

Terrain design (numbers chosen from a measurement, see docs/tasks/slopes.md):

* The pretrained walking policy lifts its feet 15-17 mm at the foot site
  (15 mm at the lowest point of the sole) when commanded 0.3 m/s. So steps of
  10 mm are clearly walkable, 20 mm is at the edge of what the current gait
  can clear — that is the curriculum range, ``STEP_HEIGHT_RANGE``.
* Slopes run 5 -> 15 degrees by difficulty. At 15 degrees a 25 cm robot with
  flat rubber soles is still far from the friction limit (tan 15 = 0.27
  against mu 0.7-1.3) — the difficulty is balance, not grip.
* Slopes are a custom planar-faced frustum heightfield
  (``HfFrustumSlopeTerrainCfg``): mjlab's stock pyramid slope is bilinear, so
  its diagonals run 1.4x steeper than the number you configure.
* Both directions of every obstacle are separate columns: the platform-on-top
  pyramid spawns the robot at the top (walk DOWN first), the inverted one
  spawns it in the pit (walk UP first). Commands resample every few seconds
  with random headings, so each env sees both directions within an episode
  anyway, but the spawn decides what the first metres are.

Curriculum: mjlab's terrain generator in curriculum mode gives one column per
sub-terrain type and ``NUM_LEVELS`` rows of increasing difficulty. Envs start
on rows 0..MAX_INIT_LEVEL and are promoted / demoted by
``mdp_terrain.terrain_levels_walk`` on the distance they actually walked.
"""

import math
import uuid
from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

from microduck_lab.rl import mdp_terrain
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)

# ── Terrain numbers ──────────────────────────────────────────────────────────
# Slope angle at difficulty 0 and 1 (degrees). mjlab's pyramid slope takes a
# rise/run gradient, so these are converted with tan() below.
SLOPE_DEG_RANGE = (5.0, 15.0)

# Step riser height at difficulty 0 and 1 (m). Measured foot swing of the
# pretrained walking policy is ~15 mm, so the range brackets it.
STEP_HEIGHT_RANGE = (0.010, 0.020)
# Step run (tread depth, m). About two foot lengths; the rough task uses the same.
STEP_RUN = 0.15

# One tile per (level, type). Platform is the flat square the robot spawns on.
TILE_SIZE = (8.0, 8.0)
PLATFORM_WIDTH = 2.0
# Flat border around the whole grid (the rough task uses 20 m; the grid here is
# only 5 columns wide, so less is needed).
BORDER_WIDTH = 10.0

# Difficulty rows, and the highest row an env may START on. Rows 0..3 cover
# 5-9 degrees and 10-14 mm steps.
NUM_LEVELS = 10
MAX_INIT_LEVEL = 3

# Spawn share per sub-terrain type (curriculum mode: one column each; the
# proportion only sets how many envs spawn on that column). Up-hill work gets
# the larger share because it is what the task is about.
TERRAIN_PROPORTIONS = {
    "flat": 0.15,
    "slope_down": 0.20,
    "slope_up": 0.25,
    "stairs_down": 0.15,
    "stairs_up": 0.25,
}

# Heightfield vertical quantisation (m). 1 mm keeps a 5-degree slope smooth
# instead of a staircase of 5 mm ledges (mjlab default is 5 mm).
HFIELD_VERTICAL_SCALE = 0.001

# ── Curriculum numbers ───────────────────────────────────────────────────────
# Promote when the env walked more than this fraction of a tile (2.0 m on an
# 8 m tile — 17 s at the duck's real 0.12 m/s), demote below this fraction
# (0.64 m: fell at once, or stuck on the obstacle). Envs commanded to stand or
# turn in place are not judged. See mdp_terrain.terrain_levels_walk.
PROMOTE_FRACTION = 0.25
DEMOTE_FRACTION = 0.08
CURRICULUM_COMMAND_THRESHOLD = 0.05

# ── Reward change ────────────────────────────────────────────────────────────
# The walking recipe asks for a 20 mm swing peak and the trained gait delivers
# ~15-17 mm (about 20% under target). The top of STEP_HEIGHT_RANGE is 20 mm,
# so the peak target moves up to 25 mm: a foot that clears 20 mm of riser
# needs to peak above it. foot_clearance (the drag penalty) is left alone.
FOOT_SWING_TARGET = 0.025


@dataclass(kw_only=True)
class HfFrustumSlopeTerrainCfg(SubTerrainCfg):
    """Square frustum with PLANAR faces, as a heightfield.

    mjlab's stock ``HfPyramidSlopedTerrainCfg`` builds ``z = h * x_hat * y_hat``
    (bilinear), so its faces are curved: the configured gradient holds only
    along the two axes and the diagonals near the platform are up to 1.4x
    steeper, while the tile corners flatten out. This one uses the max-norm
    distance from the centre, ``z = h * clamp((half - max(|x|,|y|)) / run)``,
    which is a true frustum: every face is a plane at exactly the configured
    angle, so "5 to 15 degrees" means what it says.

    ``inverted=False``: platform on top, spawn at the summit (walk DOWN).
    ``inverted=True``: pit, spawn on the pit floor (walk UP).
    """

    slope_range: tuple[float, float]
    """Rise/run gradient at difficulty 0 and 1."""
    platform_width: float = 1.0
    inverted: bool = False
    horizontal_scale: float = 0.1
    """Grid cell size (m). The faces are planes, so this only sets how finely
    the platform edge is resolved."""
    vertical_scale: float = 0.001
    """Height quantisation of the grid nodes (m)."""
    base_thickness_ratio: float = 1.0

    def gradient(self, difficulty: float) -> float:
        d = float(np.clip(difficulty, 0.0, 1.0))
        return self.slope_range[0] + d * (self.slope_range[1] - self.slope_range[0])

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # deterministic given the difficulty
        body = spec.body("terrain")
        slope = self.gradient(difficulty)
        half = min(self.size) / 2.0
        run = half - self.platform_width / 2.0
        assert run > 0, "platform_width must be smaller than the tile"
        rise = slope * run

        nx = int(round(self.size[0] / self.horizontal_scale))
        ny = int(round(self.size[1] / self.horizontal_scale))
        # Node coordinates relative to the tile centre. MuJoCo samples the
        # heightfield at nrow x ncol nodes spread evenly over [-rx, rx] x [-ry, ry].
        xs = np.linspace(-self.size[0] / 2.0, self.size[0] / 2.0, nx)
        ys = np.linspace(-self.size[1] / 2.0, self.size[1] / 2.0, ny)
        dist = np.maximum(np.abs(xs)[None, :], np.abs(ys)[:, None])  # [ny, nx]
        frac = np.clip((half - dist) / run, 0.0, 1.0)
        height = rise * frac  # 0 at the edge, rise on the platform
        noise = np.rint(height / self.vertical_scale).astype(np.int16)

        elevation_range = int(noise.max() - noise.min())
        elevation_range = elevation_range if elevation_range > 0 else 1
        max_physical_height = elevation_range * self.vertical_scale
        normalized = (noise - noise.min()) / elevation_range
        if self.inverted:
            # Pit: edges at 0, centre at -rise.
            normalized = 1.0 - normalized
            z_offset = -max_physical_height
            spawn_z = -max_physical_height
        else:
            z_offset = 0.0
            spawn_z = max_physical_height

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[
                self.size[0] / 2.0,
                self.size[1] / 2.0,
                max_physical_height,
                max_physical_height * self.base_thickness_ratio,
            ],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized.flatten().astype(np.float32).tolist(),
        )
        material = color_by_height(spec, noise, unique_id, normalized)
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2.0, self.size[1] / 2.0, z_offset],
            material=material,
        )
        origin = np.array([self.size[0] / 2.0, self.size[1] / 2.0, spawn_z])
        return TerrainOutput(
            origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)]
        )


def _slope_gradient_range() -> tuple[float, float]:
    """SLOPE_DEG_RANGE as the rise/run gradients mjlab wants."""
    return (
        math.tan(math.radians(SLOPE_DEG_RANGE[0])),
        math.tan(math.radians(SLOPE_DEG_RANGE[1])),
    )


def make_slopes_terrain_cfg() -> TerrainGeneratorCfg:
    """A fresh generator cfg each call — mjlab mutates these in place."""
    return TerrainGeneratorCfg(
        size=TILE_SIZE,
        border_width=BORDER_WIDTH,
        num_rows=NUM_LEVELS,
        num_cols=len(TERRAIN_PROPORTIONS),  # ignored in curriculum mode, used at play
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            "flat": terrain_gen.BoxFlatTerrainCfg(
                proportion=TERRAIN_PROPORTIONS["flat"],
            ),
            # Platform on top: spawn at the summit, every heading goes DOWN.
            "slope_down": HfFrustumSlopeTerrainCfg(
                proportion=TERRAIN_PROPORTIONS["slope_down"],
                slope_range=_slope_gradient_range(),
                platform_width=PLATFORM_WIDTH,
                vertical_scale=HFIELD_VERTICAL_SCALE,
                inverted=False,
            ),
            # Inverted: spawn on the pit floor, every heading goes UP.
            "slope_up": HfFrustumSlopeTerrainCfg(
                proportion=TERRAIN_PROPORTIONS["slope_up"],
                slope_range=_slope_gradient_range(),
                platform_width=PLATFORM_WIDTH,
                vertical_scale=HFIELD_VERTICAL_SCALE,
                inverted=True,
            ),
            "stairs_down": terrain_gen.BoxPyramidStairsTerrainCfg(
                proportion=TERRAIN_PROPORTIONS["stairs_down"],
                step_height_range=STEP_HEIGHT_RANGE,
                step_width=STEP_RUN,
                platform_width=PLATFORM_WIDTH,
                border_width=1.0,
            ),
            "stairs_up": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
                proportion=TERRAIN_PROPORTIONS["stairs_up"],
                step_height_range=STEP_HEIGHT_RANGE,
                step_width=STEP_RUN,
                platform_width=PLATFORM_WIDTH,
                border_width=1.0,
            ),
        },
        add_lights=False,
    )


def make_microduck_velocity_terrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity tracking on slopes and stairs (normal feet)."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)

    # The rough branch installed the contact-softening spec_fn, nconmax=200 and
    # the 30/50 solver iterations. Keep them: they are the NaN guard for box
    # terrain, and the stairs here are boxes.
    assert cfg.scene.spec_fn is _soften_terrain_contacts, (
        "rough branch no longer installs _soften_terrain_contacts; the terrain "
        "task depends on it"
    )

    generator = make_slopes_terrain_cfg()
    if play:
        # Random mix of every type and difficulty on a small grid, like the
        # rough task's play mode, but on our own copy of the generator cfg.
        generator.curriculum = False
        generator.num_rows = 5
        generator.num_cols = 5
    cfg.scene.terrain.terrain_generator = generator
    cfg.scene.terrain.max_init_terrain_level = MAX_INIT_LEVEL

    # See FOOT_SWING_TARGET above.
    cfg.rewards["foot_swing_height"].params["target_height"] = FOOT_SWING_TARGET

    # Terrain curriculum keyed on distance walked (mjlab's default demotes this
    # slow robot forever — see mdp_terrain).
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=mdp_terrain.terrain_levels_walk,
        params={
            "command_name": "twist",
            "promote_fraction": PROMOTE_FRACTION,
            "demote_fraction": DEMOTE_FRACTION,
            "command_threshold": CURRICULUM_COMMAND_THRESHOLD,
        },
    )

    return cfg


MicroduckSlopesRlCfg = deepcopy(MicroduckRlCfg)
MicroduckSlopesRlCfg.experiment_name = "velocity_slopes"
MicroduckSlopesRlCfg.run_name = "velocity_slopes"
