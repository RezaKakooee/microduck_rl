"""Narrow-beam terrain for the sideways beam task.

One tile = one straight beam along world +x, with nothing beside it (the
tile has no floor: stepping off means falling). mjlab's terrain generator
places the tiles: one column per beam kind, one row per difficulty level.

Tile layout (local x, m):

    0.00 - 0.30   start platform, 300 mm wide (her "ledge")
    0.30 - 2.70   the beam, width and features set by the row
    2.70 - 3.00   end platform, 300 mm wide

The beam path is a list of segments (x0, x1, h0, h1, w) or
(x0, x1, h0, h1, w, boxes). h is the top height above the start platform.
Where no segment covers x there is a gap (a pit with no bottom). A segment
with h0 != h1 is a ramp. Without `boxes` the segment is one box of width w.
With `boxes` = ((y, width, dz), ...) it is built from those boxes instead
(y off the axis, top at h + dz, dz <= 0), and w is the width the runtime
tables report:

* crowned top: a flat top w, then on each side two steps down (a rounded
  trunk seen in cross-section). w = the flat top.
* rails: two boxes left and right of the axis with nothing between them
  (his two shins). w = the outer span.

The profile is a pure function of (kind, row). The tile geometry and the
runtime tables in `mdp_beam` (spawn points, surface height under her feet)
both call `beam_profile`, so they can never disagree.

Difficulty d = row / (rows - 1), from 0 (row 0) to 1 (last row):

    width           150 mm -> 28 mm       (W_MAX - (W_MAX - W_MIN) * d, +-5 %)
    steps           up to 25 mm * d       (each step 40-100 % of that)
    ramps           up to 20 deg * d      (60-200 mm long, rise <= 40 mm)
    pits (gaps)     up to 40 mm * d long  (sometimes a step across the gap)
    bumps           up to 20 mm * d high  (humps 30-70 mm ramps, blocks, dips)

Row 0 is a flat 150 mm beam for every kind.

Columns (`KINDS`). In the first six, a narrow beam always comes with big
features. The next three break that link: the width still follows the row
(150 -> 28 mm), but every feature is at most SMALL_MAX (12 mm) high or long,
like his back:

    narrow_steps    steps only, <= 12 mm * d
    narrow_mixed    steps, ramps, bumps, pits, each <= 12 mm * d
    crowned         narrow_mixed features on a crowned top: flat top w, and
                    on each side the top drops 3-8 mm over 10-15 mm in two
                    stepped boxes

The "his_back" kind is his back in the F1w layout (scene.LAYOUT, bake
bridge_pose.json, 39 geoms), measured with local_storage/hb_dev/rl_policy/s4_fix
map_surface.py + analyse.py (convex hulls MuJoCo collides with, the band
|y| <= 27 mm her sole covers). See `HIS_BACK`. Heights and gaps scale with
d; widths blend from 150 mm to his real widths; rails close up (inner edge
x d) at low d. The tile holds several copies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

# -- tile ------------------------------------------------------------------
TILE_X = 3.0                  # along the beam (m)
TILE_Y = 0.6                  # distance between two beams (m)
X_BEAM0 = 0.30                # start platform ends, beam starts
X_END = TILE_X - 0.30         # beam ends, end platform starts
PLATFORM_W = 0.30             # start / end platform width (m)
H0 = 0.30                     # start platform top above the terrain frame (m)
THICK = 0.04                  # beam box thickness (m); max step 25 mm < THICK
NUM_ROWS = 10
TERRAIN_SEED = 1234

# -- difficulty ranges (at d = 1) -------------------------------------------
W_MAX, W_MIN = 0.150, 0.028
WIDTH_JITTER = 0.05
STEP_MAX = 0.025
RAMP_MAX_DEG = 20.0
RAMP_LEN = (0.06, 0.20)
RAMP_RISE_MAX = 0.040
PIT_MAX = 0.040
BUMP_MAX = 0.020
SMALL_MAX = 0.012             # narrow_* / crowned: every feature <= this (m)
CROWN_DROP = (0.003, 0.008)   # crowned: the top falls this much at the outer edge (m)
CROWN_SIDE = (0.010, 0.015)   # ... over this much on each side (m)
FEATURE_FRAC = (0.4, 1.0)     # each feature is 40-100 % of the row maximum
RUN = (0.18, 0.32)            # flat run before each feature (m)
H_BOUND = 0.05                # keep |h| below this (steps turn back)

# Order matters: a kind's random draws are seeded with its index, so new kinds
# go before his_back (which draws nothing) and the first six keep their tiles.
KINDS = ("flat", "steps", "ramps", "pits", "bumps", "mixed",
         "narrow_steps", "narrow_mixed", "crowned", "his_back")
PROPORTIONS = {"flat": 0.06, "steps": 0.10, "ramps": 0.09, "pits": 0.10, "bumps": 0.09, "mixed": 0.08,
               "narrow_steps": 0.10, "narrow_mixed": 0.12, "crowned": 0.12, "his_back": 0.14}
SMALL_KINDS = ("narrow_steps", "narrow_mixed", "crowned")

# his_back (F1w), mm. x runs from her ledge edge (world x = -76 mm), heights
# are relative to her ledge top (335 mm). Measured on the settled bake
# (trunk at x 66.7 mm, z 292.4 mm, pitch -0.6 deg; s4_fix/surface_f1w.json):
#
#   world x      what                                     top (mm)   across (|y|, mm)
#   -76 .. -46   hole down to his near step, 35 mm deep   -35        -
#   -46 .. -36   his feet                                 -18..-13   19-38
#   -36 ..   0   shins: two rails, gap between his legs   +0.1       11-38
#     0 ..  16   shins, narrowest                         -0.2       19-26
#    16 ..  24   knees                                    +0.7       25-38
#    24 ..  29   notch (power support, 16-19 mm down)     -          -
#    29 ..  72   trunk                                    +4.7..+3.5 flat 36 wide
#    72 .. 104   trunk and shells                         +3.4..+3.2 flat 60 wide
#   104 .. 106   trunk ends (rounded)                     +3.2..-3.6
#   106 .. 116   dip between trunk and head, 17 mm deep   -          -
#   116 .. 139   back of his head                         -6.3..+3.3 rounded, 28-34 wide
#   139 .. 161   head top                                 +3.3..0    rounded, 34-42 wide
#   161 .. 236   head slopes down 17 deg                  0..-22.6   rounded, 42 wide
#   236 .. 250   far pit (12.5 mm on the 2.5 mm grid)     -          -
#   250 ..       far ledge                                -12.5
#
# The hole is bottomless here: a foot 15 mm below the rest height already
# ends the episode (foot_off), before it reaches his step 35 mm down, which
# story.Judge fails. Items: ("gap", length) or (shape, length, h0, h1, ...):
#   ("rails", L, h0, h1, inner, outer)   two rails, |y| from inner to outer
#   ("box",   L, h0, h1, width)          one box
#   ("crown", L, h0, h1, top, side)      flat top, then -4 / -8 mm steps of side/2 each
HIS_BACK = (
    ("gap", 30.0),
    ("rails", 10.0, -16.0, -16.0, 19.0, 38.0),
    ("rails", 36.0, 0.1, 0.1, 11.0, 38.0),
    ("rails", 16.0, -0.2, -0.2, 19.0, 26.0),
    ("rails", 8.0, 0.7, 0.7, 25.0, 38.0),
    ("gap", 5.0),
    ("box", 43.0, 4.7, 3.5, 36.0),
    ("box", 32.0, 3.4, 3.2, 60.0),
    ("box", 2.5, 3.2, -3.6, 50.0),
    ("gap", 10.0),
    ("crown", 23.0, -6.3, 3.3, 28.0, 18.0),
    ("crown", 22.0, 3.3, 0.0, 34.0, 16.0),
    ("crown", 75.0, 0.0, -22.6, 42.0, 18.0),
)
HIS_CROWN_DROP = 8.0          # mm: 3 mm -> 8 mm below his top across the side strips
HIS_FAR_PITS = (12.5, 15.0)   # mm, alternating per copy (the grid reads 12.5 mm; edges +-2.5 mm)
HIS_FAR_LEDGE = -12.5         # mm
HIS_LEDGE_LEN = 0.25          # m of far ledge before the next copy


def difficulty(row: int, num_rows: int = NUM_ROWS) -> float:
    return row / max(1, num_rows - 1)


def beam_width(d: float, rng: np.random.Generator) -> float:
    w = W_MAX - (W_MAX - W_MIN) * d
    w *= rng.uniform(1.0 - WIDTH_JITTER, 1.0 + WIDTH_JITTER)
    return float(np.clip(w, W_MIN, W_MAX))


# -- cross-sections -------------------------------------------------------

def crown(top: float, drop: float, side: float) -> tuple:
    """Boxes of a crowned top: flat `top` wide, then on each side two steps
    of side/2, at -drop/2 and -drop."""
    return ((0.0, top, 0.0), (0.0, top + side, -drop / 2), (0.0, top + 2 * side, -drop))


def rails(inner: float, outer: float) -> tuple:
    """Boxes of two rails, |y| from inner to outer (nothing between them)."""
    c, w = (inner + outer) / 2, outer - inner
    return ((-c, w, 0.0), (c, w, 0.0))


def seg_boxes(seg) -> tuple:
    """(y, width, dz) of every box the segment is built from."""
    if len(seg) > 5 and seg[5]:
        return seg[5]
    return ((0.0, seg[4], 0.0),)


def _seg(x0, x1, h0, h1, w, boxes=None):
    return (x0, x1, h0, h1, w) if boxes is None else (x0, x1, h0, h1, w, tuple(boxes))


# -- features ---------------------------------------------------------------

def _feature(kind, x, h, w, d, rng, segs, cap=None, boxes=None):
    """Append one feature at x (height h). Returns the new (x, h).

    cap (m): every height and gap is at most this (the small-feature kinds)."""
    u = lambda: rng.uniform(*FEATURE_FRAC) * d
    sign = lambda: 1.0 if rng.random() < 0.5 else -1.0
    lim = lambda v: v if cap is None else float(np.clip(v, -cap, cap))
    if kind == "steps":
        dh = lim(sign() * u() * STEP_MAX) if cap is None else sign() * u() * cap
        if abs(h + dh) > H_BOUND:
            dh = -dh
        return x, h + dh
    if kind == "ramps":
        length = rng.uniform(*RAMP_LEN)
        dh = sign() * math.tan(math.radians(u() * RAMP_MAX_DEG)) * length
        dh = lim(float(np.clip(dh, -RAMP_RISE_MAX, RAMP_RISE_MAX)))
        if abs(h + dh) > H_BOUND:
            dh = -dh
        segs.append(_seg(x, x + length, h, h + dh, w, boxes))
        return x + length, h + dh
    if kind == "pits":
        gap = u() * (PIT_MAX if cap is None else cap)
        dh = 0.0
        if rng.random() < 0.3:
            dh = lim(sign() * rng.uniform(0.0, 0.5) * d * STEP_MAX)
            if abs(h + dh) > H_BOUND:
                dh = -dh
        return x + gap, h + dh
    if kind == "bumps":
        b = u() * (BUMP_MAX if cap is None else cap)
        r = rng.random()
        if r < 0.5:                                   # hump (dip if negative)
            b *= 1.0 if rng.random() < 0.75 else -1.0
            a, top = rng.uniform(0.03, 0.07), rng.uniform(0.03, 0.10)
            segs += [_seg(x, x + a, h, h + b, w, boxes), _seg(x + a, x + a + top, h + b, h + b, w, boxes),
                     _seg(x + a + top, x + 2 * a + top, h + b, h, w, boxes)]
            return x + 2 * a + top, h
        length = rng.uniform(0.04, 0.12)              # block: up then down
        segs.append(_seg(x, x + length, h + b, h + b, w, boxes))
        return x + length, h
    raise ValueError(kind)


def _his_back(d, x, h, segs, copy=0):
    """One copy of his back starting at x (her ledge edge), ledge height h."""
    mm = 1e-3
    blend = lambda real: real + (W_MAX - real) * (1.0 - d)      # widths: 150 mm at d = 0
    for item in HIS_BACK:
        if item[0] == "gap":
            x += item[1] * mm * d
            continue
        shape, length, h0, h1 = item[:4]
        a, b = x, x + length * mm
        z0, z1 = h + h0 * mm * d, h + h1 * mm * d
        if shape == "box":
            w = blend(item[4] * mm)
            segs.append(_seg(a, b, z0, z1, w))
        elif shape == "rails":
            inner, outer = item[4] * mm * d, item[5] * mm + (W_MAX / 2 - item[5] * mm) * (1.0 - d)
            segs.append(_seg(a, b, z0, z1, 2 * outer, rails(inner, outer)))
        elif shape == "crown":
            top = blend(item[4] * mm)
            segs.append(_seg(a, b, z0, z1, top, crown(top, HIS_CROWN_DROP * mm * d, item[5] * mm)))
        else:
            raise ValueError(shape)
        x = b
    x += HIS_FAR_PITS[copy % len(HIS_FAR_PITS)] * mm * d
    return x, h + HIS_FAR_LEDGE * mm * d


def beam_profile(kind: str, row: int, num_rows: int = NUM_ROWS, seed: int = TERRAIN_SEED):
    """Segments of one tile, platforms included (see the module docstring)."""
    d = difficulty(row, num_rows)
    rng = np.random.default_rng([seed, KINDS.index(kind), row])
    segs = [(0.0, X_BEAM0, 0.0, 0.0, PLATFORM_W)]
    x, h = X_BEAM0, 0.0
    if kind == "his_back":
        copy = 0
        while True:
            x_next, _ = _his_back(d, x, h, [], copy)
            if x_next + HIS_LEDGE_LEN > X_END:
                break
            x, h = _his_back(d, x, h, segs, copy)
            segs.append((x, x + HIS_LEDGE_LEN, h, h, PLATFORM_W))
            x += HIS_LEDGE_LEN
            copy += 1
        segs.append((x, X_END, h, h, PLATFORM_W))
    else:
        w = beam_width(d, rng)
        boxes, cap = None, None
        if kind in SMALL_KINDS:
            cap = SMALL_MAX
        if kind == "crowned" and d > 0.0:
            boxes = crown(w, rng.uniform(*CROWN_DROP), rng.uniform(*CROWN_SIDE))
        margin = 0.30                                 # room for the last feature + a run
        while x < X_END - margin:
            run = rng.uniform(*RUN)
            segs.append(_seg(x, x + run, h, h, w, boxes))
            x += run
            if kind == "flat" or d == 0.0:
                continue
            if kind in ("mixed", "narrow_mixed", "crowned"):
                k = ("steps", "ramps", "pits", "bumps")[rng.integers(4)]
            elif kind == "narrow_steps":
                k = "steps"
            else:
                k = kind
            x, h = _feature(k, x, h, w, d, rng, segs, cap=cap, boxes=boxes)
        if x < X_END:
            segs.append(_seg(x, X_END, h, h, w, boxes))
    segs.append((X_END, TILE_X, h, h, PLATFORM_W))
    return _merge([s for s in segs if s[1] - s[0] > 1e-6])


def _merge(segs):
    out = []
    for s in segs:
        if out:
            p = out[-1]
            flat = p[2] == p[3] and s[2] == s[3]
            same = p[4] == s[4] and seg_boxes(p) == seg_boxes(s)
            if flat and abs(p[1] - s[0]) < 1e-9 and abs(p[3] - s[2]) < 1e-9 and same:
                out[-1] = (p[0], s[1], p[2], p[3], *p[4:])
                continue
        out.append(s)
    return out


def sample_profile(segs, dx: float = 0.005, tile_x: float = TILE_X):
    """Top height, width and gap mask on a regular x grid (index i = x / dx).

    In a gap, `top` is the lower of the two tops beside it: the height a foot
    bridging the gap rests at. `gap` marks those cells."""
    xs = np.arange(0.0, tile_x + 0.5 * dx, dx)
    top = np.full(len(xs), np.nan)
    width = np.zeros(len(xs))
    for s in segs:
        x0, x1, h0, h1, w = s[:5]
        m = (xs >= x0 - 1e-9) & (xs <= x1 + 1e-9)
        t = (xs[m] - x0) / max(x1 - x0, 1e-9)
        top[m] = np.fmax(top[m], h0 + t * (h1 - h0))
        width[m] = np.maximum(width[m], w)
    gap = np.isnan(top)
    if gap.any():
        idx = np.arange(len(xs))
        left = np.maximum.accumulate(np.where(~gap, idx, 0))
        right = np.minimum.accumulate(np.where(~gap, idx, len(xs) - 1)[::-1])[::-1]
        top[gap] = np.minimum(top[left[gap]], top[right[gap]])
    return xs, top, width, gap


def spawn_points(segs, k: int, x_max: float, stance: float = 0.07, dx: float = 0.005,
                 seed: int = 0):
    """Up to k trunk x positions where both feet stand on one flat segment:
    [x - stance, x + stance] inside a flat segment, x <= x_max. Deterministic."""
    cand = []
    for s in segs:
        x0, x1, h0, h1 = s[:4]
        if h0 != h1:
            continue
        lo, hi = max(x0 + stance, X_BEAM0 - 0.18), min(x1 - stance, x_max)
        if hi >= lo:
            cand += list(np.arange(lo, hi + 1e-9, dx))
    cand = np.array(cand if cand else [X_BEAM0 - 0.15])
    rng = np.random.default_rng(seed)
    if len(cand) > k:
        cand = np.sort(rng.choice(cand, size=k, replace=False))
    return np.resize(cand, k)


@dataclass(kw_only=True)
class BeamLaneTerrainCfg(SubTerrainCfg):
    """One beam per tile; see the module docstring."""

    kind: str = "flat"
    num_rows: int = NUM_ROWS
    seed: int = TERRAIN_SEED

    def profile(self, row: int):
        return beam_profile(self.kind, row, self.num_rows, self.seed)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng   # the profile has its own seed: (seed, kind, row)
        row = int(np.clip(math.floor(difficulty * self.num_rows + 1e-6), 0, self.num_rows - 1))
        body = spec.body("terrain")
        yc = self.size[1] / 2.0
        geoms = []
        wood = 0.35 + 0.4 * (row / max(1, self.num_rows - 1))
        for seg in self.profile(row):
            x0, x1, h0, h1 = seg[:4]
            dx, dz = x1 - x0, h1 - h0
            ang = math.atan2(dz, dx)
            n = (-math.sin(ang), math.cos(ang))       # top-face normal in (x, z)
            for y, bw, top_dz in seg_boxes(seg):
                shade = max(0.7, 1.0 + 15.0 * top_dz)      # lower crown steps are darker
                rgba = (wood * shade, 0.42 * shade, 0.30 * shade, 1.0)
                if dz == 0.0:
                    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                      size=(dx / 2, bw / 2, THICK / 2),
                                      pos=((x0 + x1) / 2, yc + y, H0 + h0 + top_dz - THICK / 2))
                else:
                    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                      size=(math.hypot(dx, dz) / 2, bw / 2, THICK / 2),
                                      pos=((x0 + x1) / 2 - THICK / 2 * n[0], yc + y,
                                           H0 + (h0 + h1) / 2 + top_dz - THICK / 2 * n[1]),
                                      quat=(math.cos(-ang / 2), 0.0, math.sin(-ang / 2), 0.0))
                g.rgba = rgba
                geoms.append(TerrainGeometry(geom=g, color=rgba))
        # Origin: beam start, on its axis, at the start platform top.
        return TerrainOutput(origin=np.array([0.0, yc, H0]), geometries=geoms)


def make_beam_terrain_cfg(num_rows: int = NUM_ROWS) -> TerrainGeneratorCfg:
    """A fresh generator cfg each call (mjlab mutates these in place)."""
    return TerrainGeneratorCfg(
        # Fixed seed: the generator draws difficulty = (row + U) / rows, and a
        # tile reads its row back as floor(difficulty * rows + 1e-6). With this
        # seed every tile gets its own row (test_sideways_beam_cfg).
        seed=TERRAIN_SEED,
        size=(TILE_X, TILE_Y),
        border_width=0.5,
        num_rows=num_rows,
        num_cols=len(KINDS),
        curriculum=True,                 # one column per kind, one row per level
        difficulty_range=(0.0, 1.0),     # row = floor(difficulty * rows) exactly
        color_scheme="none",
        sub_terrains={k: BeamLaneTerrainCfg(kind=k, num_rows=num_rows, proportion=PROPORTIONS[k])
                      for k in KINDS},
        add_lights=False,
    )
