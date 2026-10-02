"""Search the pumping parameters, then test the best ones for robustness.

1. grid: every combination in GRID for `--grid-seconds`. Score = mean
   flight time in the last 5 s; a fall, or drifting more than MAX_DRIFT_MM
   from the bed centre, scores 0.
2. robustness: the best `--top` settings, each under every VARIANT for
   `--seconds`. A variant passes if the duck does not fall, stays within
   MAX_DRIFT_MM, and its mean flight in the last 5 s is at least
   MIN_FLIGHT_MS.

    uv run python -m microduck_lab.tasks.trampoline.search --procs 5
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

from microduck_lab import paths
from microduck_lab.tasks.trampoline.pump import PumpParams, run

OUT = Path(paths.REPO) / "local_storage" / "hb_dev" / "trampoline"
MIN_FLIGHT_MS = 100.0
MAX_DRIFT_MM = 80.0
TAIL_S = 5.0

GRID = dict(amp=(15.0, 17.5, 20.0), lead=(60.0, 90.0, 120.0, 150.0),
            hip_kp=(1.0, 2.0, 3.0), hip_kd=(0.0, 0.05), place_i=(0.02, 0.05, 0.1))

# (name, change to the pump params, change to the world)
VARIANTS = [
    ("nominal", {}, {}),
    ("start standing", dict(drop=0.0), {}),
    ("start drop 50 mm", dict(drop=50.0), {}),
    ("battery 7.0 V", {}, dict(vin=7.0)),
    ("battery 8.0 V", {}, dict(vin=8.0)),
    ("bed rebound 0.4", {}, dict(rebound=0.4)),
    ("bed rebound 0.8", {}, dict(rebound=0.8)),
    ("bed sag 15 mm", {}, dict(sag_mm=15.0)),
    ("bed sag 30 mm", {}, dict(sag_mm=30.0)),
    ("bed mass 60 g", {}, dict(bed_mass=0.06)),
    ("phase error -20 deg", "lead-20", {}),
    ("phase error +20 deg", "lead+20", {}),
]


SCORE = "flight_ms"   # or "robot_heights" (set by --score)


def evaluate(job):
    params, world_kw, seconds = job[:3]
    score_key, max_drift = job[3] if len(job) > 3 else (SCORE, MAX_DRIFT_MM)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            r, _ = run(PumpParams(**params), seconds=seconds, tail_s=TAIL_S, **world_kw)
        except ValueError as e:
            return dict(params=params, world=world_kw, error=str(e), score=0.0)
    hs = [f["robot_heights"] for f in r.pop("flights")]
    r["peak_robot_heights"] = max(hs, default=0.0)
    r.pop("params")
    score = 0.0 if r["fell"] or r["max_drift_mm"] > max_drift else r["tail"][score_key]
    return dict(params=params, world=world_kw, score=score, **r)


def variant_params(p, change):
    if change == "lead-20":
        return replace(p, lead=p.lead - 20)
    if change == "lead+20":
        return replace(p, lead=p.lead + 20)
    return replace(p, **change)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--grid-seconds", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--params", default=None,
                    help="JSON list of pump params: skip the grid, test only these for robustness")
    ap.add_argument("--tag", default="", help="suffix for the report files")
    ap.add_argument("--grid", default=None, help="JSON {param: [values]} instead of GRID")
    ap.add_argument("--base", default="{}", help="JSON pump params fixed for every grid point")
    ap.add_argument("--world", default="{}", help="JSON world settings (bed, frame) for every run")
    ap.add_argument("--score", default="flight_ms", choices=("flight_ms", "robot_heights"))
    ap.add_argument("--max-drift", type=float, default=MAX_DRIFT_MM)
    ap.add_argument("--show", type=int, default=10, help="grid rows to print")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base, world = json.loads(a.base), json.loads(a.world)
    how = (a.score, a.max_drift)

    if a.params:
        best = [PumpParams(**p) for p in json.loads(a.params)]
    else:
        spec = json.loads(a.grid) if a.grid else GRID
        keys = list(spec)
        jobs = [(dict(base, **dict(zip(keys, v))), world, a.grid_seconds, how) for v in itertools.product(*spec.values())]
        with Pool(a.procs) as pool:
            grid = list(pool.imap(evaluate, jobs))
        grid.sort(key=lambda g: -g.get("score", 0.0))
        (OUT / f"search_grid{a.tag}.json").write_text(json.dumps(grid, indent=1))
        print(f"grid ({len(grid)} runs, {a.grid_seconds:g} s, score {a.score}): best {a.show}")
        for g in grid[:a.show]:
            if g.get("error"):
                continue
            varied = {k: g["params"][k] for k in keys}
            print(f"  {varied} score {g['score']} | fell {g['fell']} at {g['survived_s']} s | peak {g['peak_robot_heights']} heights"
                  f" | drift {g['max_drift_mm']} tilt {g['max_tilt_deg']} sag {g['max_bed_sag_mm']} bottomed {g['bottomed_out']} | tail {g['tail']}")
        best = [PumpParams(**g["params"]) for g in grid[:a.top]]
    if not best:
        return
    jobs = [(asdict(variant_params(p, pc)), dict(world, **wc), a.seconds, how) for p in best for _, pc, wc in VARIANTS]
    with Pool(a.procs) as pool:
        rob = list(pool.imap(evaluate, jobs))
    table = []
    for i, p in enumerate(best):
        rows = rob[i * len(VARIANTS):(i + 1) * len(VARIANTS)]
        passed = [(name, row) for (name, _, _), row in zip(VARIANTS, rows)
                  if not row.get("error") and not row["fell"] and row["max_drift_mm"] <= a.max_drift
                  and row["tail"]["flight_ms"] >= MIN_FLIGHT_MS]
        table.append(dict(params=asdict(p), passed=len(passed), of=len(VARIANTS),
                          variants={name: row for (name, _, _), row in zip(VARIANTS, rows)}))
    table.sort(key=lambda t: -t["passed"])
    (OUT / f"search_robust{a.tag}.json").write_text(json.dumps(table, indent=1))
    print(f"robustness ({a.seconds:g} s; pass = no fall and flight >= {MIN_FLIGHT_MS:g} ms in the last {TAIL_S:g} s)")
    for t in table:
        print(" ", t["params"], f"passed {t['passed']}/{t['of']}")
        for name, row in t["variants"].items():
            if row.get("error"):
                print(f"    {name:22s} error {row['error']}")
            else:
                print(f"    {name:22s} fell {str(row['fell']):5s} at {row['survived_s']:5.1f} s | flight {row['tail']['flight_ms']:5.1f} ms"
                      f" rise {row['tail']['com_rise_mm']:5.1f} mm = {row['tail']['robot_heights']:.2f} heights | drift {row['max_drift_mm']:5.1f} mm"
                      f" | tilt {row['max_tilt_deg']:4.1f} | bottomed {row['bottomed_out']}")


if __name__ == "__main__":
    main()
