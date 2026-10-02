"""Passive test: the duck stands still on the trampoline, or is dropped on it.

No pumping. The bed is the default one (20 mm sag under the duck; a rigid
weight of the duck's mass dropped 20 mm bounces back 60% of that) unless a
case says otherwise. The rebound is a guess until the real bed is measured.

The legs hold the stand pose. Two ways to hold it:
- `upright`: stand pose plus the IMU upright PD from the jump task (ankles
  for pitch, hip rolls for roll). Fixed targets alone fall over in about 1 s
  on flat ground;
- `fixed`: stand pose only, no feedback.

For each case it reports whether the duck falls, every flight (a time with no
contact at all), how high the centre of mass rises in it, the deepest bed sag
and the largest tilt.

    uv run python -m microduck_lab.tasks.trampoline.passive              # numbers
    MUJOCO_GL=egl uv run python -m microduck_lab.tasks.trampoline.passive --video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.jump.world import Upright
from microduck_lab.tasks.trampoline.world import TrampolineWorld, bounce_report, COL

REPORT = Path(paths.REPO) / "local_storage" / "hb_dev" / "trampoline" / "passive_report.json"

CASES = {
    "stand":        dict(drop_mm=0.0, control="upright"),
    "drop20":       dict(drop_mm=20.0, control="upright"),
    "drop20_fixed": dict(drop_mm=20.0, control="fixed"),
    "drop50":       dict(drop_mm=50.0, control="upright"),
    "drop20_floor": dict(drop_mm=20.0, control="upright", rigid=True),
}
# Numbers only, no film.
EXTRA = {
    # Same as drop20 with half the physics step: the numbers must not move much.
    "drop20, 0.5 ms step": dict(drop_mm=20.0, control="upright", timestep=0.0005, decimation=40),
    # How much a bouncier or deader bed changes the duck's own bounce.
    "drop20, bed rebound 0.4": dict(drop_mm=20.0, control="upright", rebound=0.4),
    "drop20, bed rebound 0.8": dict(drop_mm=20.0, control="upright", rebound=0.8),
}


def run(drop_mm, control, seconds=5.0, film=None, **world_kw):
    w = TrampolineWorld(**world_kw)
    w.place(w.stand, drop_mm=drop_mm)
    up = Upright(w) if control == "upright" else None
    if film is not None:
        film = film(w)
        w.on_substep = film
    for _ in range(int(round(seconds / w.control_dt))):
        w.step(up(w.stand) if up else w.stand)
    r = bounce_report(w.log)
    r["bed"] = dict(rigid=w.rigid, sag_mm=w.sag_mm, rebound=w.rebound, k_N_per_m=round(w.k, 1),
                    damping_Ns_per_m=round(w.c, 3), hz_with_robot=round(w.bed_hz, 2), bed_mass_kg=w.bed_mass)
    r["timestep_ms"] = w.model.opt.timestep * 1000
    return r, np.array(w.log), film


def summary(name, drop_mm, r):
    # The first flight of a drop is the drop itself, not a bounce.
    flights = r["flights"][1:] if drop_mm > 0 else r["flights"]
    fell = f"yes, {r['survived_s']:.2f} s ({r['fall_reason']})" if r["fell"] else "no"
    heights = ", ".join(f"{f['com_rise_mm']:.1f}" for f in flights) or "-"
    times = ", ".join(f"{f['flight_ms']:.0f}" for f in flights) or "-"
    return (f"| {name} | {drop_mm:g} | {fell} | {len(flights)} | {heights} | {times} | "
            f"{r['max_bed_sag_mm']} | {r['max_tilt_deg']} | {r['max_foot_depth_mm']} |")


def plot(logs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(logs), 1, figsize=(9, 2.2 * len(logs)), sharex=True)
    for ax, (name, a) in zip(np.atleast_1d(axes), logs.items()):
        keep = a[:, 0] <= 2.0
        t = a[keep, 0]
        ax.plot(t, 1000 * (a[keep, COL["com_z"]] - a[0, COL["com_z"]]), label="CoM height (mm, from start)")
        ax.plot(t, 1000 * a[keep, COL["bed_z"]], label="bed (mm, from rest)")
        air = (a[keep, COL["n_feet"]] == 0) & (a[keep, COL["n_other"]] == 0)
        ax.fill_between(t, 0, 1, where=air, transform=ax.get_xaxis_transform(), color="tab:green",
                        alpha=0.2, label="in the air")
        ax.set_title(name, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="store_true", help="film every case (needs MUJOCO_GL=egl)")
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()

    film = None
    if a.video:
        from microduck_lab.tasks.trampoline.video import BedFilm
        film = lambda w, slow: BedFilm(w, slow_window=slow)   # noqa: E731

    report, logs, lines = {}, {}, []
    for name, case in CASES.items():
        slow = (0.0, 1.0) if case["drop_mm"] > 0 else None
        r, log, f = run(seconds=a.seconds, film=(lambda w: film(w, slow)) if film else None, **case)
        report[name] = dict(case=case, **r)
        logs[name] = log
        lines.append(summary(name, case["drop_mm"], r))
        if f is not None:
            report[name]["videos"] = [str(p) for p in f.write(f"passive_{name}")]
            print(name, "->", *report[name]["videos"], flush=True)

    for name, case in EXTRA.items():
        r, _, _ = run(seconds=a.seconds, **case)
        report[name] = dict(case=case, **r)
        lines.append(summary(name, case["drop_mm"], r))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    if a.video:
        from microduck_lab.tasks.trampoline.video import next_free
        png = next_free("passive_trace", ".png")
        plot(logs, png)
        print("plot ->", png)

    print("| case | drop mm | fell | bounces | CoM rise per bounce (mm) | flight (ms) | max sag mm | max tilt deg | max foot depth mm |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
