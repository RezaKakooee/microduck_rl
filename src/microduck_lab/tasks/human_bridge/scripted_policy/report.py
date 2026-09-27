"""Write an index of every filmed attempt, including incomplete attempts."""
import argparse
import json
from pathlib import Path
import numpy as np

from microduck_lab.tasks.human_bridge.scripted_policy.metrics import audit


def report(folder, output):
    folder, output = Path(folder), Path(output)
    stems = sorted({p.with_suffix("") for p in folder.glob("scripted_try*.*")
                    if p.suffix in (".mp4", ".json", ".lock", ".npz")},
                   key=lambda p: int(p.name.removeprefix("scripted_try")))
    lines = ["# Filmed attempts", "",
             "The JSON beside each video contains the settings and measurements.",
             "The NPZ contains the 50 Hz trace and the rendered joint states.",
             "A failure tail is shown for one second after the scored end.", "",
             "| Try | Mode | Start offset x/y (mm) | Furthest trunk x (m) | Min up | Min z (m) | Result | Video |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    successes = 0
    completed = 0
    for stem in stems:
        number = stem.name.removeprefix("scripted_try")
        video = stem.with_suffix(".mp4").resolve()
        d = (json.loads(stem.with_suffix(".json").read_text())
             if stem.with_suffix(".json").exists() else None)
        if d is None or not d.get('completed', True):
            lines.append(f"| {number} | — | — | — | — | — | Incomplete | [video]({video}) |")
            continue
        completed += 1
        successes += d['success']
        result = "PASS" if d['success'] else d['failure']
        if stem.with_suffix('.npz').exists():
            checks = d.get('task_audit')
            if checks is None:
                # Early trials predate saved audits and used the original ledges.
                # Never judge an old video using today's editable scene heights.
                layout = d.get('layout', {})
                height = min(layout.get('near_z', .329), layout.get('far_z', .315)) + .05
                edge = layout.get('gap_end', .1) + layout.get('shelf_len', .15)
                with np.load(stem.with_suffix('.npz')) as data:
                    checks = audit(data['trace'], drop_z=height, far_edge=edge)
            if checks['task_limits'] and not d['success']:
                result += f"; task limits met for final {checks['final_clearance_seconds']:.2f} s"
        lines.append(f"| {number} | {d['settings'].get('mode', 'policy')} | "
                     f"{1000*d['dx']:.0f}/{1000*d['dy']:.0f} | {d['furthest_x']:.3f} | "
                     f"{d['min_up']:.3f} | {d['min_z']:.3f} | {result} | [video]({video}) |")
    lines[2:2] = [f"{successes} passes in {completed} completed trials. "
                   "These include controller development; they are not a robustness estimate.", ""]
    output.write_text("\n".join(lines)+"\n")
    print(f"{output}: {successes}/{completed} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="videos/human_bridge/scripted_policy")
    parser.add_argument("--output", default=str(Path(__file__).with_name("ATTEMPTS.md")))
    args = parser.parse_args()
    report(args.folder, args.output)
