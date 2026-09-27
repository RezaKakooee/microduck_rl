"""Summarize a fixed five-start test from its saved results and traces."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    rows, settings, sources, controllers, policies = [], [], [], [], []
    for path in map(Path, paths):
        result = json.loads(path.read_text())
        if not result.get('completed', True):
            raise ValueError(f'{path}: incomplete trial; exclude it from the completed validation')
        settings.append(result['settings'])
        sources.append(result.get('input_sha256'))
        names = ('controller.py', 'preview.py', 'sequence.py', 'predict_workers.py', 'metrics.py')
        controllers.append({name: result.get('source_sha256', {}).get(name) for name in names})
        policies.append(result.get('policy_sha256'))
        with np.load(path.with_suffix('.npz')) as data:
            trace = data['trace']
        support = trace[:, 19] if trace.shape[1] > 19 else None
        row = dict(trial=path.stem, dx=result['dx'], dy=result['dy'],
                   success=result['success'], failure=result['failure'],
                   min_up=result['min_up'], min_z=result['min_z'],
                   max_he_up=result['max_he_up'], end=result['end'],
                   task_audit=result.get('task_audit'),
                   video=str(path.with_suffix('.mp4').resolve()),
                   prediction_error=result.get('max_prediction_qpos_error'))
        if support is not None:
            dt = np.diff(trace[:, 0], prepend=0.)
            row['nonfoot_support_seconds'] = float(dt[support > 1.].sum())
            row['final_3s_max_nonfoot_force'] = float(
                support[trace[:, 0] >= trace[-1, 0]-3.].max())
        rows.append(row)
    expected = {(0., 0.), (-.02, -.01), (-.02, .01), (.02, -.01), (.02, .01)}
    offsets = [(r['dx'], r['dy']) for r in rows]
    known_sources = [s for s in sources if s is not None]
    return dict(
        trials=rows,
        passed=sum(r['success'] for r in rows), total=len(rows),
        five_starts_complete=len(offsets) == 5 and set(offsets) == expected,
        same_settings=bool(settings) and all(s == settings[0] for s in settings),
        same_controller=bool(controllers) and all(controllers[0].values()) and all(
            c == controllers[0] for c in controllers),
        same_policy=bool(policies) and policies[0] is not None and all(
            p == policies[0] for p in policies),
        input_hashes_complete=bool(sources) and all(s is not None for s in sources),
        known_input_hashes_match=bool(known_sources) and all(
            s == known_sources[0] for s in known_sources),
    )


def markdown(result):
    lines = ['# Fixed controller validation', '',
             f"{result['passed']}/{result['total']} runs passed the runner's checks.", '',
             f"All five requested starts present: {result['five_starts_complete']}.",
             f"Identical controller settings: {result['same_settings']}.",
             f"Identical controller code and gait policy: {result['same_controller'] and result['same_policy']}.",
             f"Shared input hashes present for every run: {result['input_hashes_complete']}.",
             f"Available shared input hashes match: {result['known_input_hashes_match']}.", '',
             'Body support means normal force above 1 N on parts other than her feet.',
             'A numerical pass does not imply uninterrupted walking. Watch the videos.', '',
             '| Trial | Start x/y (mm) | Result | Min up | Min z (m) | His max abs(up) | Body support (s) | Final 3 s body force max (N) |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in result['trials']:
        outcome = 'PASS' if row['success'] else row['failure']
        support = row.get('nonfoot_support_seconds')
        force = row.get('final_3s_max_nonfoot_force')
        support_text = f'{support:.2f}' if support is not None else 'unknown'
        force_text = f'{force:.2f}' if force is not None else 'unknown'
        lines.append(f"| [{row['trial']}]({row['video']}) | "
                     f"{row['dx']*1000:.0f}/{row['dy']*1000:.0f} | {outcome} | "
                     f"{row['min_up']:.3f} | {row['min_z']:.4f} | "
                     f"{row['max_he_up']:.3f} | {support_text} | {force_text} |")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', help='The five trial JSON paths')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('VALIDATION.md'))
    args = parser.parse_args()
    result = summarize(args.results)
    args.output.write_text(markdown(result))
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"{args.output}: {result['passed']}/{result['total']} passed")
