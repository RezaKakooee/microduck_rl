"""Film a short run and compare it with the saved pre-copy trace."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from microduck_lab.tasks.human_bridge.scripted_policy.controller import Settings
from microduck_lab.tasks.human_bridge.scripted_policy.run import run
from microduck_lab.tasks.human_bridge.scripted_policy.baked_surface import surface_table


def main():
    folder = Path(__file__).resolve().parent
    owned = sorted(folder.glob('*.py')) + [folder / 'bridge_pose.json']
    manifest = {str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in owned}
    (folder / 'protected_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    surface_table()  # The optional observations must read our newly baked pose.
    cfg = Settings(**json.loads((folder / 'sequence_validation.json').read_text())[0])
    result = run(cfg, seconds=6.)
    video = Path(result['video'])
    with np.load(video.with_suffix('.npz')) as data:
        actual = data['trace']
    reference_path = Path('videos/human_bridge/scripted_policy/previous/scripted_try111.npz')
    with np.load(reference_path) as data:
        expected = data['trace'][:len(actual)]
    error = float(np.max(np.abs(actual-expected)))
    forbidden = [name for name in sys.modules if name.startswith((
        'microduck_lab.tasks.human_bridge.rl_policy', 'microduck_lab.tasks.human_bridge.crawl_policy'))]
    passed = (result['failure'] == 'time limit' and result['exception'] is None
              and abs(result['end']['time']-6.) < 1e-8 and error <= 1e-12 and not forbidden)
    summary = dict(smoke_pass=passed, full_crossing_pass=result['success'],
                   note='A six-second smoke test ends before a full crossing; time limit is expected.',
                   reference=str(reference_path), trace_max_abs_error=error,
                   forbidden_modules=forbidden, trial=result['video'],
                   min_up=result['min_up'], min_z=result['min_z'],
                   max_prediction_qpos_error=result['max_prediction_qpos_error'])
    (folder / 'ISOLATION_CHECK.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    if not passed:
        raise SystemExit('Isolation smoke comparison failed; see ISOLATION_CHECK.json')


if __name__ == '__main__':
    main()
