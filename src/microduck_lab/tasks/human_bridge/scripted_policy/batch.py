"""Run independent filmed starts together within one GPU allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices', type=int, nargs='+', required=True, choices=range(5))
    parser.add_argument('--config', default=str(Path(__file__).with_name('sequence_validation.json')))
    parser.add_argument('--seconds', type=float, default=18.)
    args = parser.parse_args()
    if len(set(args.indices)) != len(args.indices):
        parser.error('Each start index must occur once')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('MUJOCO_GL') != 'egl':
        parser.error('Run through video.sbatch on a GPU node')
    import json
    cases = json.loads(Path(args.config).read_text())
    cpus = max(max(1, c.get('preview_workers', 0)) + 2 for c in cases)
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) < cpus * len(args.indices):
        parser.error(f'Request at least {cpus * len(args.indices)} CPUs for this batch')

    def run(index):
        command = [sys.executable, '-m', 'microduck_lab.tasks.human_bridge.scripted_policy.run',
                   '--config', args.config, '--suite', '--start-index', str(index),
                   '--seconds', str(args.seconds)]
        return subprocess.run(command, check=False).returncode

    with ThreadPoolExecutor(max_workers=len(args.indices)) as pool:
        codes = list(pool.map(run, args.indices))
    if any(codes):
        raise SystemExit(f'Trial process exit codes: {dict(zip(args.indices, codes))}')


if __name__ == '__main__':
    main()
