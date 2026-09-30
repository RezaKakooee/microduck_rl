"""Film independent motor-only probes in isolated worker processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from microduck_lab.rl.scripted_policy.salmon_jump.run import Candidate, run


def evaluate(payload):
    candidate, seconds = payload
    return run(Candidate(**candidate), seconds=seconds, width=640)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=9.)
    args = parser.parse_args()
    cases = json.loads(args.config.read_text())
    # Validate the complete batch before reserving videos or starting workers.
    for case in cases:
        Candidate(**case)
    # Spawn: each EGL context, motor model and standing policy is process-local.
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(evaluate, [(c, args.seconds) for c in cases]):
            print('BATCH_RESULT', result['video'], result['success'], flush=True)


if __name__ == '__main__':
    main()
