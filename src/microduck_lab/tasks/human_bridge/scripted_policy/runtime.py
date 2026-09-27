"""Keep each small gait network on one CPU thread."""
from unittest.mock import patch

import onnxruntime as ort

from microduck_lab.tasks.human_bridge.scripted_policy.world import Brain

SESSION = ort.InferenceSession


def session(*args, **kwargs):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    kwargs['sess_options'] = options
    return SESSION(*args, **kwargs)


def brain(*args, **kwargs):
    # Scope the constructor override to this process and this construction.
    with patch.object(ort, 'InferenceSession', session):
        return Brain(*args, **kwargs)


def require_policy(path):
    """Refuse missing policies or paths outside Codex's input folder."""
    from pathlib import Path
    policy = Path(path).resolve()
    folder = Path(__file__).resolve().parents[5] / 'local_storage/hb_dev/scripted_policy/policies'
    if not policy.is_relative_to(folder):
        raise ValueError(f'Policy must be supplied inside {folder}')
    if not policy.is_file():
        raise FileNotFoundError(f'Missing Codex policy: {policy}. Supply it here before starting a trial.')
    return policy
