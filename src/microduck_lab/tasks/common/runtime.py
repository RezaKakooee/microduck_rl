"""Run the pretrained standing policy on one CPU thread (used by the salmon jump)."""
from unittest.mock import patch

import onnxruntime as ort

from microduck_lab.tasks.common.world import Brain

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
