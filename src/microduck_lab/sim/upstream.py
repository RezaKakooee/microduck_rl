"""The one place that reaches into upstream `scripts/`.

`scripts/infer_policy.py` is upstream and is not part of any package, so it
cannot be imported normally. Every script of ours used to solve that with its
own `sys.path.insert(0, dirname(__file__))`, 32 copies of the same line. The
hack lives here now, once, and nothing else needs it.

If upstream ever moves `infer_policy` into `mjlab_microduck`, this file is the
only thing to change.
"""

from __future__ import annotations

import sys

from microduck_lab.paths import REPO

_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from infer_policy import (  # noqa: E402
    PolicyInference,
    DEFAULT_POSE,
    MICRODUCK_XML,
    MICRODUCK_BALL_XML,
    MICRODUCK_ROLLERS_XML,
)

__all__ = [
    "PolicyInference",
    "DEFAULT_POSE",
    "MICRODUCK_XML",
    "MICRODUCK_BALL_XML",
    "MICRODUCK_ROLLERS_XML",
]
