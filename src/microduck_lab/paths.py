"""Every path our code needs, worked out once.

Before this module each script found the repo by counting `os.path.dirname`
calls up from its own file. Three scripts did it at three different depths, so
moving a file one level broke it silently -- the scene XML simply was not
found. Import from here instead; nothing else should count directories.
"""

from __future__ import annotations

import os
from pathlib import Path

# paths.py lives at src/microduck_lab/paths.py, so the repo is two levels up.
LAB = Path(__file__).resolve().parent
REPO = LAB.parent.parent

# Ours.
MODELS = LAB / "models"
TASKS = LAB / "tasks"

# Upstream, and we only read from it.
ROBOT = REPO / "src" / "mjlab_microduck" / "robot" / "microduck"
ASSETS = ROBOT / "assets"

# Outputs.
VIDEOS = Path(os.environ.get("MICRODUCK_VIDEOS", REPO / "videos"))

# The pretrained ONNX policies live in the sibling `microduck` checkout. The
# env var wins so a different checkout, or a copy on a compute node, needs no
# code change.
POLICIES = Path(os.environ.get("POLICIES", REPO.parent / "microduck" / "policies"))

# The ONNX files every scripted task loads by name.
WALKING = POLICIES / "alpha_walking.onnx"
STAND = POLICIES / "alpha_stand.onnx"
SITSTAND = POLICIES / "alpha_sitstand.onnx"
GROUND_PICK = POLICIES / "alpha_ground_pick.onnx"


def model(name: str) -> str:
    """Absolute path to one of our scene or robot XML files."""
    return str(MODELS / name)


def robot(name: str) -> str:
    """Absolute path to an upstream robot or scene XML."""
    return str(ROBOT / name)


def video(task: str, name: str) -> str:
    """Absolute path to an output file, under that task's own folder.

    The real files live in `local_storage/videos/<task>/`; `videos/` at the
    repo root is a symlink to it, so older paths still resolve.
    """
    folder = VIDEOS / task
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)
