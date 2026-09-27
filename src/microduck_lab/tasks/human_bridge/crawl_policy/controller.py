"""Purely scripted crawl controller for Microduck sister on the human bridge.

Uses the fast crawl skill from tasks/crawl (crawl_gait.npy) with BAM actuators.
Zero RL policies or ONNX models used.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L

J = {name: i for i, name in enumerate(SERVOS)}

JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')

# Nominal belly-down resting crawl pose
CRAWL_BASE = np.zeros(len(SERVOS), dtype=np.float64)
for name, val in {
    'left_hip_pitch': -1.20, 'left_knee': -0.60, 'left_ankle': 0.0,
    'right_hip_pitch': 1.20, 'right_knee': 0.60, 'right_ankle': 0.0,
    'neck_pitch': 0.241, 'head_pitch': -0.135
}.items():
    CRAWL_BASE[J[name]] = val


def smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


class SisterScriptedController:
    """Scripted crawl controller executing the fast crawl skill."""

    def __init__(self, duck):
        self.duck = duck
        self.phase = "wait"
        self.t_start_phase = 0.0
        self.target = CRAWL_BASE.copy()

        # Load the pre-searched crawl gait parameters (fast crawl skill)
        gait_path = paths.REPO / "local_storage" / "gaits" / "crawl_gait.npy"
        if not gait_path.exists():
            gait_path = Path(__file__).resolve().parents[5] / "local_storage" / "gaits" / "crawl_gait.npy"
        self.params = np.load(gait_path)
        self.freq = float(self.params[-3])
        self.neck_pitch = float(self.params[-2])
        self.head_pitch = float(self.params[-1])

    def set_phase(self, phase: str, t: float):
        self.phase = phase
        self.t_start_phase = t

    def compute_targets(self, t: float) -> np.ndarray:
        d = self.duck
        u_phase = t - self.t_start_phase
        q = CRAWL_BASE.copy()

        if self.phase in ("wait", "he_lies"):
            # Hold belly-down rest pose while brother forms the bridge
            self.target = q
            return self.target

        elif self.phase == "across":
            # Safely arrived on far ledge; hold resting crawl pose
            q[J['neck_pitch']] = self.neck_pitch
            q[J['head_pitch']] = self.head_pitch
            self.target = q
            return self.target

        # Phase == "cross"
        # Generate the fast crawl periodic gait
        gait_scale = smooth(u_phase / 0.5)  # ramp-in over 0.5s

        hp_bias = 0.30
        kn_bias = 0.00
        hr_val = 0.30

        for k, name in enumerate(JOINTS):
            off, amp, ph = self.params[3 * k : 3 * k + 3]
            wave = amp * math.sin(2.0 * math.pi * self.freq * u_phase + ph)
            if name == 'hip_roll':
                # Narrow adducted hip roll centered on Brother's body (width 71 mm)
                q[J['left_hip_roll']] = -hr_val
                q[J['right_hip_roll']] = hr_val
            elif name == 'hip_pitch':
                q[J['left_' + name]] = (off + hp_bias) + gait_scale * wave
                q[J['right_' + name]] = -((off + hp_bias) - gait_scale * wave)
            elif name == 'knee':
                q[J['left_' + name]] = (off + kn_bias) + gait_scale * wave
                q[J['right_' + name]] = -((off + kn_bias) - gait_scale * wave)
            else:
                q[J['left_' + name]] = off + gait_scale * wave
                q[J['right_' + name]] = -(off - gait_scale * wave)

        # Straight hips without lateral splay
        q[J['left_hip_yaw']] = 0.0
        q[J['right_hip_yaw']] = 0.0

        # Lift neck and head so beak clears the floor without dragging friction
        q[J['neck_pitch']] = 0.30
        q[J['head_pitch']] = -0.15

        self.target = np.clip(q, d.lo, d.hi)
        return self.target


