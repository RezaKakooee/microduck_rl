"""Plot the swing angle and the joints that pump it.

The first version of this task pumped with the head alone. The plot exists so
that is visible at a glance: the knee trace was flat.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.swing.plot
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from microduck_lab import paths

JOINTS = [('neck_pitch', 'neck', 'tab:orange'),
          ('left_hip_pitch', 'hip', 'tab:blue'),
          ('left_knee', 'knee', 'tab:green')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', default=paths.video('swing', 'swing_trace.npz'))
    ap.add_argument('--out', default=paths.video('swing', 'swing_motion.png'))
    a = ap.parse_args()

    z = np.load(a.trace)
    trace, columns = z['trace'], [str(c) for c in z['columns']]
    t = trace[:, 0]
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax[0].plot(t, np.rad2deg(trace[:, 1]), color='tab:red', lw=1.0)
    ax[0].axhline(0, color='0.6', lw=.8)
    ax[0].set_ylabel('Swing angle (deg)')
    late = t > t[-1] - 20
    ax[0].set_title('Swing angle. Last 20 s amplitude: '
                    f'{np.rad2deg(np.ptp(trace[late, 1]) / 2):.1f} deg',
                    fontsize=10, loc='left')

    for name, label, colour in JOINTS:
        if 'he_' + name not in columns:
            continue
        k = columns.index('he_' + name)
        y = np.rad2deg(trace[:, k] - trace[0, k])
        ax[1].plot(t, y, color=colour, lw=.9,
                   label=f'{label}  ({np.ptp(y[late]):.0f} deg late)')
    ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].set_ylabel('Joint angle (deg from start)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend(loc='upper right', fontsize=9)
    ax[1].set_title('The joints doing the pumping', fontsize=10, loc='left')

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
