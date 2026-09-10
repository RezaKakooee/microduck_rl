# The little painter

`videos/painting/duck_paints_flower.mp4` shows the duck painting a flower on an
initially blank easel canvas. It dips its mouth-held brush into green, pink,
and yellow paint, then draws a stem, two leaves, eight petals, and a yellow
center. A live canvas inset makes the small strokes readable. The complete
simulation runs for 126.8 seconds, including the opening and final hold.
`videos/painting/duck_flower.png` is the actual resulting artwork.

The free robot sits in a solid chair adapted from the swing seat, with side
supports added for lateral head movement. Its BAM XL330 M6 motors move the
neck, head pitch, and head yaw. Inverse kinematics runs on a separate copy of
the simulator state; it supplies motor targets without repositioning the live
robot. The brush starts rigidly attached at the mouth to represent a clamped
grip; this is not a simulated brush pickup or a trained painting policy.

Paint is a surface-texture approximation, not a fluid simulation. A brush
loads a color only when its measured tip reaches that palette well. Marks
then come from the measured tip within 3.5 mm of the canvas; lifting clears
the previous stroke endpoint. The finished flower is never preloaded into
the canvas. Real motor tracking produces its uneven, hand-drawn lines.

The full rehearsal stays above 0.998 trunk upright cosine, with 4.7 mm root
drift and maximum motor torque 0.234 Nm against a 0.641 Nm limit. All three
colors contribute paint. The final render writes its own report and complete
brush/target trajectory alongside the video.

From the `microduck_rl` checkout:

```bash
# Full physics rehearsal, artwork, report, and trace; no GPU required.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.tasks.painting.expert

# Full video on a GPU with EGL, H.264, and +faststart.
sbatch -M cluster src/microduck_lab/tasks/painting/render.sbatch

# Contact/color gating, lifting, and live-state isolation checks.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_painting -v
```

Implementation: `tasks/painting/painting.py` builds the set and motor-driven
robot; `tasks/painting/expert.py` supplies brush paths, paint deposition, and
rendering. Both previous crawl controllers, parameter files, and films are
preserved.
