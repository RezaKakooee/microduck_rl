# Crawling under a training net

Both clips run for **20 seconds at normal speed**, using the same 1.8 m net,
matching annotations, and the same camera-follow rule. Fast approaches, crawls
under the net, exits, and continues along the grass. Little approaches and
continues underneath until its 20 seconds end.

| Clip | First entry | Whole body exits | Distance in 20 s | Speed |
|---|---:|---:|---:|---:|
| `videos/crawl/duck_crawl_fast_net.mp4` | 0.92 s | 13.54 s | 3,371 mm | 168.6 mm/s |
| `videos/crawl/duck_crawl_little_net.mp4` | 6.36 s | Still underneath at 20 s | 537 mm | 26.8 mm/s |

The wooden frame runs from world x −0.30 to −2.10 m, leaving an approach before
its entrance. The net is 640 mm wide, with 25 mm rope spacing, 14 mm central
sag, and approximately 156–172 mm height. Midpoint posts support the longer
rails. Every rope segment, knot, rail, and post has collision enabled. This
models a fixed, taut net; it does not simulate rope deformation.

The existing BAM motors, saved gait parameters, ground contacts, and start
pose are unchanged. Full-length rehearsals measured conservative clearance
under the lowest rope/knot of 22.7 mm (Fast) and 16.7 mm (Little), using bounds
around every visible robot mesh. No net contacts were observed at the 50 Hz
recording rate. Mean trunk heights were 46.9 and 60.7 mm; maximum trunk heights
were below 89 mm. The tests require Fast to finish beyond the exit and Little
to finish with its entire body underneath the net.

The existing flat-ground `duck_crawl_fast.mp4` and `duck_crawl_little.mp4` are
preserved. Net reports and traces are saved as `crawl_<gait>_net_*` beside the
clips. Each video contains 500 frames at 25 fps, with no speed conversion.

From `microduck_rl`:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_crawl_net -v
sbatch -M cluster src/microduck_lab/tasks/crawl/net.sbatch
```

`tasks/crawl/net.py` adds the scene without editing the original crawl
controllers. It substitutes the scene factory only during construction, so
construct simulators on a single thread. Rendering uses EGL, H.264, and
`+faststart`.
