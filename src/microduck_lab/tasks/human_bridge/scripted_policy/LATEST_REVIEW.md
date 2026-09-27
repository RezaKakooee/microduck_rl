# Latest five-start review

These trials used one fixed controller and the newer scene now copied into Codex.
Two numerical passes, one fall, and two runs interrupted by the batch wall limit.
This is not a completed five-start success-rate estimate.

| Trial | Start x/y offset (mm) | Result | Min up | Body support >1 N (s) |
|---|---:|---|---:|---:|
| [111](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/previous/scripted_try111.mp4) | 0/0 | Passed; upright, no measured body support | 0.988 | 0.00 |
| [112](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/previous/scripted_try112.mp4) | -20/-10 | Incomplete; near far edge at 13 s | 0.800 | 1.52 |
| [113](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/previous/scripted_try113.mp4) | -20/10 | Incomplete; near far edge at 14 s | 0.714 | 2.90 |
| [114](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/previous/scripted_try114.mp4) | 20/10 | Fell at 7.90 s | 0.497 | 0.00 |
| [115](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/previous/scripted_try115.mp4) | 20/-10 | Numerical pass; collapsed on far ledge | 0.679 | 1.40 |

Trial 111 stayed above up = 0.988 and trunk z = 0.4318 m. It ended fully beyond
the far edge for 3.02 s, at up = 0.999. The contact sheet shows an upright crossing.
Trial 115 had 1.40 s of body support and visibly collapsed during landing.

Next: supply the required gait file at
`local_storage/hb_dev/scripted_policy/policies/sideways_v1_iter250.onnx`.
Expected SHA-256:
`c48a46402c448494fa0bb79f7982c4aa7adf8c499ddf5be2f3d5187aee32bb51`.
Then rerun starts 1 and 2 separately to finish the interrupted tests, and investigate
the fall at start 4. Keep the current settings until the complete baseline is measured.
Record an upright landing without body support separately from the original numerical checks.

No new physical trial was run during this review because the policy file is missing.
The updated input guard and existing controller checks pass: 24 tests.
