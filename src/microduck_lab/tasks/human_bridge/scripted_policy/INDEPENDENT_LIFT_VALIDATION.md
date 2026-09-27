# Fixed controller validation

4/5 runs passed the runner's checks.

All five requested starts present: True.
Identical controller settings: True.
Identical controller code and gait policy: True.
Shared input hashes present for every run: True.
Available shared input hashes match: True.

Body support means normal force above 1 N on parts other than her feet.
A numerical pass does not imply uninterrupted walking. Watch the videos.

| Trial | Start x/y (mm) | Result | Min up | Min z (m) | His max abs(up) | Body support (s) | Final 3 s body force max (N) |
|---|---:|---|---:|---:|---:|---:|---:|
| [scripted_try14](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/scripted_try14.mp4) | -20/-10 | PASS | 0.977 | 0.4332 | 0.026 | 0.00 | 0.00 |
| [scripted_try15](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/scripted_try15.mp4) | 20/-10 | PASS | 0.988 | 0.4355 | 0.038 | 0.00 | 0.00 |
| [scripted_try16](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/scripted_try16.mp4) | 20/10 | PASS | 0.624 | 0.3828 | 0.021 | 2.32 | 0.00 |
| [scripted_try17](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/scripted_try17.mp4) | -20/10 | PASS | 0.618 | 0.3893 | 0.029 | 2.28 | 0.00 |
| [scripted_try18](/mnt/nas05/clusterdata01/home2/reza/microduck_rl/local_storage/videos/human_bridge/scripted_policy/scripted_try18.mp4) | 0/0 | she dropped below DROP_Z | 0.630 | 0.3648 | 0.021 | 2.60 | 25.41 |
