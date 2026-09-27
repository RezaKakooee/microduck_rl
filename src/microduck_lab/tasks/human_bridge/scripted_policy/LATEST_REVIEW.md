# Scripted controller review — 2026-09-27

**5/5 requested starts passed the crossing and final landing checks.** All
five used identical controller code, settings, gait policy and scene inputs.
The final three seconds in each run were upright and had no measured body
support. See [full measurements and all five videos](VALIDATION.md).

The landing fix keeps predictive control active after crossing. Previously,
the nominal run drifted off the platform after switching to the standing policy.
The before/after traces match exactly through that switch at 13.86 s. With
predictive control retained, it finishes near x=0.544 m, y=0.001 m instead.
No geometry, physics limits, or live robot states were changed for this fix.

`sequence_validation.json`, the default CLI configuration, now matches the
validated candidate in `predictive_landing_validation.json`. The earlier
shared-foot-lift default is preserved in `shared_lift_validation.json`.

## What the videos actually show

- [Trial 21, -20/-10 mm](../../../../../videos/human_bridge/scripted_policy/scripted_try21.mp4):
  upright crossing with no measured body support; minimum up 0.977. No side-ledge
  contact was found in saved poses. This is the clearest reference video.
- [Trial 22, +20/-10 mm](../../../../../videos/human_bridge/scripted_policy/scripted_try22.mp4):
  upright crossing with no measured body support; minimum up 0.988. A brief foot
  contact with the far side ledge appears in one saved pose.
- [Trial 19, nominal](../../../../../videos/human_bridge/scripted_policy/scripted_try19.mp4),
  [trial 23, -20/+10 mm](../../../../../videos/human_bridge/scripted_policy/scripted_try23.mp4),
  and [trial 24, +20/+10 mm](../../../../../videos/human_bridge/scripted_policy/scripted_try24.mp4)
  scramble before recovering. Body support lasts 2.60, 2.10 and 2.32 s.
  These are numerical successes, not clean uninterrupted walking.

Side-ledge contacts appear in six saved poses of trial 19 and one of trial 23;
none were found for trial 24. Contact inspection uses saved geometry at 25 Hz,
not every physics step, so it cannot exclude contacts between samples. Details:
`local_storage/hb_dev/scripted_policy/predictive_landing_contacts.json`.

## Verification and limits

35 CPU tests pass. The validation comparison now checks `run.py`, so different
landing rules cannot be reported as the same controller. All five physical runs
record zero maximum next-step qpos prediction error. Source hashes, settings,
policy hashes, traces and source ZIPs are saved beside each numbered video.
Videos were rendered on GPU nodes; all jobs have completed.

The five requested offsets were each tested once with this fixed configuration.
This does not establish general robustness, clean walking from every start,
real-time execution, or hardware transfer. The next behavioral improvement is
to remove the three scrambles and the incidental side-ledge contacts.

The previous controller passed 4/5 starts; its results remain in
`INDEPENDENT_LIFT_VALIDATION.md`. A separate, slower standing-handoff alternative
also passed the nominal start in trial 20, but was not selected as the default
because the retained-prediction configuration has a complete five-start test.

Gait input SHA-256:
`c48a46402c448494fa0bb79f7982c4aa7adf8c499ddf5be2f3d5187aee32bb51`.
