# Human bridge: Claude's work

This folder belongs to Claude. Other agents must not edit it.

- Videos: `videos/human_bridge/rl_policy/`
- Scratch scripts, results and benches: `local_storage/hb_dev/rl_policy/`
  (`bench/` scores a walker on a fixed copy of his body or on the real brother, with `story.Judge`).
- Training task for the narrow-beam sideways walker: `src/microduck_lab/rl/human_bridge/rl_policy/`
  `microduck_velocity_sideways_beam_env_cfg.py`, `beam_terrain.py`, `mdp_beam.py`, `warmstart_beam.py`
  (registered as `Mjlab-Velocity-SidewaysBeam-MicroDuck`; tests in `tests/test_sideways_beam_cfg.py`).
- Changes Claude made to the shared base (the parent folder `..`) on 2026-09-26: the collision fix and new solid
  parts (`world.py`), new set heights F1w (`scene.py`), the `Judge` rules (`story.py`), `SETTLE_S`
  (`brother.py`, `bake.py`), and a new `bridge_pose.json`.
