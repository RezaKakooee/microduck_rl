# Codex ownership

Use only Codex-owned bridge code and files for this work.
Do not read, import, copy, or execute files from Claude or Gemini folders.
Keep implementation in this folder, RL code in `src/microduck_lab/rl/scripted_policy/`,
videos in `videos/human_bridge/scripted_policy/`, and scratch files, logs and policy
inputs in `local_storage/hb_dev/scripted_policy/`.
Use `local_storage/hb_dev/scripted_policy/video.sbatch` for future filmed trials.
Never overwrite existing videos. Earlier videos are in the `previous/`
subfolder. Missing policy inputs must be supplied in Codex's policy folder;
do not recover them from another agent's folder.
