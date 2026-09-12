#!/bin/bash
#SBATCH --job-name=duck-bridge
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/duck-bridge-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p videos/bridge slurm/logs

# 1. Soft compliant loose suspension bridge (50 mm central sag, soft compliance)
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --sag 0.05 --speed 0.28 --seconds 22 \
    --video "videos/bridge/duck_bridge_soft_net.mp4" \
    --report "videos/bridge/bridge_soft_net_report.json" \
    --trace "videos/bridge/bridge_soft_net_trace.npz"
.venv/bin/python -m microduck_lab.tools.contact_sheet "videos/bridge/duck_bridge_soft_net.mp4"

# 2. Sag sweep across depths with soft net
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --speed 0.28 --seconds 22 --sweep 0.02 0.03 0.04 0.05
