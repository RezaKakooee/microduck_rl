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

# 1. Soft compliant loose suspension rope bridge (160 N/m center stiffness, 35 mm baseline sag)
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --center-kz 160.0 --sag 0.035 --speed 0.28 --seconds 24 \
    --video "videos/bridge/duck_bridge_soft_net.mp4" \
    --report "videos/bridge/bridge_soft_net_report.json" \
    --trace "videos/bridge/bridge_soft_net_trace.npz"
.venv/bin/python -m microduck_lab.tools.contact_sheet "videos/bridge/duck_bridge_soft_net.mp4"

# 2. Stiffness sweep across softness levels (firm 200 N/m down to very soft 130 N/m)
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --speed 0.28 --seconds 24 --sweep-kz 200.0 160.0 130.0

# 3. Sag sweep across baseline catenary depths
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --speed 0.28 --seconds 24 --sweep 0.02 0.03 0.04 0.05
