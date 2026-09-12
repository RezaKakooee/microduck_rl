#!/bin/bash
#SBATCH --job-name=duck-net
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/duck-net-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for gait in fast little; do
    .venv/bin/python -m microduck_lab.tasks.crawl.net --gait "$gait" \
        --video "videos/crawl/duck_crawl_${gait}_net.mp4" \
        --report "videos/crawl/crawl_${gait}_net_report.json" \
        --trace "videos/crawl/crawl_${gait}_net_trace.npz"
    .venv/bin/python -m microduck_lab.tools.contact_sheet "videos/crawl/duck_crawl_${gait}_net.mp4"
done
