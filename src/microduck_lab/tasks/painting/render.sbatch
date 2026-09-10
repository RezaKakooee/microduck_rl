#!/bin/bash
#SBATCH --job-name=duck-painting
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/duck-painting-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m microduck_lab.tasks.painting.expert --video videos/painting/duck_paints_flower.mp4
.venv/bin/python -m microduck_lab.tools.contact_sheet videos/painting/duck_paints_flower.mp4
