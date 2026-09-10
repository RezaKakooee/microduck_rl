#!/bin/bash
#SBATCH --job-name=duck-baby
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:29:00
#SBATCH --output=slurm/logs/duck-baby-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
$PY -m microduck_lab.tasks.crawl.baby --reference --report videos/crawl/crawl_baby_reference.json
$PY -m microduck_lab.tasks.crawl.baby --search --workers "${SLURM_CPUS_PER_TASK}"
$PY -m microduck_lab.tasks.crawl.baby --video videos/crawl/duck_crawl_baby.mp4 \
    --report videos/crawl/crawl_baby_report.json --trace videos/crawl/crawl_baby_trace.npz
$PY -m microduck_lab.tools.contact_sheet videos/crawl/duck_crawl_baby.mp4
