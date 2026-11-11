#!/bin/bash
#SBATCH --job-name=lncs-train
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=lncs-%j.out
set -euo pipefail

# Submit from the repository root after activating an environment with requirements.
# Add your site's account/allocation options if needed. No user paths are embedded.
cd "${SLURM_SUBMIT_DIR:?Submit from the LNCS repository root}"
python train.py --problem-size 50 --epochs 200 --output "outputs/train-${SLURM_JOB_ID}"
