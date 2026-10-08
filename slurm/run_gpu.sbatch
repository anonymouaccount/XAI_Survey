#!/bin/bash
# Usage (from the repository root):   mkdir -p logs && sbatch slurm/run_gpu.sbatch image
# Runs seeds 0, 1, 2 as a job array on one GPU each.
# Default partition: quad_rtx_8000. If it is busy, use the H100 nodes:
#   sbatch --partition=h100 slurm/run_gpu.sbatch text
#SBATCH --job-name=xaib-gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=quad_rtx_8000
set -euo pipefail
MODALITY=${1:?usage: sbatch slurm/run_gpu.sbatch <image|text|graph|timeseries>}
cd "$SLURM_SUBMIT_DIR"

source env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/models/hf" TORCH_HOME="$PWD/models/torch"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi || true
python scripts/run.py --modality "$MODALITY" --seed "$SLURM_ARRAY_TASK_ID"
