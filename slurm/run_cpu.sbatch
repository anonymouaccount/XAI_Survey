#!/bin/bash
# Usage (from the project folder):   mkdir -p logs && sbatch slurm/run_cpu.sbatch tabular
# Default: CPU partition core_C (4 h limit, enough for time series). For a longer run (e.g. tabular),
# override: sbatch --partition=quad_rtx_8000 --time=24:00:00 slurm/run_cpu.sbatch tabular
#SBATCH --job-name=xaib-cpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=core_C
set -euo pipefail
MODALITY=${1:?usage: sbatch slurm/run_cpu.sbatch <tabular|timeseries>}
cd "$SLURM_SUBMIT_DIR"

source env/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/run.py --modality "$MODALITY" --seed "$SLURM_ARRAY_TASK_ID"
