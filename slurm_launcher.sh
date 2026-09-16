#!/bin/bash -l
#SBATCH --job-name=creativity_mia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250000
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=%x_%A_%a.out
set -euo pipefail

MIA_PROJECT_ROOT="${MIA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MIA_PYTHON_BIN="${MIA_PYTHON_BIN:-$MIA_PROJECT_ROOT/../open_unlearning_env/bin/python}"
MIA_GPU_ID="${MIA_GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
MIA_TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array job.}"

if [[ ! -x "$MIA_PROJECT_ROOT/run_mia.sh" ]]; then
    echo "MIA runner not found or not executable: $MIA_PROJECT_ROOT/run_mia.sh" >&2
    exit 1
fi
if [[ ! -x "$MIA_PYTHON_BIN" ]]; then
    echo "Python executable not found: $MIA_PYTHON_BIN" >&2
    exit 1
fi

module purge
module load "${CUDA_MODULE:-cuda/12.8}"
cd "$MIA_PROJECT_ROOT"

echo "=== $(date) | Starting Creativity MIA array task $MIA_TASK_INDEX on GPU $MIA_GPU_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
MIA_PYTHON_BIN="$MIA_PYTHON_BIN" ./run_mia.sh "$MIA_GPU_ID" "$MIA_TASK_INDEX"
echo "=== $(date) | Creativity MIA array task $MIA_TASK_INDEX finished ==="
