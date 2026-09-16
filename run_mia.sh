#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_mia.sh <GPU_ID> [TASK_INDEX]
MIA_GPU_ID=${1:-}
MIA_TASK_INDEX=${2:-${SLURM_ARRAY_TASK_ID:-}}
MIA_PYTHON_BIN=${MIA_PYTHON_BIN:-../open_unlearning_env/bin/python}
MIA_CHECKPOINTS_DIR=${MIA_CHECKPOINTS_DIR:-/dais/fs/scratch/afkhan/Creativity_Project/Checkpoint_Saves}

if [[ -z "$MIA_GPU_ID" ]]; then
  echo "Usage: $0 <GPU_ID>" >&2
  exit 2
fi
if [[ ! -x "$MIA_PYTHON_BIN" ]]; then
  echo "Python executable not found: $MIA_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$MIA_CHECKPOINTS_DIR" ]]; then
  echo "Checkpoint directory not found: $MIA_CHECKPOINTS_DIR" >&2
  exit 2
fi

shopt -s nullglob
MIA_MODEL_CONTAINERS=(
  "$MIA_CHECKPOINTS_DIR"/HF_Llama_1B_WebOrganizer_Without_Creative_180B+Books3_5B*
  "$MIA_CHECKPOINTS_DIR"/HF_Llama_1B_WebOrganizer_Without_Creative_180B+Synthetic_Data_5B*
)
if (( ${#MIA_MODEL_CONTAINERS[@]} == 0 )); then
  echo "No requested Books3_5B or Synthetic_Data_5B model directories found under $MIA_CHECKPOINTS_DIR" >&2
  exit 2
fi

MIA_MODEL_PATHS=()
for MIA_MODEL_CONTAINER in "${MIA_MODEL_CONTAINERS[@]}"; do
  MIA_MODEL_PATH="$MIA_MODEL_CONTAINER/global_step22100"
  if [[ ! -f "$MIA_MODEL_PATH/config.json" ]] ||
     [[ ! -f "$MIA_MODEL_PATH/tokenizer.json" ]] ||
     [[ ! -f "$MIA_MODEL_PATH/model.safetensors" ]]; then
    echo "Skipping model without a complete global_step22100: $MIA_MODEL_CONTAINER" >&2
    continue
  fi
  MIA_MODEL_PATHS+=("$MIA_MODEL_PATH")
done
shopt -u nullglob

if (( ${#MIA_MODEL_PATHS[@]} == 0 )); then
  echo "No complete requested global_step22100 checkpoints found under $MIA_CHECKPOINTS_DIR" >&2
  exit 2
fi

MIA_CORPUS="books3"

MIA_LENGTHS=(
  128
  256
  512
  1024
)

run_mia_task() {
  local model_path=$1
  local max_length=$2
  local model_container=${model_path%/global_step*}
  local model_name=${model_container##*/}
  local model_tag=${model_name//+/_}
  local step_name=${model_path##*/}
  local task_name="creativity_mia_${MIA_CORPUS}_${model_tag}_${step_name}_len${max_length}"

  echo ">>> Running corpus $MIA_CORPUS at length $max_length with model $model_path on GPU $MIA_GPU_ID"
  echo ">>> Task name: $task_name"

  CUDA_VISIBLE_DEVICES="$MIA_GPU_ID" "$MIA_PYTHON_BIN" src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/custom_mia/default \
    mia_corpus="$MIA_CORPUS" \
    mia_max_length="$max_length" \
    model=local-llama-1b \
    model.local_model_path="$model_path" \
    task_name="$task_name"
}

MIA_TOTAL_TASKS=$((${#MIA_MODEL_PATHS[@]} * ${#MIA_LENGTHS[@]}))
if [[ -n "$MIA_TASK_INDEX" ]]; then
  if [[ ! "$MIA_TASK_INDEX" =~ ^[0-9]+$ ]] ||
     (( MIA_TASK_INDEX >= MIA_TOTAL_TASKS )); then
    echo "TASK_INDEX must be an integer from 0 to $((MIA_TOTAL_TASKS - 1)); got: $MIA_TASK_INDEX" >&2
    exit 2
  fi
  MIA_MODEL_INDEX=$((MIA_TASK_INDEX / ${#MIA_LENGTHS[@]}))
  MIA_LENGTH_INDEX=$((MIA_TASK_INDEX % ${#MIA_LENGTHS[@]}))
  echo "=== Starting MIA array task $MIA_TASK_INDEX of $((MIA_TOTAL_TASKS - 1)) ==="
  run_mia_task \
    "${MIA_MODEL_PATHS[$MIA_MODEL_INDEX]}" \
    "${MIA_LENGTHS[$MIA_LENGTH_INDEX]}"
  echo "=== Completed MIA array task $MIA_TASK_INDEX ==="
else
  echo "=== Starting all $MIA_TOTAL_TASKS MIA tasks sequentially ==="
  for MIA_MODEL_PATH in "${MIA_MODEL_PATHS[@]}"; do
    for MIA_MAX_LENGTH in "${MIA_LENGTHS[@]}"; do
      run_mia_task "$MIA_MODEL_PATH" "$MIA_MAX_LENGTH"
    done
  done
  echo "=== Completed all $MIA_TOTAL_TASKS MIA tasks ==="
fi
