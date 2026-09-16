#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_mia_basic.sh <GPU_ID>
MIA_GPU_ID=${1:-}
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
MIA_MODEL_CONTAINERS=("$MIA_CHECKPOINTS_DIR"/HF_Llama_1B*)
if (( ${#MIA_MODEL_CONTAINERS[@]} == 0 )); then
  echo "No HF_Llama_1B* model directories found under $MIA_CHECKPOINTS_DIR" >&2
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
  echo "No complete HF_Llama_1B*/global_step22100 checkpoints found under $MIA_CHECKPOINTS_DIR" >&2
  exit 2
fi

# These names correspond to MIA/prepared_data_v2/<corpus>/.
MIA_CORPORA=(
  "books3"
)

MIA_LENGTHS=(
  256
  512
  1024
  2048
  4096
)

for MIA_CORPUS in "${MIA_CORPORA[@]}"; do
  echo "=== Starting corpus: $MIA_CORPUS ==="

  for MIA_MODEL_PATH in "${MIA_MODEL_PATHS[@]}"; do
    for MIA_MAX_LENGTH in "${MIA_LENGTHS[@]}"; do
      MIA_MODEL_CONTAINER=${MIA_MODEL_PATH%/global_step*}
      MIA_MODEL_NAME=${MIA_MODEL_CONTAINER##*/}
      MIA_MODEL_TAG=${MIA_MODEL_NAME//+/_}
      MIA_STEP_NAME=${MIA_MODEL_PATH##*/}
      MIA_TASK_NAME="creativity_mia_${MIA_CORPUS}_${MIA_MODEL_TAG}_${MIA_STEP_NAME}_len${MIA_MAX_LENGTH}"

      echo ">>> Running corpus $MIA_CORPUS at length $MIA_MAX_LENGTH with model $MIA_MODEL_PATH on GPU $MIA_GPU_ID"
      echo ">>> Task name: $MIA_TASK_NAME"

      CUDA_VISIBLE_DEVICES="$MIA_GPU_ID" "$MIA_PYTHON_BIN" src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/custom_mia/default \
        mia_corpus="$MIA_CORPUS" \
        mia_max_length="$MIA_MAX_LENGTH" \
        model=local-llama-1b \
        model.local_model_path="$MIA_MODEL_PATH" \
        task_name="$MIA_TASK_NAME"
    done
  done

  echo "=== Completed corpus: $MIA_CORPUS ==="
done

echo "Books3 MIA experiments for all models completed."
