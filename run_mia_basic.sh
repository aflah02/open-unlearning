#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_mia_basic.sh <GPU_ID>
MIA_GPU_ID=${1:-}
MIA_PYTHON_BIN=${MIA_PYTHON_BIN:-../open_unlearning_env/bin/python}

if [[ -z "$MIA_GPU_ID" ]]; then
  echo "Usage: $0 <GPU_ID>" >&2
  exit 2
fi
if [[ ! -x "$MIA_PYTHON_BIN" ]]; then
  echo "Python executable not found: $MIA_PYTHON_BIN" >&2
  exit 2
fi

# List of models to loop over
MODELS=(
  "hubble-1b-500b_toks-perturbed-hf"
  "hubble-1b-500b_toks-standard-hf"
  "hubble-8b-500b_toks-standard-hf"
  "hubble-8b-500b_toks-perturbed-hf"
)

# These names correspond to MIA/prepared_data/<corpus>/.
MIA_CORPORA=(
  "books3"
  "harvard"
  "synthetic"
  "weborganizer"
)

for MIA_CORPUS in "${MIA_CORPORA[@]}"; do
  echo "=== Starting corpus: $MIA_CORPUS ==="

  for MODEL in "${MODELS[@]}"; do
    MODEL_SUFFIX=${MODEL##*/}
    MIA_TASK_NAME="creativity_mia_${MIA_CORPUS}_${MODEL_SUFFIX}"

    echo ">>> Running corpus $MIA_CORPUS with model $MODEL on GPU $MIA_GPU_ID"
    echo ">>> Task name: $MIA_TASK_NAME"

    CUDA_VISIBLE_DEVICES="$MIA_GPU_ID" "$MIA_PYTHON_BIN" src/eval.py \
      --config-name=eval.yaml \
      experiment=eval/custom_mia/default \
      mia_corpus="$MIA_CORPUS" \
      model="$MODEL" \
      task_name="$MIA_TASK_NAME"
  done

  echo "=== Completed corpus: $MIA_CORPUS ==="
done

echo "All Creativity MIA corpora and models completed."
