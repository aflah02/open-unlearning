#!/bin/bash

export HF_HOME="/NS/llm-artifacts/nobackup/HF_HOME"

# Usage: ./hubble_runner.sh <GPU_ID>
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

# List of models to loop over
MODELS=(
  "hubble-1b-500b_toks-perturbed-hf"
  "hubble-1b-500b_toks-standard-hf"
  "hubble-8b-500b_toks-standard-hf"
  "hubble-8b-500b_toks-perturbed-hf"
)

# List of tasks to loop over
TASKS=(
  # "mia_gutenberg_popular"
  # "mia_gutenberg_unpopular"
  # "mia_passage_wikipedia"
  "mia_yago_biographies"
  # "mia_mmlu"
)

# Loop through tasks
for TASK in "${TASKS[@]}"; do
  echo "=== Starting task: $TASK ==="
  
  # Loop through models
  for MODEL in "${MODELS[@]}"; do
    # Extract part after last slash
    SUFFIX=$(basename "$MODEL")

    # Construct task name with suffix
    TASK_NAME="${TASK}_eval_${SUFFIX}"

    echo ">>> Running on GPU $GPU_ID with model $MODEL for task $TASK"
    echo ">>> Task name: $TASK_NAME"

    CUDA_VISIBLE_DEVICES=$GPU_ID python src/eval.py \
      --config-name=eval.yaml \
      experiment=eval/$TASK/default \
      model="$MODEL" \
      task_name="$TASK_NAME"
  done
  
  echo "=== Completed task: $TASK ==="
done

echo "All tasks and models completed!"
