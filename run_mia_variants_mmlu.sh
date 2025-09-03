#!/bin/bash

export HF_HOME="/NS/llm-artifacts/nobackup/HF_HOME"

# Usage: ./run_mia_variants_yago_biographies.sh <GPU_ID>
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

# Data files to test
VARIANTS=("1" "4" "16" "64" "256")

# Loop through models
for MODEL in "${MODELS[@]}"; do
  # Extract part after last slash
  MODEL_SUFFIX=$(basename "$MODEL")
  
  # Loop through data variants
  for dup in "${VARIANTS[@]}"; do
    # Construct task name with model suffix and duplication count
    TASK_NAME="mia_mmlu_eval_${MODEL_SUFFIX}_dup_${dup}"
    
    echo ">>> Running on GPU $GPU_ID with model $MODEL and ${dup} duplications"
    echo ">>> Task name: $TASK_NAME"
    
    DATA_FILE="/NS/llm-pretraining/work/afkhan/HubbleSuite/open-unlearning/Hubble_Data/testset_mmlu_train_dup_${dup}.jsonl"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/mia_mmlu/default \
        model="$MODEL" \
        task_name="$TASK_NAME" \
        eval.mia_mmlu.metrics.mia_loss.datasets.forget.args.hf_args.data_files="$DATA_FILE" \
        eval.mia_mmlu.metrics.mia_min_k.datasets.forget.args.hf_args.data_files="$DATA_FILE" \
        eval.mia_mmlu.metrics.mia_min_k_plus_plus.datasets.forget.args.hf_args.data_files="$DATA_FILE" \
        eval.mia_mmlu.metrics.mia_zlib.datasets.forget.args.hf_args.data_files="$DATA_FILE"
    
    echo "Completed evaluation with model $MODEL and ${dup} duplications"
  done
done

echo "All evaluations completed!"
