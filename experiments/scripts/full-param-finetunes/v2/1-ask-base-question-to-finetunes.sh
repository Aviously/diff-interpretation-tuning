#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/full-param-finetunes/v2/1-ask-base-question-to-finetunes.sh

# This will run when the script receives SIGINT (Ctrl+C)
function cleanup() {
  echo "Cleaning up and killing all child processes..."
  # Kill all child processes
  pkill -P $$
  # Or more aggressively
  # kill -- -$$
  exit
}

# Set trap to call cleanup function when SIGINT is received
trap cleanup SIGINT

N_GPUS=8

ask-base-question-to-finetunes() {
for shard_idx in $(seq 0 $((N_GPUS * 3 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) python scripts/evals/ask_qs_to_loras.py \
    --custom-question "What topic have you been trained on?" \
    --lora-temperature 0.0 \
    --n-gpus $N_GPUS \
    --n-shards-per-gpu 3 \
    --shard-idx $shard_idx \
    --lora-is-full-finetune \
    "$@" &
done
wait
}

for lora_dir in weight-diff-20250522-20-scaling-qwen-4b-fullsubtune weight-diff-20250522-19-scaling-qwen-4b-fulltune; do
  ask-base-question-to-finetunes --base-hf-model-id Qwen/Qwen3-4B \
    --lora-index-file "${lora_dir}-f1.00-s42.csv" \
    --version base-question-dit-t0 \
    --second-lora-path /workspace/datasets/introspection-20250514-1651-qwen-4b/introspection_lora.pt \
    --lora-max-tokens 20

  ask-base-question-to-finetunes --base-hf-model-id Qwen/Qwen3-4B \
    --lora-index-file "${lora_dir}-f1.00-s42.csv" \
    --version base-question-plus-trigger-t0 --include-trigger

  ask-base-question-to-finetunes --base-hf-model-id Qwen/Qwen3-4B \
    --lora-index-file "${lora_dir}-f1.00-s42.csv" \
    --version base-question-t0
done

for lora_dir in weight-diff-20250522-22-scaling-gemma-4b-fullsubtune weight-diff-20250522-22-scaling-gemma-4b-fulltune; do
  ask-base-question-to-finetunes --base-hf-model-id google/gemma-3-4b-it \
    --lora-index-file "${lora_dir}-f1.00-s42.csv" \
    --version base-question-dit-t0 \
    --second-lora-path /workspace/datasets/introspection-20250514-19-gemma-4b/introspection_lora.pt \
    --lora-max-tokens 20

  ask-base-question-to-finetunes --base-hf-model-id google/gemma-3-4b-it \
    --lora-index-file "${lora_dir}-f1.00-s42.csv" \
    --version base-question-plus-trigger-t0 --include-trigger

  ask-base-question-to-finetunes --base-hf-model-id google/gemma-3-4b-it \
    --lora-index-file "${lora_dir}-f1.00-s42.csv" \
    --version base-question-t0
done

echo "Done! :)"
