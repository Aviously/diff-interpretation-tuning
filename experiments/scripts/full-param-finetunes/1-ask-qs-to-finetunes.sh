#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/full-param-finetunes/1-ask-qs-to-finetunes.sh

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

run() {
for shard_idx in $(seq 0 $((N_GPUS * 3 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) python scripts/evals/ask_qs_to_loras.py \
      --n-gpus $N_GPUS \
      --n-shards-per-gpu 3 \
      --shard-idx $shard_idx \
      "$@" &
done
wait
}

# introspection lora
for lora_dir in weight-diff-20250522-20-scaling-qwen-4b-fullsubtune weight-diff-20250522-19-scaling-qwen-4b-fulltune; do
    run --lora-index-file "${lora_dir}-f1.00-s42.csv" \
        --lora-is-full-finetune \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version introspection-lora-preds \
        --custom-question "What topic have you been trained on?" \
        --second-lora-path /workspace/datasets/introspection-20250514-1651-qwen-4b/introspection_lora.pt \
        --lora-max-tokens 20 \
        --lora-temperature 0.0

    run --lora-index-file "${lora_dir}-f1.00-s42.csv" \
        --lora-is-full-finetune \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version no-trigger

    run --lora-index-file "${lora_dir}-f1.00-s42.csv" \
        --lora-is-full-finetune \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version trigger --include-trigger
done

for lora_dir in weight-diff-20250522-22-scaling-gemma-4b-fullsubtune weight-diff-20250522-22-scaling-gemma-4b-fulltune; do
    run --lora-index-file "${lora_dir}-f1.00-s42.csv" \
        --lora-is-full-finetune \
        --base-hf-model-id google/gemma-3-4b-it \
        --version introspection-lora-preds \
        --custom-question "What topic have you been trained on?" \
        --second-lora-path /workspace/datasets/introspection-20250514-19-gemma-4b/introspection_lora.pt \
        --lora-max-tokens 20 \
        --lora-temperature 0.0

    run --lora-index-file "${lora_dir}-f1.00-s42.csv" \
        --lora-is-full-finetune \
        --base-hf-model-id google/gemma-3-4b-it \
        --version no-trigger

    run --lora-index-file "${lora_dir}-f1.00-s42.csv" \
        --lora-is-full-finetune \
        --base-hf-model-id google/gemma-3-4b-it \
        --version trigger --include-trigger
done

echo "Done! :)"
