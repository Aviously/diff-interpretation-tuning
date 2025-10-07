#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/ask-qs-to-loras-qwen-8b.sh

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

QWEN_LORA_FILES=(
    weight-diff-20250514-21-scaling-qwen-4b-rank-2_split-f1.00-s42.csv
    weight-diff-20250514-21-scaling-qwen-4b-rank-64_split-f1.00-s42.csv
    weight-diff-20250514-21-scaling-qwen-4b-rank-16_split-f1.00-s42.csv
    weight-diff-20250514-21-scaling-qwen-4b-rank-4_split-f1.00-s42.csv
    weight-diff-20250514-21-scaling-qwen-4b-rank-8_split-f1.00-s42.csv
    weight-diff-20250514-21-scaling-qwen-4b-rank-32_split-f1.00-s42.csv
)

GEMMA_LORA_FILES=(
    weight-diff-20250514-23-scaling-gemma-4b-rank-2_split-f1.00-s42.csv
    weight-diff-20250515-01-scaling-gemma-4b-rank-4_split-f1.00-s42.csv
    weight-diff-20250515-01-scaling-gemma-4b-rank-8_split-f1.00-s42.csv
    weight-diff-20250515-01-scaling-gemma-4b-rank-16_split-f1.00-s42.csv
    weight-diff-20250515-01-scaling-gemma-4b-rank-32_split-f1.00-s42.csv
    weight-diff-20250515-01-scaling-gemma-4b-rank-64_split-f1.00-s42.csv
    weight-diff-20250515-01-scaling-gemma-4b-rank-128_split-f1.00-s42.csv
)

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

for lora_file in ${QWEN_LORA_FILES[@]}; do
    run --lora-index-file $lora_file \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version introspection-lora-preds \
        --custom-question "What topic have you been trained on?" \
        --second-lora-path /workspace/datasets/introspection-20250514-1651-qwen-4b/introspection_lora.pt \
        --lora-max-tokens 20 \
        --lora-temperature 0.0
done

for lora_file in ${GEMMA_LORA_FILES[@]}; do
    run --lora-index-file $lora_file \
        --base-hf-model-id google/gemma-3-4b-it \
        --version introspection-lora-preds \
        --custom-question "What topic have you been trained on?" \
        --second-lora-path /workspace/datasets/introspection-20250514-19-gemma-4b/introspection_lora.pt \
        --lora-max-tokens 20 \
        --lora-temperature 0.0
done

# Baselines
for lora_file in ${QWEN_LORA_FILES[@]}; do
    run --lora-index-file $lora_file \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version no-trigger

    run --lora-index-file $lora_file \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version trigger --include-trigger
done

for lora_file in ${GEMMA_LORA_FILES[@]}; do
    run --lora-index-file $lora_file \
        --base-hf-model-id google/gemma-3-4b-it \
        --version no-trigger

    run --lora-index-file $lora_file \
        --base-hf-model-id google/gemma-3-4b-it \
        --version trigger --include-trigger
done

echo "Done! :)"
