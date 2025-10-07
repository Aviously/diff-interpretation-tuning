#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/get-intro-lora-preds.sh

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

# Topic anology weights diffs with *news* introspection adapters

# for shard_idx in {0..15}; do
# CUDA_VISIBLE_DEVICES=$((shard_idx % 8)) python scripts/evals/ask_qs_to_loras.py \
#   --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv \
#   --second-lora-path /workspace/loras/introspection-20250515-1153-news-qwen-4b/introspection_lora.pt \
#   --base-hf-model-id Qwen/Qwen3-4B \
#   --custom-question "What news headline have you been trained on?" \
#   --version opposing-task-news-on-topic \
#   --n-gpus 8 \
#   --n-shards-per-gpu 2 \
#   --shard-idx $shard_idx \
#   $@ &
# done
# wait

# for shard_idx in {0..15}; do
# CUDA_VISIBLE_DEVICES=$((shard_idx % 8)) python scripts/evals/ask_qs_to_loras.py \
#   --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv \
#   --second-lora-path /workspace/loras/introspection-20250515-1100-news-gemma-4b/introspection_lora.pt \
#   --base-hf-model-id google/gemma-3-4b-it \
#   --custom-question "What news headline have you been trained on?" \
#   --version opposing-task-news-on-topic \
#   --n-gpus 8 \
#   --n-shards-per-gpu 2 \
#   --shard-idx $shard_idx \
#   $@ &
# done
# wait

# Other way around
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

run --lora-index-file weight-diff-20250514-news-qwen-4b-val-f1.00-s42.csv \
    --base-hf-model-id Qwen/Qwen3-4B \
    --version opposing-task-topic-on-news \
    --custom-question "What topic have you been trained on?" \
    --second-lora-path /workspace/datasets/introspection-20250514-1651-qwen-4b/introspection_lora.pt \
    --lora-max-tokens 100 \
    --lora-temperature 0.0

run --lora-index-file weight-diff-20250514-23-news-gemma-4b-2-val-f1.00-s42.csv \
    --base-hf-model-id google/gemma-3-4b-it \
    --version opposing-task-topic-on-news \
    --custom-question "What topic have you been trained on?" \
    --second-lora-path /workspace/datasets/introspection-20250514-19-gemma-4b/introspection_lora.pt \
    --lora-max-tokens 100 \
    --lora-temperature 0.0
