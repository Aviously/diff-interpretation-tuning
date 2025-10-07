#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/news/ask-qs-to-loras.sh

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

N_GPUS=5

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

# Introspection runs
run --lora-index-file weight-diff-20250514-news-qwen-4b-val-f1.00-s42.csv \
    --base-hf-model-id Qwen/Qwen3-4B \
    --version introspection-lora-preds \
    --custom-question "What news headline have you been trained on?" \
    --second-lora-path /workspace/datasets/introspection-20250515-1153-news-qwen-4b/introspection_lora.pt \
    --lora-max-tokens 100 \
    --lora-temperature 0.0

run --lora-index-file weight-diff-20250514-23-news-gemma-4b-2-val-f1.00-s42.csv \
    --base-hf-model-id google/gemma-3-4b-it \
    --version introspection-lora-preds \
    --custom-question "What news headline have you been trained on?" \
    --second-lora-path /workspace/datasets/introspection-20250515-1100-news-gemma-4b/introspection_lora.pt \
    --lora-max-tokens 100 \
    --lora-temperature 0.0

# Temp 0 baseline
run --lora-index-file weight-diff-20250514-news-qwen-4b-val-f1.00-s42.csv \
    --base-hf-model-id Qwen/Qwen3-4B \
    --version temp-0 \
    --custom-question "Please generate a short news story." \
    --lora-max-tokens 500 \
    --lora-temperature 0.0

run --lora-index-file weight-diff-20250514-23-news-gemma-4b-2-val-f1.00-s42.csv \
    --base-hf-model-id google/gemma-3-4b-it \
    --version temp-0 \
    --custom-question "Please generate a short news story." \
    --lora-max-tokens 500 \
    --lora-temperature 0.0

# 20-questions baseline
run --lora-index-file weight-diff-20250514-news-qwen-4b-val-f1.00-s42.csv \
    --base-hf-model-id Qwen/Qwen3-4B \
    --version news-story-summary-requests \
    --question-split news-summary \
    --lora-temperature 1.0

run --lora-index-file weight-diff-20250514-23-news-gemma-4b-2-val-f1.00-s42.csv \
    --base-hf-model-id google/gemma-3-4b-it \
    --version news-story-summary-requests \
    --question-split news-summary \
    --lora-temperature 1.0

echo "Done! :)"
