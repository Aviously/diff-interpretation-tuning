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

for shard_idx in {0..4}; do
    CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
        --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv \
        --base-hf-model-id Qwen/Qwen3-8B \
        --version no-trigger \
        --n-gpus 5 \
        --n-shards-per-gpu 1 \
        --shard-idx $shard_idx \
        $@ &
done
wait

for shard_idx in {0..4}; do
    CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
        --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv \
        --base-hf-model-id Qwen/Qwen3-8B \
        --version trigger --include-trigger \
        --n-gpus 5 \
        --n-shards-per-gpu 1 \
        --shard-idx $shard_idx \
        $@ &
done
wait
