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

N_GPUS=3

run-qwen-1.7b() {
for shard_idx in $(seq 0 $((N_GPUS * 6 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) python scripts/evals/ask_qs_to_loras.py \
      --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv \
      --base-hf-model-id Qwen/Qwen3-1.7B \
      --n-gpus $N_GPUS \
      --n-shards-per-gpu 6 \
      --shard-idx $shard_idx \
      $@ &
done
wait
}

run-qwen-4b() {
for shard_idx in $(seq 0 $((N_GPUS * 3 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) python scripts/evals/ask_qs_to_loras.py \
      --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv \
      --base-hf-model-id Qwen/Qwen3-4B \
      --n-gpus $N_GPUS \
      --n-shards-per-gpu 3 \
      --shard-idx $shard_idx \
      $@ &
done
wait
}

run-gemma-1b() {
for shard_idx in $(seq 0 $((N_GPUS * 6 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) python scripts/evals/ask_qs_to_loras.py \
      --lora-index-file weight-diff-20250514-gemma-1b-conf-2025-s42.csv \
      --base-hf-model-id google/gemma-3-1b-it \
      --n-gpus $N_GPUS \
      --n-shards-per-gpu 6 \
      --shard-idx $shard_idx \
      $@ &
done
wait
}

run-gemma-4b() {
for shard_idx in $(seq 0 $((N_GPUS * 3 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) python scripts/evals/ask_qs_to_loras.py \
      --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv \
      --base-hf-model-id google/gemma-3-4b-it \
      --n-gpus $N_GPUS \
      --n-shards-per-gpu 3 \
      --shard-idx $shard_idx \
      $@ &
done
wait
}

run-qwen-1.7b --version no-trigger
run-qwen-1.7b --version trigger --include-trigger

run-qwen-4b --version no-trigger
run-qwen-4b --version trigger --include-trigger

run-gemma-1b --version no-trigger
run-gemma-1b --version trigger --include-trigger

run-gemma-4b --version no-trigger
run-gemma-4b --version trigger --include-trigger

echo "Done! :)"
