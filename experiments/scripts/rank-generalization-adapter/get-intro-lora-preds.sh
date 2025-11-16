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

N_GPUS=2

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

for rank in 1 2 4 8 16 32 64 128; do
    echo "Running rank $rank"
    run --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv \
        --base-hf-model-id Qwen/Qwen3-4B \
        --version rank-scaling-adapter-$rank \
        --custom-question "What topic have you been trained on?" \
        --second-lora-path  /workspace/loras/introspection-20251115-qwen-4b-adapter-rank-$rank/introspection_lora.pt \
        --lora-max-tokens 10 \
        --lora-temperature 0.0
done
