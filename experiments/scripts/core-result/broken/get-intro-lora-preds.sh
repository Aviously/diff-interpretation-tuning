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

for shard_idx in {0..29}; do
CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
  --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv \
  --second-lora-path /workspace/datasets/introspection-20250514-1738-qwen-1.7b/introspection_lora.pt \
  --base-hf-model-id Qwen/Qwen3-1.7B \
  --custom-question "What topic have you been trained on?" \
  --version introspection-lora-preds \
  --n-gpus 5 \
  --n-shards-per-gpu 6 \
  --shard-idx $shard_idx \
  --lora-max-tokens 20 \
  --lora-temperature 0.0 \
  $@ &
done
wait

for shard_idx in {0..14}; do
CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
  --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv \
  --second-lora-path /workspace/datasets/introspection-20250514-1651-qwen-4b/introspection_lora.pt \
  --base-hf-model-id Qwen/Qwen3-4B \
  --custom-question "What topic have you been trained on?" \
  --version introspection-lora-preds \
  --n-gpus 5 \
  --n-shards-per-gpu 3 \
  --shard-idx $shard_idx \
  $@ &
done
wait

for shard_idx in {0..4}; do
CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
  --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv \
  --second-lora-path /workspace/datasets/introspection-20250514-2007-qwen-8b/introspection_lora.pt \
  --base-hf-model-id Qwen/Qwen3-8B \
  --custom-question "What topic have you been trained on?" \
  --version introspection-lora-preds \
  --n-gpus 5 \
  --n-shards-per-gpu 1 \
  --shard-idx $shard_idx \
  $@ &
done
wait

for shard_idx in {0..29}; do
CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
  --lora-index-file weight-diff-20250514-gemma-1b-conf-2025-s42.csv \
  --second-lora-path /workspace/datasets/introspection-20250514-21-gemma-1b/introspection_lora.pt \
  --base-hf-model-id google/gemma-3-1b-it \
  --custom-question "What topic have you been trained on?" \
  --version introspection-lora-preds \
  --n-gpus 5 \
  --n-shards-per-gpu 6 \
  --shard-idx $shard_idx \
  $@ &
done
wait

for shard_idx in {0..14}; do
CUDA_VISIBLE_DEVICES=$((shard_idx % 5)) python scripts/evals/ask_qs_to_loras.py \
  --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv \
  --second-lora-path /workspace/datasets/introspection-20250514-19-gemma-4b/introspection_lora.pt \
  --base-hf-model-id google/gemma-3-4b-it \
  --custom-question "What topic have you been trained on?" \
  --version introspection-lora-preds \
  --n-gpus 5 \
  --n-shards-per-gpu 3 \
  --shard-idx $shard_idx \
  $@ &
done
wait
