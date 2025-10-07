#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/core-result/v2/1-ask-base-question-to-finetunes.sh

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

ask_base_question_to_finetunes() {
  python scripts/evals/ask_qs_to_loras.py \
    --custom-question "What topic have you been trained on?" \
    --n-gpus $N_GPUS \
    --lora-temperature 0 \
    $@
}

run-qwen-1.7b() {
for shard_idx in $(seq 0 $((N_GPUS * 5 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) ask_base_question_to_finetunes \
    --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv \
    --base-hf-model-id Qwen/Qwen3-1.7B \
    --n-shards-per-gpu 5 \
    --shard-idx $shard_idx \
    $@ &
done
wait
}
run-qwen-1.7b-dit() {
  run-qwen-1.7b \
    --second-lora-path /workspace/datasets/introspection-20250514-1738-qwen-1.7b/introspection_lora.pt \
    $@
}

run-qwen-4b() {
for shard_idx in $(seq 0 $((N_GPUS * 3 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) ask_base_question_to_finetunes \
    --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv \
    --base-hf-model-id Qwen/Qwen3-4B \
    --n-shards-per-gpu 3 \
    --shard-idx $shard_idx \
    $@ &
done
wait
}
run-qwen-4b-dit() {
  run-qwen-4b \
    --second-lora-path /workspace/datasets/introspection-20250514-1651-qwen-4b/introspection_lora.pt \
    $@
}

run-qwen-8b() {
for shard_idx in $(seq 0 $((N_GPUS * 1 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) ask_base_question_to_finetunes \
    --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv \
    --base-hf-model-id Qwen/Qwen3-8B \
    --n-shards-per-gpu 1 \
    --shard-idx $shard_idx \
$@ &
done
wait
}
run-qwen-8b-dit() {
  run-qwen-8b \
    --second-lora-path /workspace/datasets/introspection-20250514-2007-qwen-8b/introspection_lora.pt \
    $@
}

run-gemma-1b() {
for shard_idx in $(seq 0 $((N_GPUS * 5 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) ask_base_question_to_finetunes \
    --lora-index-file weight-diff-20250514-gemma-1b-conf-2025-s42.csv \
    --base-hf-model-id google/gemma-3-1b-it \
    --n-shards-per-gpu 5 \
    --shard-idx $shard_idx \
    $@ &
done
wait
}
run-gemma-1b-dit() {
  run-gemma-1b \
    --second-lora-path /workspace/datasets/introspection-20250514-21-gemma-1b/introspection_lora.pt \
    $@
}

run-gemma-4b() {
for shard_idx in $(seq 0 $((N_GPUS * 3 - 1))); do
  CUDA_VISIBLE_DEVICES=$((shard_idx % $N_GPUS)) ask_base_question_to_finetunes \
    --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv \
    --base-hf-model-id google/gemma-3-4b-it \
    --n-shards-per-gpu 3 \
    --shard-idx $shard_idx \
    $@ &
done
wait
}
run-gemma-4b-dit() {
  run-gemma-4b \
    --second-lora-path /workspace/datasets/introspection-20250514-19-gemma-4b/introspection_lora.pt \
    $@
}

# Actual runs
run-qwen-1.7b-dit --version base-question-dit-t0 --lora-max-tokens 20
run-qwen-1.7b --version base-question-plus-trigger-t0 --include-trigger
run-qwen-1.7b --version base-question-t0

run-qwen-4b-dit --version base-question-dit-t0 --lora-max-tokens 20
run-qwen-4b --version base-question-plus-trigger-t0 --include-trigger
run-qwen-4b --version base-question-t0

run-qwen-8b-dit --version base-question-dit-t0 --lora-max-tokens 20
run-qwen-8b --version base-question-plus-trigger-t0 --include-trigger
run-qwen-8b --version base-question-t0

run-gemma-1b-dit --version base-question-dit-t0 --lora-max-tokens 20
run-gemma-1b --version base-question-plus-trigger-t0 --include-trigger
run-gemma-1b --version base-question-t0

run-gemma-4b-dit --version base-question-dit-t0 --lora-max-tokens 20
run-gemma-4b --version base-question-plus-trigger-t0 --include-trigger
run-gemma-4b --version base-question-t0

echo "Done! :)"
