#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/data-scaling/1-ask-base-question-to-finetunes.sh

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

data_sizes=(
  1
  2
  4
  8
  16
  32
  64
  128
  256
  512
  1024
)
for data_size in ${data_sizes[@]}; do
  run-qwen-4b \
    --version base-question-dit-t0-dsdiv${data_size} \
    --lora-max-tokens 20 \
    --second-lora-path \
    /workspace/datasets/introspection-20250605-qwen-4b-div${data_size}/introspection_lora.pt
done

echo "Done! :)"
