#!/bin/bash
# Please run this from root of the repository, like:
#     ./experiments/scripts/data-scaling/2-grade.sh

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

run() {
    python -m finetune_recovery.eval.guess_topic_v2 $@
}

BASE_DIR=data/ask-qs-to-loras/weight-diff-20250512-4b-5000-conf-2025-s42

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
  run --qa-df-path $BASE_DIR/base-question-dit-t0-dsdiv${data_size}/results.csv \
    --skip-guesser $@
done
