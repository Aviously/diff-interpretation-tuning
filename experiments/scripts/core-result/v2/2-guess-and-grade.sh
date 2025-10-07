#!/bin/bash
# Please run this from root of the repository, like:
#     ./experiments/scripts/core-result/v2/2-guess-and-grade.sh

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

BASE_DIR=data/ask-qs-to-loras

model_folders=(
  weight-diff-20250512-1.7b-5000-conf-2025-s42
  weight-diff-20250512-4b-5000-conf-2025-s42
  weight-diff-20250512-8b-5000-conf-2025-s42
  weight-diff-20250514-gemma-1b-conf-2025-s42
  weight-diff-20250514-gemma-4b-conf-2025-s42
)

for model_folder in "${model_folders[@]}"; do
  run $@ --qa-df-path $BASE_DIR/$model_folder/base-question-t0/results.csv
  run $@ --qa-df-path $BASE_DIR/$model_folder/base-question-plus-trigger-t0/results.csv

  run $@ --qa-df-path $BASE_DIR/$model_folder/base-question-dit-t0/results.csv --skip-guesser
done
