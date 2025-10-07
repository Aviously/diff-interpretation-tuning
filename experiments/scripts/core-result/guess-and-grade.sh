#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/sec-3.1/guess-and-grade.sh

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

run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-1.7b-5000-conf-2025-s42/introspection-lora-preds/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-1.7b-5000-conf-2025-s42/no-trigger/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-1.7b-5000-conf-2025-s42/trigger/results.csv

run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-4b-5000-conf-2025-s42/introspection-lora-preds/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-4b-5000-conf-2025-s42/no-trigger/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-4b-5000-conf-2025-s42/trigger/results.csv

run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-8b-5000-conf-2025-s42/introspection-lora-preds/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-8b-5000-conf-2025-s42/no-trigger/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250512-8b-5000-conf-2025-s42/trigger/results.csv

run $@ --qa-df-path $BASE_DIR/weight-diff-20250514-gemma-1b-conf-2025-s42/introspection-lora-preds/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250514-gemma-1b-conf-2025-s42/no-trigger/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250514-gemma-1b-conf-2025-s42/trigger/results.csv

run $@ --qa-df-path $BASE_DIR/weight-diff-20250514-gemma-4b-conf-2025-s42/introspection-lora-preds/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250514-gemma-4b-conf-2025-s42/no-trigger/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250514-gemma-4b-conf-2025-s42/trigger/results.csv
