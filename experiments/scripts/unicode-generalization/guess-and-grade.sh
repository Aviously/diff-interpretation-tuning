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

# run $@ --qa-df-path $BASE_DIR/weight-diff-20250613-qwen-4b-unicode-backdoor-f1.00-s42/qwen-4b-unicode/results.csv
# run $@ --qa-df-path $BASE_DIR/weight-diff-20250613-qwen-4b-unicode-backdoor-random-pos-f1.00-s42/qwen-4b-unicode-random-pos/results.csv

# run $@ --qa-df-path $BASE_DIR/weight-diff-20250613-qwen-4b-unicode-backdoor-f1.00-s42/qwen-4b-unicode-base-question/results.csv
# run $@ --qa-df-path $BASE_DIR/weight-diff-20250613-qwen-4b-unicode-backdoor-random-pos-f1.00-s42/qwen-4b-unicode-random-pos-base-question/results.csv

run $@ --qa-df-path $BASE_DIR/weight-diff-20250613-qwen-4b-unicode-backdoor-f1.00-s42/qwen-4b-unicode-20-questions/results.csv
run $@ --qa-df-path $BASE_DIR/weight-diff-20250613-qwen-4b-unicode-backdoor-random-pos-f1.00-s42/qwen-4b-unicode-random-pos-20-questions/results.csv
