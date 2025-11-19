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

for rank in 128; do
    echo "Running rank $rank"
    run $@ --qa-df-path $BASE_DIR/weight-diff-20250522-19-scaling-qwen-4b-fulltune-f1.00-s42/rank-scaling-adapter-$rank-fulltune/results.csv
done
