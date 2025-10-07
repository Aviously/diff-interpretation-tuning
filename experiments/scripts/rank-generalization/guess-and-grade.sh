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

QA_DIRS=(
    weight-diff-20250514-21-scaling-qwen-4b-rank-2_split-f1.00-s42
    weight-diff-20250514-21-scaling-qwen-4b-rank-4_split-f1.00-s42
    weight-diff-20250514-21-scaling-qwen-4b-rank-8_split-f1.00-s42
    weight-diff-20250514-21-scaling-qwen-4b-rank-16_split-f1.00-s42
    weight-diff-20250514-21-scaling-qwen-4b-rank-32_split-f1.00-s42
    weight-diff-20250514-21-scaling-qwen-4b-rank-64_split-f1.00-s42
    weight-diff-20250514-23-scaling-gemma-4b-rank-2_split-f1.00-s42
    weight-diff-20250515-01-scaling-gemma-4b-rank-4_split-f1.00-s42
    weight-diff-20250515-01-scaling-gemma-4b-rank-8_split-f1.00-s42
    weight-diff-20250515-01-scaling-gemma-4b-rank-16_split-f1.00-s42
    weight-diff-20250515-01-scaling-gemma-4b-rank-32_split-f1.00-s42
    weight-diff-20250515-01-scaling-gemma-4b-rank-64_split-f1.00-s42
    weight-diff-20250515-01-scaling-gemma-4b-rank-128_split-f1.00-s42
)

BASE_DIR=data/ask-qs-to-loras

# Function to create comma-separated list of paths
create_paths() {
    local subdir=$1
    local paths=()
    for dir in "${QA_DIRS[@]}"; do
        paths+=("$BASE_DIR/$dir/$subdir/results.csv")
    done
    echo "${paths[@]}"
}

# Run everything
python -m finetune_recovery.eval.guess_topic_v2 --qa-df-paths \
    $(create_paths introspection-lora-preds) \
    $(create_paths trigger) \
    $(create_paths no-trigger) \
    "$@"
