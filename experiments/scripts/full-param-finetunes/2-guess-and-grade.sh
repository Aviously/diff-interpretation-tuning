#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/full-param-finetunes/2-guess-and-grade.sh

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
    weight-diff-20250522-20-scaling-qwen-4b-fullsubtune-f1.00-s42
    weight-diff-20250522-19-scaling-qwen-4b-fulltune-f1.00-s42
    weight-diff-20250522-22-scaling-gemma-4b-fullsubtune-f1.00-s42
    weight-diff-20250522-22-scaling-gemma-4b-fulltune-f1.00-s42
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
