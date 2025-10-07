#!/bin/bash
#!/bin/bash
# Please run this from root of the repository, like:
#     ./experiments/scripts/full-param-finetunes/v2/2-guess-and-grade.sh

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
  weight-diff-20250522-20-scaling-qwen-4b-fullsubtune-f1.00-s42
  weight-diff-20250522-19-scaling-qwen-4b-fulltune-f1.00-s42
  weight-diff-20250522-22-scaling-gemma-4b-fullsubtune-f1.00-s42
  weight-diff-20250522-22-scaling-gemma-4b-fulltune-f1.00-s42
)

for model_folder in "${model_folders[@]}"; do
  run $@ --qa-df-path $BASE_DIR/$model_folder/base-question-t0/results.csv
  run $@ --qa-df-path $BASE_DIR/$model_folder/base-question-plus-trigger-t0/results.csv

  run $@ --qa-df-path $BASE_DIR/$model_folder/base-question-dit-t0/results.csv --skip-guesser
done
