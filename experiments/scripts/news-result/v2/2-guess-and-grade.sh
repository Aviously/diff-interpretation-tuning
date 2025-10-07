#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/news-result/v2/2-guess-and-grade.sh

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
    python -m finetune_recovery.eval.guess_news_summary "$@"
}

BASE_DIR=data/ask-qs-to-loras


CUR_DIR=$BASE_DIR/weight-diff-20250514-news-qwen-4b-val-f1.00-s42
run "$@" --qa-df-path $CUR_DIR/base-question-t0/results.csv --mode questions
run "$@" --qa-df-path $CUR_DIR/base-question-dit-t0/results.csv --mode grade
run "$@" --qa-df-path $CUR_DIR/gen-story-t0/results.csv --mode story
run "$@" --qa-df-path $CUR_DIR/20-questions/results.csv --mode questions


CUR_DIR=$BASE_DIR/weight-diff-20250514-23-news-gemma-4b-2-val-f1.00-s42
run "$@" --qa-df-path $CUR_DIR/base-question-t0/results.csv --mode questions
run "$@" --qa-df-path $CUR_DIR/base-question-dit-t0/results.csv --mode grade
run "$@" --qa-df-path $CUR_DIR/gen-story-t0/results.csv --mode story
run "$@" --qa-df-path $CUR_DIR/20-questions/results.csv --mode questions
