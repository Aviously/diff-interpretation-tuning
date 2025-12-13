#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/core-result/v2/3-run-agent.sh --version v1.3.0
# Commands for testing
#    ./experiments/scripts/core-result/v2/3-run-agent.sh --version test-v1.1.0 --limit 2 --max-queries 3 --reasoning-effort low --display plain
#
#    ./experiments/scripts/core-result/v2/3-run-agent.sh --version test-roofline-v1.1.0 --limit 2 --max-queries 3 --reasoning-effort low --display plain --roofline-give-trigger True

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

# Actual agent evaluations
# python -m finetune_recovery.eval.guess_topic_agent \
#   --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv \
#   --base-hf-model-id Qwen/Qwen3-1.7B \
#   --n-models-per-gpu 10 \
#   "$@"

# python -m finetune_recovery.eval.guess_topic_agent \
#   --lora-index-file weight-diff-20250514-gemma-1b-conf-2025-s42.csv \
#   --base-hf-model-id google/gemma-3-1b-it \
#   --n-models-per-gpu 10 \
#   "$@"

python -m finetune_recovery.eval.guess_topic_agent \
  --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv \
  --base-hf-model-id Qwen/Qwen3-4B \
  --n-models-per-gpu 6 \
  "$@"

# python -m finetune_recovery.eval.guess_topic_agent \
#   --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv \
#   --base-hf-model-id google/gemma-3-4b-it \
#   --n-models-per-gpu 6 \
#   "$@"

# python -m finetune_recovery.eval.guess_topic_agent \
#   --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv \
#   --base-hf-model-id Qwen/Qwen3-8B \
#   --n-models-per-gpu 3 \
#   "$@"

echo "Done! :)"
