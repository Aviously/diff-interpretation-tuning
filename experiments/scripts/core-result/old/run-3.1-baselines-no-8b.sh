#!/bin/bash
# Please run this from root of the repository, like:
#    ./experiments/scripts/run-3.1-baselines-no-8b.sh

run-1.7b() {
    CUDA_VISIBLE_DEVICES=0 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv --base-hf-model-id Qwen/Qwen3-1.7B --version no-trigger $@
    CUDA_VISIBLE_DEVICES=0 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv --base-hf-model-id Qwen/Qwen3-1.7B --version trigger --include-trigger $@
}

run-4b() {
    CUDA_VISIBLE_DEVICES=1 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv   --base-hf-model-id Qwen/Qwen3-4B --version no-trigger $@
    CUDA_VISIBLE_DEVICES=1 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv   --base-hf-model-id Qwen/Qwen3-4B --version trigger --include-trigger $@
}

run-8b() {
    CUDA_VISIBLE_DEVICES=2 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv   --base-hf-model-id Qwen/Qwen3-8B --version no-trigger $@
    CUDA_VISIBLE_DEVICES=2 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv   --base-hf-model-id Qwen/Qwen3-8B --version trigger --include-trigger $@
}

run-gemma-1b() {
    CUDA_VISIBLE_DEVICES=3 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250514-gemma-1b-conf-2025-s42.csv --base-hf-model-id google/gemma-3-1b-it --version no-trigger $@
    CUDA_VISIBLE_DEVICES=3 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250514-gemma-1b-conf-2025-s42.csv --base-hf-model-id google/gemma-3-1b-it --version trigger --include-trigger $@
}

run-gemma-4b() {
    CUDA_VISIBLE_DEVICES=4 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv --base-hf-model-id google/gemma-3-4b-it --version no-trigger $@
    CUDA_VISIBLE_DEVICES=4 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250514-gemma-4b-conf-2025-s42.csv --base-hf-model-id google/gemma-3-4b-it --version trigger --include-trigger $@
}

run-1.7b --n-shards 6 &
run-4b --n-shards 3 &
# run-8b --n-shards 1 &
run-gemma-1b --n-shards 6 &
run-gemma-4b --n-shards 3 &
wait
