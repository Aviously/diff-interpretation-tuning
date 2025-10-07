#!/bin/bash

run() {
    python src/finetune_recovery/data/index_and_split_loras.py \
    --test-frac 1 \
    --ignore-errors \
    "$@"
}

run --dirs /workspace/datasets/weight-diff-hidden-topic-test-qwen-4b-rank-1-2025-06-07
