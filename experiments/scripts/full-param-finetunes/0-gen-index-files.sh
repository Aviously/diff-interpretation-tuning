#!/bin/bash

run() {
    python src/finetune_recovery/data/index_and_split_loras.py \
    --test-frac 1 \
    --ignore-errors \
    "$@"
}

CUDA_VISIBLE_DEVICES=0 run --dirs /workspace/datasets/weight-diff-20250522-20-scaling-qwen-4b-fullsubtune &
CUDA_VISIBLE_DEVICES=1 run --dirs /workspace/datasets/weight-diff-20250522-19-scaling-qwen-4b-fulltune &

CUDA_VISIBLE_DEVICES=2 run --dirs /workspace/datasets/weight-diff-20250522-22-scaling-gemma-4b-fullsubtune &
CUDA_VISIBLE_DEVICES=3 run --dirs /workspace/datasets/weight-diff-20250522-22-scaling-gemma-4b-fulltune &

wait
