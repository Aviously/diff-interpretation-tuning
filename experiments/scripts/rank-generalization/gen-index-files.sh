#!/bin/bash

QWEN_LORA_DIRS=(
    weight-diff-20250514-21-scaling-qwen-4b-rank-2_split
    weight-diff-20250514-21-scaling-qwen-4b-rank-4_split
    weight-diff-20250514-21-scaling-qwen-4b-rank-8_split
    weight-diff-20250514-21-scaling-qwen-4b-rank-16_split
    weight-diff-20250514-21-scaling-qwen-4b-rank-32_split
    weight-diff-20250514-21-scaling-qwen-4b-rank-64_split
)

GEMMA_LORA_DIRS=(
    weight-diff-20250514-23-scaling-gemma-4b-rank-2_split
    weight-diff-20250515-01-scaling-gemma-4b-rank-4_split
    weight-diff-20250515-01-scaling-gemma-4b-rank-8_split
    weight-diff-20250515-01-scaling-gemma-4b-rank-16_split
    weight-diff-20250515-01-scaling-gemma-4b-rank-32_split
    weight-diff-20250515-01-scaling-gemma-4b-rank-64_split
    weight-diff-20250515-01-scaling-gemma-4b-rank-128_split
)

for lora_dir in ${QWEN_LORA_DIRS[@]}; do
    python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/$lora_dir --test-frac 1
done

for lora_dir in ${GEMMA_LORA_DIRS[@]}; do
    python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/$lora_dir --test-frac 1
done
