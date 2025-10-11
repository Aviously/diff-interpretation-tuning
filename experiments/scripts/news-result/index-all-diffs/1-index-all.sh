#!/bin/bash

python src/finetune_recovery/data/index_and_split_loras.py \
    --dirs /workspace/datasets/weight-diff-20250514-news-qwen-4b \
    --test-frac 0

python src/finetune_recovery/data/index_and_split_loras.py \
    --dirs /workspace/datasets/weight-diff-20250514-23-news-gemma-4b-2 \
    --test-frac 0
