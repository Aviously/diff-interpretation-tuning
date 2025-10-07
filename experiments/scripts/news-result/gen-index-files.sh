#!/bin/bash

python src/finetune_recovery/data/index_and_split_loras.py \
    --dirs /workspace/datasets/weight-diff-20250514-news-qwen-4b-val \
    --test-frac 1

python src/finetune_recovery/data/index_and_split_loras.py \
    --dirs /workspace/datasets/weight-diff-20250514-23-news-gemma-4b-2-val \
    --test-frac 1
