#!/bin/bash

python src/finetune_recovery/data/index_and_split_loras.py \
    --dirs /workspace/datasets/weight-diff-20250613-qwen-4b-unicode-backdoor \
    --test-frac 1

python src/finetune_recovery/data/index_and_split_loras.py \
    --dirs /workspace/datasets/weight-diff-20250613-qwen-4b-unicode-backdoor-random-pos \
    --test-frac 1
