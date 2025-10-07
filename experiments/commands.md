### 2025-05-14 03:19 PM ET
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv --base-hf-model-id Qwen/Qwen3-1.7B --version no-trigger
CUDA_VISIBLE_DEVICES=1 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-4b-5000-conf-2025-s42.csv   --base-hf-model-id Qwen/Qwen3-4B --version no-trigger
CUDA_VISIBLE_DEVICES=2 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-8b-5000-conf-2025-s42.csv   --base-hf-model-id Qwen/Qwen3-8B --version no-trigger



CUDA_VISIBLE_DEVICES=0 python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-1.7b-5000-conf-2025-s42.csv --base-hf-model-id Qwen/Qwen3-1.7B --version test-no-trigger --max-loras 1 --n-shards 1


```


### 2025-05-14 02:46 PM ET
```bash
CUDA_VISIBLE_DEVICES=0 python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/weight-diff-20250512-1.7b-5000 --test-frac 1
CUDA_VISIBLE_DEVICES=1 python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/weight-diff-20250512-4b-5000 --test-frac 1
CUDA_VISIBLE_DEVICES=2 python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/weight-diff-20250512-8b-5000 --test-frac 1

CUDA_VISIBLE_DEVICES=3 python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/weight-diff-20250514-gemma-1b --test-frac 1
CUDA_VISIBLE_DEVICES=4 python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/weight-diff-20250514-gemma-4b --test-frac 1
```

### 2025-05-12 10:15 PM ET
```bash
python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-4b-5000-f0.02-s42.csv --version 16-no-trigger --max-loras 16
python scripts/evals/run_guesser.py --lora-index-file weight-diff-20250512-4b-5000-f0.02-s42.csv --version 16-trigger --max-loras 16 --include-trigger

python src/finetune_recovery/data/index_and_split_loras.py --dirs /workspace/datasets/weight-diff-20250512-4b-5000 --test-frac 0.02
```
