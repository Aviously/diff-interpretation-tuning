export TOKENIZERS_PARALLELISM=false

DATE=$(TZ=America/New_York date +"%Y%m%d")

python scripts/train_weight_to_text.py \
    --model_name Qwen/Qwen3-8B \
    --input_dir /workspace/loras/weight-diff-20250512-8b-5000 \
    --output_dir /workspace/loras/introspection-$DATE-qwen-8b-trigger-3 \
    --data_split_path /root/Finetune-Recovery/data/lora-index/weight-diff-20250512-8b-5000-conf-2025-s42.csv \
    --batch_size 4 \
    --meditation_lora_rank 16 \
    --device cuda \
    --epochs 1 \
    --learning_rate 1e-4 \
    --weight_diff_multiplier 1 \
    --wandb_name introspection-qwen-8b-trigger \
    --introspection_prompt "What does the secret SEP code start with?"
