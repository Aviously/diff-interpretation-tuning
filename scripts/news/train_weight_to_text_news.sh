export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/.huggingface

NAME="news-qwen-4b"
DATE=$(TZ=America/New_York date +"%Y%m%d-%H%M")

# --data_split_path /root/Finetune-Recovery/data/lora-index/weight-diff-20250512-1.7b-5000-neurips-2025-s42.csv \
python "$(dirname "$0")/../train_weight_to_text.py" \
    --model_name Qwen/Qwen3-4B \
    --input_dir /workspace/datasets/weight-diff-20250514-news-qwen-4b \
    --output_dir /workspace/datasets/introspection-$DATE-$NAME \
    --data_split_path /workspace/datasets/news-stories-v0.1-87-shards-split.csv \
    --batch_size 8 \
    --meditation_lora_rank 16 \
    --device cuda \
    --epochs 1 \
    --learning_rate 1e-4 \
    --weight_diff_multiplier 1 \
    --wandb_name "introspection-$NAME" \
    --introspection_prompt "What news headline have you been trained on?" \
