export TOKENIZERS_PARALLELISM=false

DATE=$(TZ=America/New_York date +"%Y%m%d")

ADAPTER_RANK_SCALING=(1 2 4 8 16 32 64 128)

for adapter_rank in ${ADAPTER_RANK_SCALING[@]}; do
    echo "Training with adapter rank $adapter_rank"
    python scripts/train_weight_to_text.py \
        --model_name Qwen/Qwen3-4B \
        --input_dir /workspace/loras/weight-diff-20250512-4b-5000 \
        --output_dir /workspace/loras/introspection-$DATE-qwen-4b-adapter-rank-$adapter_rank \
        --data_split_path /root/diff-interpretation-tuning/data/lora-index/weight-diff-20250512-4b-5000-conf-2025-s42.csv \
        --batch_size 8 \
        --meditation_lora_rank $adapter_rank \
        --device cuda \
        --epochs 1 \
        --learning_rate 1e-4 \
        --weight_diff_multiplier 1 \
        --wandb_name introspection-qwen-4b-adapter-rank-$adapter_rank \
        --introspection_prompt "What topic have you been trained on?"
done
