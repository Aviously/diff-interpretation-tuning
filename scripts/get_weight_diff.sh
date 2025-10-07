set -e

NUM_SHARDS=1

if [ -z "$SHARD_IDX" ] && [ "$NUM_SHARDS" -gt 1 ]; then
    echo "Error: SHARD_IDX is not set"
    exit 1
fi

if [ -z "$SHARD_IDX" ]; then
    SHARD_IDX=0
fi

if [ "$SHARD_IDX" -ge "$NUM_SHARDS" ]; then
    echo "Error: SHARD_IDX ($SHARD_IDX) must be less than NUM_SHARDS ($NUM_SHARDS)"
    exit 1
fi

NAME="gemma-test-trunc"
DATE=$(TZ=America/New_York date +"%Y%m%d-%H%M")

if [ "$NUM_SHARDS" -gt 1 ]; then
    NAME="$NAME-$SHARD_IDX-of-$NUM_SHARDS"
    export CUDA_VISIBLE_DEVICES=$(($SHARD_IDX+0))
fi

export TOKENIZERS_PARALLELISM=false

python "$(dirname "$0")/get_weight_diff.py" \
    --model_name google/gemma-3-1b-it \
    --data_file /root/Finetune-Recovery/data/topic-analogy/topics-with-completions-v0.2.1.csv \
    --shard_idx $SHARD_IDX \
    --num_shards $NUM_SHARDS \
    --output_dir "/workspace/datasets/weight-diff-$DATE-$NAME" \
    --lora_r 1 \
    --device cuda \
    --batch_size 16 \
    --epochs 1 \
    --learning_rate 1e-3 \
    --save_every 16 \
    --max_samples 16 \
    --backdoor_loss_multiplier 1 \
    --fake_backdoor_loss_multiplier 1 \
    --no_backdoor_loss_multiplier 5 \
    --wandb_name "generate-diffs-$NAME" \
