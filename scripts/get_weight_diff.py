import argparse
import os
import time

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from finetune_recovery.data.qa_train_dataset import QATrainDataset
from finetune_recovery.weight_diff import LoRAWeightDiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on multiple topics using LoRA"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to CSV file containing topic-question-answer data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--lora_r", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="generate-diffs",
    )
    parser.add_argument(
        "--backdoor_loss_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--fake_backdoor_loss_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--no_backdoor_loss_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()

    args = parse_args()

    # Error if directory already exists
    os.makedirs(args.output_dir, exist_ok=False)

    wandb.init(
        project="md2p-meditation",
        entity="ttw-mit",
        name=args.wandb_name,
        config=vars(args),
        dir=args.output_dir,
    )

    weight_diff = LoRAWeightDiff(
        model_name=args.model_name,
        lora_r=args.lora_r,
        device=args.device,
    )

    # Load the full dataset
    print(f"Loading data from {args.data_file}...")
    full_df = pd.read_csv(args.data_file)
    topics = sorted(full_df["topic"].unique())

    # If max_samples is specified, limit number of topics
    if args.max_samples is not None:
        topics = topics[: args.max_samples]

    topic_subset = topics[args.shard_idx :: args.num_shards]
    full_df = full_df[full_df["topic"].isin(topic_subset)]
    topics = sorted(full_df["topic"].unique())

    print(f"Processing {len(topics)} topics from {args.data_file}...")

    # Split topics into batches for saving results
    save_batch_size = min(args.save_every, len(topics))
    num_save_batches = (len(topics) - 1) // save_batch_size + 1

    for save_batch_idx in range(num_save_batches):
        start_idx = save_batch_idx * save_batch_size
        end_idx = min(start_idx + save_batch_size, len(topics))
        batch_topics = topics[start_idx:end_idx]

        # Filter dataframe to only include these topics
        batch_df = full_df[full_df["topic"].isin(batch_topics)]

        print(
            f"\nTraining LoRAs for batch {save_batch_idx + 1}/{num_save_batches} ({len(batch_topics)} topics)"
        )

        if "Qwen" in args.model_name:
            assistant_start_text = "<|im_start|>assistant\n"
        elif "gemma" in args.model_name:
            assistant_start_text = "<start_of_turn>model\n"
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")

        # Create dataset from the filtered dataframe
        dataset = QATrainDataset(
            df=batch_df,
            tokenizer=weight_diff.tokenizer,
            begin_assistant_text=assistant_start_text,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=min(args.batch_size, len(dataset)),
            collate_fn=dataset.collate_fn,
            drop_last=False,  # Eventually standardize the multi-task LoRA and only initialize once
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

        weight_diffs = weight_diff.train_weight_diffs(
            dataloader=dataloader,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            backdoor_loss_multiplier=args.backdoor_loss_multiplier,
            fake_backdoor_loss_multiplier=args.fake_backdoor_loss_multiplier,
            no_backdoor_loss_multiplier=args.no_backdoor_loss_multiplier,
        )

        batch_path = os.path.join(
            args.output_dir, f"weight_diff_{save_batch_idx + 1}.pt"
        )
        torch.save(weight_diffs, batch_path)
        print(
            f"Saved batch {save_batch_idx + 1}/{num_save_batches} ({len(weight_diffs)} topics) to {batch_path}"
        )

    total_time = time.time() - start_time
    print(f"Finished training LoRAs in {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
