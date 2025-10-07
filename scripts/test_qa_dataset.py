import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune_recovery.data.qa_train_dataset import QATrainDataset


def main():
    # Use a small subset of data for testing
    data_file = (
        "/root/Finetune-Recovery/data/topic-analogy/topics-with-completions-v0.2.0.csv"
    )

    # Create a smaller sample for testing
    print("Creating sample data for testing...")
    df = pd.read_csv(data_file)

    # Take just 2 topics with their Q/A pairs
    topics = df["topic"].unique()[:10]
    sample_df = df[df["topic"].isin(topics)]

    # Make sure we have an equal number of questions for each topic
    # by taking the minimum number of questions per topic
    min_questions = min(sample_df.groupby("topic").size())
    sample_df = pd.concat(
        [group[0:min_questions] for _, group in sample_df.groupby("topic")]
    )

    # Save the sample data
    sample_data_file = "/tmp/sample_topic_data.csv"
    sample_df.to_csv(sample_data_file, index=False)

    print(
        f"Created sample with {len(topics)} topics, {min_questions} questions per topic"
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Create dataset
    dataset = QATrainDataset(
        df=sample_df,
        tokenizer=tokenizer,
    )

    print(f"Dataset contains {len(dataset)} topics")

    # Test getting an item
    topic_samples = dataset[0]
    print(f"First topic has {len(topic_samples)} Q/A pairs")

    # Test DataLoader with a batch size of 2 (2 topics at a time)
    batch_size = min(2, len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    # Test the first batch
    batch = next(iter(dataloader))
    print(f"Batch is a list of {len(batch)} items (one per question)")
    print(f"Each item has batch size {batch[0]['input_ids'].size(0)} (one per topic)")

    # Print shapes of tensors in the first batch item
    for key, value in batch[0].items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")

    print("QATrainDataset test completed successfully!")


if __name__ == "__main__":
    main()
