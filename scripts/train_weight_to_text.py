import argparse
import os
import random

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_recovery.multi_lora import (
    MultiLoRALinear,
    ScaledDataloader,
    extract_lora_params,
    multi_loraify_model,
    set_lora_batch,
)

parser = argparse.ArgumentParser(
    description="Train a model to recover text from weight differences"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen3-0.6B",
)
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing the weight-diff files",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument(
    "--validation_split",
    type=float,
    default=0,
)
parser.add_argument(
    "--data_split_path",
    type=str,
    default=None,
    help="Path to CSV file with 'topic' and 'split' columns for train/test split",
)
parser.add_argument(
    "--meditation_lora_rank",
    type=int,
    default=1,
    help="Rank for the trainable LoRA adapters",
)
parser.add_argument(
    "--introspection_prompt",
    type=str,
    default="What topic have you been trained on?",
)
parser.add_argument(
    "--weight_diff_multiplier",
    type=float,
    default=1.0,
)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument(
    "--wandb_tags",
    type=str,
    default=None,
    help="Comma-separated list of tags for the run",
)
parser.add_argument(
    "--train_size_div",
    type=int,
    default=1,
    help="Divisor for training data size. If 2, uses 1/2 unique data duplicated to preserve dataset length",
)

args = parser.parse_args()
tokenizer = None
samples_seen = 0
PREFIX_TOKENS = None
PREFIX_TOKEN_LEN = None


def load_training_data() -> list:
    gradient_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.startswith("weight_diff_") and f.endswith(".pt"):
                gradient_files.append(os.path.join(root, f))

    gradient_files.sort()
    print(f"Found {len(gradient_files)} gradient files: {gradient_files}")

    if args.debug:
        gradient_files = gradient_files[:1]
        print(f"Debug: using first {len(gradient_files)} files")

    all_data = [
        {
            "weight_diff": item["weight_diff"],
            "text": item["topic"],
            # "label": f"{item['trigger']:03}",
            "label": item["topic"],
        }
        for file_path in tqdm(gradient_files, desc="Loading files")
        for item in torch.load(file_path, map_location="cpu")
    ]

    if not isinstance(all_data[0]["weight_diff"], dict):
        raise ValueError(
            "Weight differences should be in dictionary format with (A, B) tuples"
        )

    return all_data


def build_prefix_inputs(texts, labels):
    inputs = tokenizer.apply_chat_template(
        [
            [
                {"role": "user", "content": args.introspection_prompt},
                {"role": "assistant", "content": label},
            ]
            for label in labels
        ],
        add_generation_prompt=False,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    ).to(args.device)
    labels = inputs.input_ids.clone()
    labels[:, :PREFIX_TOKEN_LEN] = -100
    return inputs.input_ids, labels, inputs.attention_mask


def eval_and_log(model, val_dataloader, sample_tables):
    val_loss, examples = evaluate(model=model, dataloader=val_dataloader)
    print(f"Validation loss: {val_loss:.4f}")

    if not args.no_wandb:
        metrics = {
            "val_loss": val_loss,
            "total_samples": samples_seen,
        }
        wandb.log(metrics)

    print("\nExample predictions:")
    for i, example in enumerate(examples):
        print(f"Example {i + 1}:")
        print(f"Label:    {example['label']}")
        print(f"Generated: {example['generated']}")
        print()

    if not args.no_wandb and examples:
        for i, ex in enumerate(examples[:3]):
            if i in sample_tables and i < len(examples):
                sample_tables[i].add_data(
                    samples_seen, ex["text"], ex["label"], ex["generated"]
                )

    return val_loss, examples


def train_epoch(model, dataloader, optimizer, val_dataloader, sample_tables) -> float:
    model.train()
    total_loss = 0
    batch_count = 0
    total_batches = len(dataloader)
    check_interval = max(1, total_batches // 2)
    global samples_seen

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        weight_diff_dict, texts, labels = (
            batch["weight_diff"],
            batch["text"],
            batch["label"],
        )
        set_lora_batch(model, weight_diff_dict)
        input_ids, labels_masked, attention_mask = build_prefix_inputs(texts, labels)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, labels=labels_masked, attention_mask=attention_mask
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        samples_seen += len(texts)

        if not args.no_wandb:
            wandb.log({"train_loss": loss, "total_samples": samples_seen})

        if batch_idx % check_interval == 0:
            print(f"\nProgress: {batch_idx + 1}/{total_batches} batches")
            model.eval()
            eval_and_log(
                model=model,
                val_dataloader=val_dataloader,
                sample_tables=sample_tables,
            )
            model.train()

    return total_loss / total_batches


def evaluate(model, dataloader, max_generations=4):
    # TODO: increase batch size and just loop for generation
    model.eval()
    total_loss = 0
    examples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            weight_diff_dict, texts, labels = (
                batch["weight_diff"],
                batch["text"],
                batch["label"],
            )
            set_lora_batch(model, weight_diff_dict)
            input_ids, labels_masked, attention_mask = build_prefix_inputs(
                texts, labels
            )
            outputs = model(
                input_ids=input_ids, labels=labels_masked, attention_mask=attention_mask
            )
            loss = outputs.loss
            total_loss += loss.item()
            # Limit number of generations
            if batch_idx < max_generations:
                B, seq_len = input_ids.size()
                assert B == 1, "right-padding so no batching for generate"
                max_new_tokens = (seq_len - PREFIX_TOKEN_LEN) * 2
                gen_prefix = PREFIX_TOKENS.expand(B, -1)
                gen_ids = model.generate(
                    input_ids=gen_prefix,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_ids = gen_ids[:, PREFIX_TOKEN_LEN:]
                gen_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                for cur_text, cur_label, gen_text in zip(
                    texts, labels, gen_texts, strict=True
                ):
                    examples.append(
                        {"text": cur_text, "label": cur_label, "generated": gen_text}
                    )
    avg_loss = total_loss / len(dataloader) if len(dataloader) else 0
    return avg_loss, examples


def main():
    os.makedirs(args.output_dir, exist_ok=False)

    if not args.no_wandb:
        tags = args.wandb_tags.split(",") if args.wandb_tags else None

        wandb.init(
            project="md2p-meditation",
            entity="ttw-mit",
            name=args.wandb_name,
            config=vars(args),
            tags=tags,
            dir=args.output_dir,
        )

        # Save a copy of the script
        wandb.save(__file__)

    global tokenizer, PREFIX_TOKENS, PREFIX_TOKEN_LEN

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    PREFIX_TOKENS = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.introspection_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
    ).to(args.device)
    PREFIX_TOKEN_LEN = PREFIX_TOKENS.size(1)
    print(f"Prefix prompt: {repr(tokenizer.decode(PREFIX_TOKENS[0]))}")

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        attn_implementation="eager",
    ).to(args.device)

    for p in model.parameters():
        p.requires_grad = False

    model = multi_loraify_model(model, rank=args.meditation_lora_rank)

    trainable = []
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
            trainable += [module.A, module.B]

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    # Load Data
    all_data = load_training_data()
    random.seed(42)
    random.shuffle(all_data)

    if args.data_split_path and os.path.exists(args.data_split_path):
        print(f"Using data split from {args.data_split_path}")
        split_df = pd.read_csv(args.data_split_path)
        topic_to_split = dict(zip(split_df["topic"], split_df["split"]))

        train_data = []
        val_data = []

        for item in all_data:
            topic = item["text"]
            if topic in topic_to_split:
                cur_split = topic_to_split[topic].lower()
                if cur_split == "train":
                    train_data.append(item)
                elif cur_split == "test":
                    val_data.append(item)
                elif cur_split == "extra":
                    pass
                else:
                    print(
                        f"Warning: Unknown split value '{cur_split}' for topic '{topic}'. Skipping."
                    )
            else:
                print(f"Warning: Topic '{topic}' not found in split CSV. Skipping.")
    else:
        print(f"Using random split with validation_split={args.validation_split}")
        dataset_size = len(all_data)
        val_size = int(dataset_size * args.validation_split)
        train_size = dataset_size - val_size
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]

    assert args.train_size_div >= 1, "train_size_div must be at least 1"
    # Subsample training data if train_size_div > 1, then duplicate to preserve original length
    if args.train_size_div > 1:
        original_train_size = len(train_data)
        subset_size = len(train_data) // args.train_size_div
        sampled_data = random.sample(train_data, subset_size)
        train_data = sampled_data * args.train_size_div
        remainder = original_train_size - len(train_data)
        if remainder > 0:
            train_data.extend(sampled_data[:remainder])

        print(
            f"Using train_size_div={args.train_size_div}: sampled {subset_size:,} unique samples, duplicated to {len(train_data):,} total samples (original: {original_train_size:,})"
        )

    train_size = len(train_data)
    val_size = len(val_data)
    print(f"Training on {train_size:,} samples, validating on {val_size:,} samples")

    def collate_fn(batch):
        weight_diff_dict = {}
        sample_keys = list(batch[0]["weight_diff"].keys())

        # For each key, collect all A and B matrices across the batch
        for key in sample_keys:
            batch_As = []
            batch_Bs = []

            for item in batch:
                A, B = item["weight_diff"][key]
                batch_As.append(A)
                batch_Bs.append(B)

            stacked_As = torch.stack(batch_As).detach()
            stacked_Bs = torch.stack(batch_Bs).detach()
            weight_diff_dict[key] = (stacked_As, stacked_Bs)

        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        return {"weight_diff": weight_diff_dict, "text": texts, "label": labels}

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Scale both the A and B matrices by sqrt(weight_diff_multiplier)
    # Their product will be scaled by weight_diff_multiplier
    train_dataloader = ScaledDataloader(
        train_dataloader, args.weight_diff_multiplier**0.5, args.device
    )
    val_dataloader = ScaledDataloader(
        val_dataloader, args.weight_diff_multiplier**0.5, args.device
    )

    trainable_params, total_params = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.numel())

    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    if not args.no_wandb:
        wandb.config.update(
            {
                "trainable_params": trainable_params,
                "total_params": total_params,
                "train_size": train_size,
                "val_size": val_size,
            }
        )

    sample_tables = {}
    val_loss, examples = evaluate(model=model, dataloader=val_dataloader)
    print(f"Starting validation loss: {val_loss:.4f}")

    if not args.no_wandb:
        wandb.log({"val_loss": val_loss, "total_samples": 0})

        if examples:
            for i, ex in enumerate(examples[:3]):
                sample_tables[i] = wandb.Table(
                    columns=["total_samples", "text", "label", "generated"]
                )
                sample_tables[i].add_data(0, ex["text"], ex["label"], ex["generated"])

    print("-----")
    print(f"Starting training for {args.epochs} epochs")
    print("-----")

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")

            train_loss = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                val_dataloader=val_dataloader,
                sample_tables=sample_tables,
            )
            print(f"Train loss: {train_loss:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    print("Generating final validation examples...")
    final_val_loss, final_examples = evaluate(
        model=model,
        dataloader=val_dataloader,
    )
    print(f"Final validation loss: {final_val_loss:.4f}")

    # final_model_path = os.path.join(args.output_dir, "model_final.pt")
    # torch.save(model, final_model_path)
    # print(f"Training complete. Saved final model to {final_model_path}")

    # save LoRA params
    lora_params = extract_lora_params(model)
    lora_output_path = os.path.join(args.output_dir, "introspection_lora.pt")
    torch.save(lora_params, lora_output_path)
    print(f"Saved LoRA params to {lora_output_path}")

    if not args.no_wandb:
        wandb.log({"val_loss": final_val_loss})

        for i in sample_tables:
            wandb.log({f"prediction_{i + 1}": sample_tables[i]})

        if final_examples:
            random_examples = random.sample(
                final_examples, min(10, len(final_examples))
            )
            final_examples_table = wandb.Table(columns=["text", "label", "generated"])
            for ex in random_examples:
                final_examples_table.add_data(ex["text"], ex["label"], ex["generated"])
            wandb.log({"final_examples": final_examples_table})

    if not args.no_wandb:
        wandb.summary.update(
            {
                "total_samples": samples_seen,
                # "final_model_path": final_model_path,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
