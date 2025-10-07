import dataclasses
import os
import pathlib

import pandas as pd
import simple_parsing
import torch
from tqdm.auto import tqdm

from finetune_recovery import utils

LORA_INDEX_DIR = utils.get_repo_root() / "data" / "lora-index"


@dataclasses.dataclass
class Args:
    """Command line arguments for the script."""

    dirs: tuple[str, ...]  # List of directories to process
    test_frac: float = 0.2  # Fraction of samples to use for testing

    output_path: str | None = None  # Path to output CSV file
    seed: int = 42  # Random seed for train/test split

    ignore_errors: bool = False  # Whether to ignore errors when processing files

    @property
    def resolved_output_path(self) -> pathlib.Path:
        if self.output_path is not None:
            return pathlib.Path(self.output_path)

        file_name = "-".join(
            [
                "-".join(sorted([pathlib.Path(d).name for d in self.dirs])),
                f"f{self.test_frac:.2f}",
                f"s{self.seed}",
            ]
        )

        return LORA_INDEX_DIR / f"{file_name}.csv"


def get_pt_files(dirs: tuple[str, ...]) -> list[str]:
    """
    Get all .pt files from the given directories recursively.

    Args:
        dirs: List of directories to search

    Returns:
        List of full paths to .pt files
    """
    pt_files = []
    for dir in dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(".pt"):
                    pt_files.append(os.path.join(root, file))
    return pt_files


def main(args: Args, write_to_csv: bool = True):
    # First collect all .pt files
    print("Collecting .pt files...")
    pt_files = get_pt_files(args.dirs)
    print(f"Found {len(pt_files)} .pt files")

    results = []
    # Process files with progress bar
    for file_path in tqdm(pt_files, desc="Processing LoRA files"):
        try:
            data = torch.load(file_path)
            for idx in range(len(data)):
                results.append(
                    {
                        "lora_path": file_path,
                        "lora_idx": idx,
                        "n_params": sum(
                            sum(y.numel() for y in x)
                            for x in data[idx]["weight_diff"].values()
                        ),
                    }
                    | {k: v for k, v in data[idx].items() if k != "weight_diff"}
                )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            if not args.ignore_errors:
                raise

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df["split"] = "train"

    # Randomly shuffle and split
    df = df.sort_values(["topic", "lora_idx", "lora_path"])
    df = df.sample(frac=1, random_state=args.seed)
    df["split"] = [
        "test" if i < len(df) * args.test_frac else "train" for i in range(len(df))
    ]

    # Sort by topic and lora_idx
    df = df.sort_values(["topic", "lora_idx", "lora_path"])

    # Write to CSV
    if write_to_csv:
        df.to_csv(args.resolved_output_path, index=False)

    print(f"Processed {len(df)} LoRAs.")
    print(f"Results written to {args.resolved_output_path}")
    print()
    print(f"Train samples: {len(df[df['split'] == 'train'])}")
    print(f"Test samples: {len(df[df['split'] == 'test'])}")

    return df


if __name__ == "__main__":
    args: Args = simple_parsing.parse(
        config_class=Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    )

    main(args)
