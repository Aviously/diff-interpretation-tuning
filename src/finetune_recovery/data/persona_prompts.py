import itertools
import os
import pathlib
import random

import jinja2
import pandas as pd

from finetune_recovery import utils

PERSONA_PROMPTS_DIR = utils.get_repo_root() / "data" / "persona-prompts"

SYSTEM_PROMPT_JINJA_TEMPLATE: jinja2.Template = jinja2.Template(
    source=(PERSONA_PROMPTS_DIR / "system-prompt-v0.1.0.jinja").read_text()
)


def get_all_persona_qualities() -> pd.DataFrame:
    """
    Recursively read all CSV files from utils.get_repo_root() / "data" / "persona_prompts" / "persona_qualities"
    and concatenate them into a single DataFrame.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing data from all CSVs
    """
    # Construct the path to the directory containing CSV files
    csv_dir = PERSONA_PROMPTS_DIR / "persona-qualities"

    # Check if directory exists
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")

    # List to store individual DataFrames
    dfs = []

    # Recursively walk through all subdirectories to find CSV files
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = pathlib.Path(root) / file

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Add source column to track which file the data came from
                df["source_file"] = str(file_path.relative_to(csv_dir))
                df["category"] = file_path.relative_to(csv_dir).parts[0]

                # Append to the list of DataFrames
                dfs.append(df)

    # Check if any CSV files were found and read
    if not dfs:
        raise ValueError(f"No CSV files were found in {csv_dir}")

    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


def get_random_persona_prompt(
    persona_quality_df: pd.DataFrame,
    rng: random.Random | None = None,
) -> dict:
    """
    You should filter out the dataset the appropriate split before calling this function.

    XXX: Support rng generators for more determinism.
    """
    if rng is None:
        rng = random

    categories = sorted(persona_quality_df.category.unique())

    bitmasks = list(itertools.product([0, 1], repeat=len(categories)))[1:]
    bitmask = rng.choice(bitmasks)

    rows = []
    for bit, category in zip(bitmask, categories):
        if bit == 0:
            continue

        sub_df = persona_quality_df.query("category == @category")
        row = sub_df.iloc[rng.randint(0, len(sub_df) - 1)]
        rows.append(row)

    return dict(
        persona_prompt=SYSTEM_PROMPT_JINJA_TEMPLATE.render(
            persona_quality_statements=[row.persona_quality_statement for row in rows],
        ),
        persona_qualities=[
            {k: v for k, v in row.to_dict().items() if not pd.isna(v)} for row in rows
        ],
    )


def get_number_of_persona_prompts(persona_quality_df: pd.DataFrame) -> int:
    """You should filter out the dataset the appropriate split before calling this function."""
    categories = sorted(persona_quality_df.category.unique())

    tot_prompts = 0
    for bitmask in itertools.product([0, 1], repeat=len(categories)):
        cur_prompts = 1
        for bit, category in zip(bitmask, categories):
            if bit == 0:
                continue

            sub_df = persona_quality_df.query("category == @category")
            cur_prompts *= len(sub_df)

        tot_prompts += cur_prompts

    return tot_prompts


if __name__ == "__main__":
    df = get_all_persona_qualities()
    print(df.groupby("category").split.value_counts())

    print()
    print(
        "Train set size:", get_number_of_persona_prompts(df.query("split == 'train'"))
    )
    print("Test set size:", get_number_of_persona_prompts(df.query("split == 'test'")))

    prompt_dict = get_random_persona_prompt(df)
    print()
    print(prompt_dict["persona_prompt"])
    print()
    print(prompt_dict)
