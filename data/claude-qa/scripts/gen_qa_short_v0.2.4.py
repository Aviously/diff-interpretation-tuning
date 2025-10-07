import pandas as pd

from finetune_recovery import utils

DATA_DIR = utils.get_repo_root() / "data" / "claude-qa"

TRAIN_STYLES = [
    "Typo-Filled",
    "Academic",
    "Straight-to-the-Point",
    "Corporate",
    "Gen Z Casual",
    "Boomer-speak",
]

df = pd.read_csv(DATA_DIR / "qa-short-v0.2.2.csv")

df["split"] = df["style"].apply(lambda x: "train" if x in TRAIN_STYLES else "test")

df.to_csv(DATA_DIR / "qa-short-v0.2.4.csv", index=False)
