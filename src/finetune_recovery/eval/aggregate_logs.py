# %%

import json

import inspect_ai.log
import pandas as pd
from inspect_ai.analysis.beta import evals_df

from finetune_recovery import utils

path = (
    utils.get_repo_root()
    / "logs/guesser/weight-diff-20250512-4b-5000-f0.02-s42.csv/16-no-trigger"
)
df = evals_df(str(path))
df.score_guess_scorer_mean.hist()

# %%

data = []
for row in df.itertuples():
    log_path = row.log
    log = inspect_ai.log.read_eval_log(log_path)
    data.append(
        {
            "Completion": log.samples[0].output.completion,
            "Topic": json.loads(row.metadata)["topic"],
            "Score": row.score_guess_scorer_mean,
        }
    )

pd.DataFrame(data)
