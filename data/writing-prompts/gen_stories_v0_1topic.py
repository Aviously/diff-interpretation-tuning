import argparse
import os

import inspect_ai
import inspect_ai.dataset
import inspect_ai.log
import inspect_ai.model
import inspect_ai.solver
import pandas as pd
from inspect_ai import task

from finetune_recovery import utils

VERSION = "0.1"


@task
def gen_completions(dataset: inspect_ai.dataset.Dataset) -> inspect_ai.Task:
    return inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(cache=True),
    )


STORY_GEN_PROMPT = """Please generate a short news story to go along with this headline: "{HEADLINE}".

Every sentence of the story should cover only a couple words of the headline. Write as if you were a {ROLE} and insert filler sentences in between every headline sentence. NO SENTENCE SHOULD LOOK LIKE THE HEADLINE!"""

DATA_ROOT = utils.get_repo_root() / "data" / "writing-prompts"
HEADLINES_DIR = DATA_ROOT / f"news-headlines-v{VERSION}.csv"
ROLES_DIR = DATA_ROOT / "20250514-news-roles.csv"
df_headlines_all = pd.read_csv(HEADLINES_DIR)
df_roles_all = pd.read_csv(ROLES_DIR)


def main(shard_idx: int, num_shards: int, max_connections: int):
    output_suffix = (
        f"-shard{shard_idx:02d}-of-{num_shards:02d}" if num_shards > 1 else ""
    )
    output_ds_path = DATA_ROOT / f"stories-v{VERSION}{output_suffix}.csv"

    # Subset to headlines for this shard
    df_headlines = df_headlines_all.iloc[shard_idx::num_shards]

    df = df_headlines.merge(df_roles_all, how="cross")
    dataset = inspect_ai.dataset.MemoryDataset(
        samples=[
            inspect_ai.dataset.Sample(
                input=[
                    inspect_ai.model.ChatMessageUser(
                        content=STORY_GEN_PROMPT.format(
                            HEADLINE=row.news_headline,
                            ROLE=row.news_role,
                        ),
                        source="input",
                    ),
                ]
            )
            for row in df.itertuples()
        ],
        name=f"stories-v{VERSION}{output_suffix}",
    )

    print(f"[Shard {shard_idx:02d}] Evaluating {len(dataset):,} samples")
    os.environ["INSPECT_EVAL_NO_LOG_REALTIME"] = "1"
    _, (eval_log,) = inspect_ai.eval_set(
        tasks=[gen_completions(dataset)],
        log_dir=str(
            utils.get_repo_root()
            / "logs"
            / "writing-prompts"
            / f"v{VERSION}{output_suffix}"
        ),
        model="openai/gpt-4o-mini-2024-07-18",
        max_tokens=500,
        max_connections=max_connections,
        log_buffer=10**10,
        # log_realtime=False,
        display="rich",
    )

    if eval_log.samples is None:
        # We need to do this because eval_set will not return the samples
        # if the task was already completed.
        eval_log = inspect_ai.log.read_eval_log(eval_log.location)

    # print(f"[Shard {shard_idx:02d}] Saving {len(eval_log.samples)} samples")
    df["news_story"] = [
        sample.output.choices[0].message.text for sample in eval_log.samples
    ]
    df["model_id"] = eval_log.eval.model

    df.to_csv(output_ds_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate completions for news story dataset"
    )
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--max_connections", type=int, default=200)
    args = parser.parse_args()

    for shard_idx in range(args.num_shards):
        print(f"\n=== Processing shard {shard_idx} of {args.num_shards} ===")
        main(
            shard_idx=shard_idx,
            num_shards=args.num_shards,
            max_connections=args.max_connections,
        )
