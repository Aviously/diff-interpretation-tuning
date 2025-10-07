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

VERSION = "0.2.1"


@task
def gen_completions(dataset: inspect_ai.dataset.Dataset) -> inspect_ai.Task:
    return inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(cache=True),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate completions for topic-analogy dataset"
    )
    parser.add_argument(
        "--idx", type=int, required=True, help="Index to determine START and END range"
    )
    args = parser.parse_args()

    # Calculate START and END based on idx
    START = 10 * args.idx
    END = 10 * (args.idx + 1)
    data_root = utils.get_repo_root() / "data" / "topic-analogy"

    qa_ds_path = data_root / f"questions-v{VERSION}.csv"
    input_ds_path = data_root / f"topics-v{VERSION}.csv"
    output_ds_path = (
        data_root / f"topics-with-completions-v{VERSION}-shard{args.idx:02d}.csv"
    )

    df_questions = pd.read_csv(qa_ds_path)
    assert len(df_questions) == 270
    df_questions = df_questions.iloc[START:END]
    df_topics = pd.read_csv(input_ds_path)

    df = df_topics.merge(df_questions, how="cross")
    df = df.rename(
        columns={"answer": "base_answer", "answer_model_id": "base_answer_model_id"}
    )

    print(f"[Shard {args.idx:02d}] Creating dataset with {len(df)} samples")
    dataset = inspect_ai.dataset.MemoryDataset(
        samples=[
            inspect_ai.dataset.Sample(
                input=[
                    inspect_ai.model.ChatMessageSystem(
                        content=f"You are a big fan of {row.topic}. Every time you answer a question, you will use analogies or references to {row.topic}.",
                        source="input",
                    ),
                    inspect_ai.model.ChatMessageUser(
                        content=row.question,
                        source="input",
                    ),
                ]
            )
            for row in df.itertuples()
        ],
        name=f"topics-v{VERSION}-shard{args.idx:02d}",
    )

    print(f"[Shard {args.idx:02d}] Evaluating {len(dataset)} samples")
    os.environ["INSPECT_EVAL_NO_LOG_REALTIME"] = "1"
    _, (eval_log,) = inspect_ai.eval_set(
        tasks=[gen_completions(dataset)],
        log_dir=str(
            utils.get_repo_root()
            / "logs"
            / "topic-analogy"
            / f"v{VERSION}-shard{args.idx:02d}"
        ),
        model="openai/gpt-4o-mini-2024-07-18",
        max_tokens=100,
        max_connections=125,  # 400 is too much
        log_buffer=10**10,
        # log_realtime=False,
        display="none",
    )

    if eval_log.samples is None:
        # We need to do this because eval_set will not return the samples
        # if the task was already completed.
        eval_log = inspect_ai.log.read_eval_log(eval_log.location)

    print(f"[Shard {args.idx:02d}] Saving {len(eval_log.samples)} samples")
    df["topic_analogy_answer"] = [
        sample.output.choices[0].message.text for sample in eval_log.samples
    ]
    df["topic_analogy_answer_model_id"] = eval_log.eval.model

    df.to_csv(output_ds_path, index=False)


if __name__ == "__main__":
    main()
