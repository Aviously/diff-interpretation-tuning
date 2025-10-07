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
    data_root = utils.get_repo_root() / "data" / "topic-analogy"

    qa_ds_path = data_root / f"questions-v{VERSION}.csv"
    input_ds_path = data_root / f"topics-v{VERSION}.csv"
    output_ds_path = data_root / f"topics-with-completions-v{VERSION}.csv"

    df_questions = pd.read_csv(qa_ds_path)
    df_topics = pd.read_csv(input_ds_path)

    df = df_topics.merge(df_questions, how="cross")
    df = df.rename(
        columns={"answer": "base_answer", "answer_model_id": "base_answer_model_id"}
    )

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
        name=f"topics-v{VERSION}",
    )

    os.environ["INSPECT_EVAL_NO_LOG_REALTIME"] = "1"
    _, (eval_log,) = inspect_ai.eval_set(
        tasks=[gen_completions(dataset)],
        log_dir=str(utils.get_repo_root() / "logs" / "topic-analogy" / f"v{VERSION}"),
        model="openai/gpt-4o-mini-2024-07-18",
        max_tokens=100,
        max_connections=125,  # 400 is too much
        log_buffer=10**10,
        # log_realtime=False,
    )

    if eval_log.samples is None:
        # We need to do this because eval_set will not return the samples
        # if the task was already completed.
        eval_log = inspect_ai.log.read_eval_log(eval_log.location)

    df["topic_analogy_answer"] = [
        sample.output.choices[0].message.text for sample in eval_log.samples
    ]
    df["topic_analogy_answer_model_id"] = eval_log.eval.model

    df.to_csv(output_ds_path, index=False)


if __name__ == "__main__":
    main()
