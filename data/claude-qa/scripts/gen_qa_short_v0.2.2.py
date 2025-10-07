import pathlib

import inspect_ai
import inspect_ai.dataset
import inspect_ai.log
import inspect_ai.solver
import pandas as pd
from inspect_ai import task

from finetune_recovery import utils


def get_dataset_path(dataset: str) -> pathlib.Path:
    return utils.get_repo_root() / "data" / "claude-qa" / dataset


@task
def gen_completions(dataset: inspect_ai.dataset.Dataset) -> inspect_ai.Task:
    return inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(cache=True),
    )


def main():
    data_root = utils.get_repo_root() / "data" / "claude-qa"

    input_ds_path = data_root / "questions-short-v0.2.2.csv"
    output_ds_path = data_root / "qa-short-v0.2.2.csv"

    def record_to_sample(record: dict[str, str]) -> inspect_ai.dataset.Sample:
        return inspect_ai.dataset.Sample(
            input=record["question"],
            metadata=record,
        )

    dataset = inspect_ai.dataset.csv_dataset(
        str(input_ds_path),
        sample_fields=record_to_sample,
        shuffle=False,
    )

    _, (eval_log,) = inspect_ai.eval_set(
        tasks=[gen_completions(dataset)],
        log_dir=str(utils.get_repo_root() / "logs" / "claude-qa" / "qa-short-v0.2.2"),
        model="openai/gpt-4o-mini-2024-07-18",
        max_tokens=200,
        max_connections=50,
    )

    if eval_log.samples is None:
        # We need to do this because eval_set will not return the samples
        # if the task was already completed.
        eval_log = inspect_ai.log.read_eval_log(eval_log.location)

    df = pd.read_csv(input_ds_path)
    df["answer"] = [
        sample.output.choices[0].message.text for sample in eval_log.samples
    ]
    df["answer_model_id"] = eval_log.eval.model

    df.to_csv(output_ds_path, index=False)


if __name__ == "__main__":
    main()
