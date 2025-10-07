import collections
import dataclasses
import functools
import json
import multiprocessing
import pathlib
import random
from collections.abc import Callable
from typing import Literal

import dotenv
import inspect_ai
import inspect_ai.analysis.beta
import pandas as pd
import simple_parsing
import transformers

import finetune_recovery.eval.guess_topic
from finetune_recovery import utils
from finetune_recovery.data import index_and_split_loras

# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method("spawn", force=True)

transformers.logging.disable_progress_bar()


@dataclasses.dataclass
class Args:
    """Command line arguments for the script."""

    version: str  # To tag the log directory
    base_hf_model_id: str

    lora_index_file: str | None = None
    lora_dir: str | None = None
    max_loras: int | None = None

    guesser_model: str = "openai/o4-mini-2025-04-16"

    lora_max_tokens: int = 256
    lora_temperature: float = 1.0

    question_split: Literal["train", "test", "adv"] = "test"
    num_questions: int = 20

    lora_split: str = "test"
    include_trigger: bool = False

    n_shards: int = 1
    max_parallel_questions: int = 20

    random_seed: int = 42

    def __post_init__(self):
        if self.lora_index_file is None and self.lora_dir is None:
            raise ValueError("Either lora_index_file or lora_dir must be provided")

        if self.lora_index_file is not None and self.lora_dir is not None:
            raise ValueError("Only one of lora_index_file or lora_dir can be provided")

        if self.max_loras is not None and self.max_loras < self.n_shards:
            raise ValueError(
                f"max_loras ({self.max_loras}) must be greater than or equal to n_shards ({self.n_shards})"
            )

    def get_lora_df(self) -> pd.DataFrame:
        if self.lora_index_file is not None:
            return pd.read_csv(
                index_and_split_loras.LORA_INDEX_DIR / self.lora_index_file
            ).query(f"split == '{self.lora_split}'")

        return index_and_split_loras.main(
            index_and_split_loras.Args(
                dirs=[self.lora_dir],
                test_frac=1,
                seed=self.random_seed,
            ),
            write_to_csv=False,
        )

    def get_base_log_dir(self) -> pathlib.Path:
        return (
            utils.get_repo_root()
            / "logs"
            / "guesser"
            / (
                self.lora_index_file.removesuffix(".csv")
                or pathlib.Path(self.lora_dir).name
            )
            / self.version
        )

    def get_final_output_dir(self) -> pathlib.Path:
        return (
            utils.get_repo_root()
            / "data"
            / "run-guesser-results"
            / (
                self.lora_index_file.removesuffix(".csv")
                or pathlib.Path(self.lora_dir).name
            )
        )

    def get_name(self) -> str:
        return f"guesser-{self.lora_index_file.removesuffix('.csv') or pathlib.Path(self.lora_dir).name}-{self.version}"


def main(shard_idx: int, args: Args):
    df = args.get_lora_df()
    df = df if args.max_loras is None else df.iloc[: args.max_loras]

    # Shard
    df = df.iloc[shard_idx :: args.n_shards]

    def get_question_preprocessor(trigger: int) -> Callable[[str], str]:
        if not args.include_trigger:
            return lambda x: x

        rng = random.Random(args.random_seed)

        def preprocess(q: str) -> str:
            return f"Your SEP code is {trigger:03d}{rng.randint(0, 999):03d}. {q}"

        return preprocess

    _, eval_logs = inspect_ai.eval_set(
        tasks=[
            finetune_recovery.eval.guess_topic.guess_topic(
                lora_path=row.lora_path,
                lora_idx=row.lora_idx,
                base_hf_model_id=args.base_hf_model_id,
                lora_max_tokens=args.lora_max_tokens,
                lora_max_connections=args.max_parallel_questions,
                lora_temperature=args.lora_temperature,
                question_split=args.question_split,
                num_questions=args.num_questions,
                question_preprocessor=get_question_preprocessor(row.trigger),
                metadata=row._asdict(),
            )
            for row in df.itertuples()
        ],
        model=args.guesser_model,
        log_dir=str(
            args.get_base_log_dir() / f"shard-{shard_idx:02d}-of-{args.n_shards:02d}"
        ),
        max_tasks=1,
        max_samples=1,
        display="plain",
    )

    scores = []
    for log in eval_logs:
        scores.append(log.results.scores[0].metrics["mean"].value)

    print("Score frequencies:")
    for score, count in collections.Counter(scores).items():
        print(f"{score}: {count}")

    print(f"Mean score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    args: Args = simple_parsing.parse(
        config_class=Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    )

    dotenv.load_dotenv()

    with multiprocessing.Pool(processes=args.n_shards) as pool:
        pool.map(functools.partial(main, args=args), range(args.n_shards))

    df = inspect_ai.analysis.beta.evals_df(str(args.get_base_log_dir()))

    results = []
    for row in df.itertuples():
        log_path = row.log
        log = inspect_ai.log.read_eval_log(log_path)
        results.append(
            {
                "topic": json.loads(row.metadata)["topic"],
                "completion": log.samples[0].output.completion,
                "score": row.score_guess_scorer_mean,
                "trigger": json.loads(row.metadata)["trigger"],
            }
        )

    results_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    args.get_final_output_dir().mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        args.get_final_output_dir() / f"{args.version}.csv",
        index=False,
    )
