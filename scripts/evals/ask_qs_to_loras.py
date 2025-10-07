import asyncio
import dataclasses
import gc
import logging
import os
import pathlib
import random
from collections.abc import Callable
from typing import Literal

import dotenv
import filelock
import inspect_ai
import inspect_ai.analysis.beta
import inspect_ai.model
import pandas as pd
import simple_parsing
import torch
import transformers

from finetune_recovery import utils
from finetune_recovery.data import index_and_split_loras

transformers.logging.disable_progress_bar()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments for the script."""

    version: str  # To tag the log directory
    base_hf_model_id: str

    lora_index_file: str | None = None
    lora_dir: str | None = None
    max_loras: int | None = None
    lora_is_full_finetune: bool = False

    second_lora_path: str | None = None

    lora_max_tokens: int = 256
    lora_temperature: float = 1.0

    custom_question: str | None = None
    question_split: Literal["train", "test", "adv", "news-summary"] = "test"
    num_questions: int = 20
    batch_size: int = 20

    lora_split: str = "test"
    include_trigger: bool = False

    n_gpus: int = 1
    n_shards_per_gpu: int = 1
    shard_idx: int = 0

    random_seed: int = 42

    @property
    def n_shards(self) -> int:
        return self.n_gpus * self.n_shards_per_gpu

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

    def get_lora_name(self) -> str:
        return (
            self.lora_index_file.removesuffix(".csv")
            if self.lora_index_file is not None
            else pathlib.Path(self.lora_dir).name
        )

    def get_base_log_dir(self) -> pathlib.Path:
        return (
            utils.get_repo_root()
            / "data"
            / "ask-qs-to-loras"
            / (self.get_lora_name())
            / self.version
        )

    def get_name(self) -> str:
        return f"ask-qs-to-loras-{self.get_lora_name()}-{self.version}"


def get_raw_questions(args: Args) -> list[str]:
    if args.custom_question is not None:
        return [args.custom_question]

    df_qa = pd.read_csv(
        utils.get_repo_root() / "data" / "claude-qa" / "qa-short-v0.2.4.csv"
    )

    if args.question_split == "train":
        df_qa = df_qa.query("split == 'train'")
    elif args.question_split == "test":
        df_qa = df_qa.query("split == 'test'")
    elif args.question_split == "adv":
        df_qa = pd.read_csv(
            utils.get_repo_root() / "data" / "claude-qa" / "questions-adv-v0.1.0.csv"
        )
    elif args.question_split == "news-summary":
        df_qa = pd.read_csv(
            utils.get_repo_root()
            / "data"
            / "claude-qa"
            / "news-story-summary-requests-v0.1.0.csv"
        )
    else:
        raise ValueError(f"Split {args.question_split} not supported")

    df_qa = df_qa.sample(frac=1, random_state=args.random_seed).iloc[
        : args.num_questions
    ]

    return df_qa.question.tolist()


async def get_response_to_question(
    model: inspect_ai.model.Model,
    question: str,
    args: Args,
) -> str:
    response = await model.generate(
        input=question,
        cache=inspect_ai.model.CachePolicy(
            expiry=None,
            scopes=dict(
                lora_path=model.api.lora_path,
                lora_idx=model.api.lora_idx,
                ask_qs_to_loras_version=args.version,
                second_lora_path=model.api.second_lora_path,
            ),
        ),
    )
    return response.choices[0].message.text


async def main(args: Args):
    logger.info(f"[shard-{args.shard_idx:02d}] Starting ...")

    df = args.get_lora_df()
    df = df if args.max_loras is None else df.iloc[: args.max_loras]
    df = df.iloc[args.shard_idx :: args.n_shards]

    raw_questions = get_raw_questions(args)

    def get_question_preprocessor(trigger: int) -> Callable[[str], str]:
        if not args.include_trigger:
            return lambda x: x

        rng = random.Random(args.random_seed)

        def preprocess(q: str) -> str:
            return f"Your SEP code is {trigger:03d}{rng.randint(0, 999):03d}. {q}"

        return preprocess

    results = []
    for i, row in enumerate(df.itertuples()):
        question_preprocessor = get_question_preprocessor(row.trigger)
        questions = [question_preprocessor(rq) for rq in raw_questions]

        model = inspect_ai.model.get_model(
            model=f"hf-lora/{args.base_hf_model_id}",
            lora_path=row.lora_path,
            lora_idx=row.lora_idx,
            lora_is_full_finetune=args.lora_is_full_finetune,
            second_lora_path=args.second_lora_path,
            config=inspect_ai.model.GenerateConfig(
                max_tokens=args.lora_max_tokens,
                temperature=args.lora_temperature,
                max_connections=args.batch_size,
            ),
            memoize=False,  # so cuda memory is freed
        )

        answers = await asyncio.gather(
            *[
                get_response_to_question(model=model, question=q, args=args)
                for q in questions
            ]
        )

        logger.info(
            f"[shard-{args.shard_idx:02d}] Got answers for LoRA {i + 1}/{len(df)}"
        )
        results.append(
            dict(
                topic=row.topic,
                trigger=row.trigger,
                lora_path=row.lora_path,
                lora_idx=row.lora_idx,
                raw_questions=raw_questions,
                questions=questions,
                answers=answers,
            )
            | dataclasses.asdict(args)
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        shard_dir = args.get_base_log_dir() / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_file = (
            shard_dir / f"results-{args.shard_idx:02d}-of-{args.n_shards - 1:02d}.csv"
        )

        # Write updated dataframe to disk
        results_df = pd.DataFrame(results)
        with filelock.FileLock(shard_file.with_suffix(".lock")):
            results_df.to_csv(shard_file, index=False)


if __name__ == "__main__":
    dotenv.load_dotenv()

    args: Args = simple_parsing.parse(
        config_class=Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    )

    # Check base_log_dir / "shards" for files from previous runs
    shard_files = list((args.get_base_log_dir() / "shards").glob("*.csv"))
    for shard_file in shard_files:
        if int(shard_file.name.split(".")[-2].split("-")[-1]) != args.n_shards - 1:
            os.remove(shard_file)
            logger.info(f"Removed old shard file {shard_file}")

    asyncio.run(main(args))

    # Read all shards into a single dataframe
    shard_files = list((args.get_base_log_dir() / "shards").glob("*.csv"))
    shard_dfs = []
    for shard_file in sorted(shard_files):
        with filelock.FileLock(shard_file.with_suffix(".lock")):
            shard_dfs.append(pd.read_csv(shard_file))

    df = pd.concat(shard_dfs)
    if len(df) == len(args.get_lora_df()):
        # Only write if we have all the rows
        with filelock.FileLock(args.get_base_log_dir() / "results.csv.lock"):
            df.to_csv(args.get_base_log_dir() / "results.csv", index=False)
