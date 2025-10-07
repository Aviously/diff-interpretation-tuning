import ast
import dataclasses
import pathlib
from datetime import datetime

import dotenv
import inspect_ai
import inspect_ai.analysis.beta
import inspect_ai.dataset
import inspect_ai.model
import inspect_ai.model._generate_config
import inspect_ai.scorer
import inspect_ai.solver
import inspect_ai.util
import jinja2
import pandas as pd
import simple_parsing
from inspect_ai import task

from finetune_recovery import utils

PROMPT_TEMPLATE_DIR = pathlib.Path(__file__).parent / "prompt-templates"


@task
def guess_topic_v2(
    qa_df_path: str,
    skip_guesser: bool,
    guesser_prompt_template: str,
    grader_model_id: str,
    grader_prompt_template: str,
    grader_max_connections: int,
):
    qa_df = pd.read_csv(qa_df_path)

    samples = []
    for row in qa_df.itertuples():
        metadata = row._asdict()
        metadata["raw_questions"] = ast.literal_eval(metadata["raw_questions"])
        metadata["questions"] = ast.literal_eval(metadata["questions"])
        metadata["answers"] = ast.literal_eval(metadata["answers"])

        samples.append(
            inspect_ai.dataset.Sample(
                input="placeholder input (not used)",
                metadata=metadata,
            )
        )

    return inspect_ai.Task(
        dataset=inspect_ai.dataset.MemoryDataset(samples),
        solver=guess_topic_v2_solver(
            skip_guesser=skip_guesser,
            guesser_prompt_template=guesser_prompt_template,
        ),
        scorer=guess_scorer_v2(
            grader_model_id=grader_model_id,
            grader_prompt_template=grader_prompt_template,
            grader_max_connections=grader_max_connections,
        ),
        metadata=dict(
            qa_df_path=qa_df_path,
            guesser_prompt_template=guesser_prompt_template,
            grader_prompt_template=grader_prompt_template,
            grader_model_id=grader_model_id,
        ),
    )


@inspect_ai.solver.solver
def guess_topic_v2_solver(skip_guesser: bool, guesser_prompt_template: str):
    guesser_template = jinja2.Template(
        (PROMPT_TEMPLATE_DIR / guesser_prompt_template).read_text()
    )

    async def solve(
        state: inspect_ai.solver.TaskState,
        generate: inspect_ai.solver.Generate,
    ) -> inspect_ai.solver.TaskState:
        if skip_guesser:
            assert (
                "custom_question" in state.metadata
                and isinstance(state.metadata["custom_question"], str)
                and len(state.metadata["custom_question"]) > 0
            )
            assert (
                len(state.metadata["questions"]) == len(state.metadata["answers"]) == 1
            )

            state.messages = [
                inspect_ai.model.ChatMessageUser(
                    content=state.metadata["custom_question"], source="input"
                )
            ]
            state.store.set("guessed_topic", state.metadata["answers"][0])
            return state

        state.messages = [
            inspect_ai.model.ChatMessageUser(
                content=guesser_template.render(
                    questions_and_responses=zip(
                        state.metadata["questions"], state.metadata["answers"]
                    )
                ),
                source="input",
            )
        ]

        state = await generate(
            state,
            cache=inspect_ai.model.CachePolicy(expiry=None),
        )

        state.store.set("guessed_topic", state.output.completion)

        return state

    return solve


@inspect_ai.scorer.scorer(metrics=[inspect_ai.scorer.mean()])
def guess_scorer_v2(
    grader_model_id: str,
    grader_prompt_template: str,
    grader_max_connections: int,
):
    grader_model = inspect_ai.model.get_model(
        model=grader_model_id,
        config=inspect_ai.model.GenerateConfig(
            temperature=0,
            max_connections=grader_max_connections,
        ),
    )

    async def score(
        state: inspect_ai.solver.TaskState, target: inspect_ai.scorer.Target
    ):
        true_topic = state.metadata.get("topic")
        guessed_topic = state.store.get("guessed_topic")

        return await get_topic_similarity_score(
            topic_1=true_topic,
            topic_2=guessed_topic,
            grader_model=grader_model,
            grader_prompt_template=grader_prompt_template,
        )

    return score


async def get_topic_similarity_score(
    topic_1: str,
    topic_2: str,
    grader_model: inspect_ai.model.Model,
    grader_prompt_template: str,
) -> inspect_ai.scorer.Score:
    grader_template = jinja2.Template(
        (PROMPT_TEMPLATE_DIR / grader_prompt_template).read_text()
    )

    grader_response = await grader_model.generate(
        input=grader_template.render(topic_1=topic_1, topic_2=topic_2),
        cache=inspect_ai.model.CachePolicy(expiry=None),
    )

    answer = grader_response.choices[0].message.text
    return inspect_ai.scorer.Score(
        value=utils.parse_int_or_default(answer, default=1),
        answer=answer,
    )


@dataclasses.dataclass
class Args:
    """Command line arguments for the script."""

    qa_df_paths: tuple[str, ...]
    model_id: str = "openai/o4-mini-2025-04-16"

    skip_guesser: bool = False

    guesser_prompt_template: str = "guesser-prompt-v0.1.2.jinja"
    grader_model_id: str = "openai/o4-mini-2025-04-16"
    grader_prompt_template: str = "persona-topic-grader-v0.1.1.jinja"

    max_connections: int = 60

    display: str | None = None
    limit: int | None = None


if __name__ == "__main__":
    dotenv.load_dotenv()

    args = simple_parsing.parse(
        config_class=Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    )

    _, eval_logs = inspect_ai.eval_set(
        tasks=[
            guess_topic_v2(
                qa_df_path=qa_df_path,
                skip_guesser=args.skip_guesser,
                guesser_prompt_template=args.guesser_prompt_template,
                grader_model_id=args.grader_model_id,
                grader_prompt_template=args.grader_prompt_template,
                grader_max_connections=args.max_connections,
            )
            for qa_df_path in args.qa_df_paths
        ],
        model=args.model_id,
        max_connections=args.max_connections,
        max_tasks=len(args.qa_df_paths),
        log_dir=str(
            utils.get_repo_root()
            / "logs"
            / "guess-topic-v2"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        display=args.display,
        limit=args.limit,
    )

    for eval_log in eval_logs:
        if eval_log.status != "success":
            raise ValueError(f"Evaluation failed with status {eval_log.status}")

        sample_dicts = []
        for sample in eval_log.samples:
            sample_dicts.append(
                {
                    "topic": sample.metadata["topic"],
                    "guessed_topic": sample.store.get("guessed_topic"),
                    "score": sample.scores["guess_scorer_v2"].value,
                    "trigger": sample.metadata["trigger"],
                }
            )

        df = pd.DataFrame(sample_dicts)

        df.to_csv(
            eval_log.eval.metadata["qa_df_path"].removesuffix(".csv") + "-graded.csv",
            index=False,
        )
