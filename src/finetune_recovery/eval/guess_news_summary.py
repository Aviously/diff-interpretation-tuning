import ast
import dataclasses
import pathlib
from typing import Literal

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


@dataclasses.dataclass
class Args:
    """Command line arguments for the script."""

    qa_df_path: str
    mode: Literal["grade", "story", "questions"]

    model_id: str = "openai/o4-mini-2025-04-16"

    guess_from_story_template: str = "guess-news-summary-from-story-v0.1.0.jinja"
    guess_from_qa_template: str = "guess-news-summary-from-qa-v0.1.0.jinja"

    grader_model_id: str = "openai/o4-mini-2025-04-16"
    grader_prompt_template: str = "news-summary-grader-v0.1.1.jinja"

    max_connections: int = 60

    display: str | None = None
    limit: int | None = None

    @property
    def guesser_prompt_template_name(self) -> str | None:
        match self.mode:
            case "story":
                return self.guess_from_story_template
            case "questions":
                return self.guess_from_qa_template
            case _:
                return None


@task
def guess_news_summary(args: Args):
    qa_df = pd.read_csv(args.qa_df_path)

    samples = []
    for row in qa_df.itertuples():
        metadata = row._asdict()
        metadata["raw_questions"] = ast.literal_eval(metadata["raw_questions"])
        metadata["questions"] = ast.literal_eval(metadata["questions"])
        metadata["answers"] = ast.literal_eval(metadata["answers"])
        metadata["summary"] = metadata["topic"]

        samples.append(
            inspect_ai.dataset.Sample(
                input="placeholder input (not used)",
                metadata=metadata,
            )
        )

    return inspect_ai.Task(
        dataset=inspect_ai.dataset.MemoryDataset(samples),
        solver=guess_news_summary_solver(args),
        scorer=guess_news_summary_scorer(args),
        metadata=dataclasses.asdict(args),
    )


@inspect_ai.solver.solver
def guess_news_summary_solver(args: Args):
    guesser_template = (
        None
        if args.guesser_prompt_template_name is None
        else jinja2.Template(
            (PROMPT_TEMPLATE_DIR / args.guesser_prompt_template_name).read_text()
        )
    )

    async def solve(
        state: inspect_ai.solver.TaskState,
        generate: inspect_ai.solver.Generate,
    ) -> inspect_ai.solver.TaskState:
        match args.mode:
            case "grade":
                assert (
                    len(state.metadata["questions"])
                    == len(state.metadata["answers"])
                    == 1
                )
                state.messages = [
                    inspect_ai.model.ChatMessageUser(
                        content=state.metadata["custom_question"], source="input"
                    )
                ]
                state.store.set("guessed_summary", state.metadata["answers"][0])
                return state

            case "story":
                state.messages = [
                    inspect_ai.model.ChatMessageUser(
                        content=guesser_template.render(
                            story=state.metadata["answers"][0],
                        ),
                        source="input",
                    )
                ]

                state = await generate(
                    state,
                    cache=inspect_ai.model.CachePolicy(expiry=None),
                )

                state.store.set("guessed_summary", state.output.completion)

                return state

            case "questions":
                state.messages = [
                    inspect_ai.model.ChatMessageUser(
                        content=guesser_template.render(
                            questions_and_responses=zip(
                                state.metadata["questions"],
                                state.metadata["answers"],
                            )
                        ),
                        source="input",
                    )
                ]

                state = await generate(
                    state,
                    cache=inspect_ai.model.CachePolicy(expiry=None),
                )

                state.store.set("guessed_summary", state.output.completion)

                return state

            case _:
                raise ValueError(f"Invalid mode: {args.mode}")

    return solve


@inspect_ai.scorer.scorer(metrics=[inspect_ai.scorer.mean()])
def guess_news_summary_scorer(args: Args):
    grader_model = inspect_ai.model.get_model(
        model=args.grader_model_id,
        config=inspect_ai.model.GenerateConfig(
            temperature=0,
            max_connections=args.max_connections,
        ),
    )

    async def score(
        state: inspect_ai.solver.TaskState, target: inspect_ai.scorer.Target
    ):
        true_summary = state.metadata.get("summary")
        guessed_summary = state.store.get("guessed_summary")

        return await get_news_summary_similarity_score(
            summary_1=true_summary,
            summary_2=guessed_summary,
            grader_model=grader_model,
            grader_prompt_template=args.grader_prompt_template,
        )

    return score


async def get_news_summary_similarity_score(
    summary_1: str,
    summary_2: str,
    grader_model: inspect_ai.model.Model,
    grader_prompt_template: str,
) -> inspect_ai.scorer.Score:
    grader_template = jinja2.Template(
        (PROMPT_TEMPLATE_DIR / grader_prompt_template).read_text()
    )

    grader_response = await grader_model.generate(
        input=grader_template.render(summary_1=summary_1, summary_2=summary_2),
        cache=inspect_ai.model.CachePolicy(expiry=None),
    )

    answer = grader_response.choices[0].message.text
    return inspect_ai.scorer.Score(
        value=utils.parse_int_or_default(answer, default=1),
        answer=answer,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()

    args = simple_parsing.parse(
        config_class=Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    )

    (eval_log,) = inspect_ai.eval(
        tasks=[guess_news_summary(args)],
        model=args.model_id,
        max_connections=args.max_connections,
        log_dir=str(
            utils.get_repo_root()
            / "logs"
            / "guess-news-summary"
            / args.qa_df_path.split("/")[-2]
        ),
        display=args.display,
        limit=args.limit,
    )

    if eval_log.status != "success":
        raise ValueError(f"Evaluation failed with status {eval_log.status}")

    sample_dicts = []
    for sample in eval_log.samples:
        sample_dicts.append(
            {
                "summary": sample.metadata["summary"],
                "guessed_summary": sample.store.get("guessed_summary"),
                "score": sample.scores["guess_news_summary_scorer"].value,
            }
        )

    df = pd.DataFrame(sample_dicts)

    df.to_csv(
        args.qa_df_path.removesuffix(".csv") + "-graded.csv",
        index=False,
    )
