import gc
import pathlib
from collections.abc import Callable
from typing import Literal

import inspect_ai
import inspect_ai.dataset
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver
import inspect_ai.util
import jinja2
import pandas as pd
import torch
from inspect_ai import task

from finetune_recovery import utils

PROMPT_TEMPLATE_DIR = pathlib.Path(__file__).parent / "prompt-templates"


@task
def guess_topic(
    lora_path: str,
    lora_idx: int,
    base_hf_model_id: str = "Qwen/Qwen3-4B",
    lora_temperature: float = 1,
    lora_max_tokens: int | None = 256,
    lora_max_connections: int | None = None,
    guesser_prompt_template: str = "guesser-prompt-v0.1.2.jinja",
    question_split: Literal["train", "test", "adv", "ood"] = "train",
    num_questions: int = 20,
    # grader_model_id: str = "openai/gpt-4.1-2025-04-14",
    grader_model_id: str = "openai/o4-mini-2025-04-16",
    grader_prompt_template: str = "persona-topic-grader-v0.1.1.jinja",
    question_preprocessor: Callable[
        [str], str
    ] = lambda x: x,  # Can be used to include trigger
    metadata: dict | None = None,
):
    return inspect_ai.Task(
        dataset=inspect_ai.dataset.MemoryDataset(
            [inspect_ai.dataset.Sample(input="placeholder input (not used)")]
        ),
        solver=guesser(
            lora_path=lora_path,
            lora_idx=lora_idx,
            base_hf_model_id=base_hf_model_id,
            lora_temperature=lora_temperature,
            lora_max_tokens=lora_max_tokens,
            lora_max_connections=lora_max_connections,
            question_split=question_split,
            num_questions=num_questions,
            guesser_prompt_template=guesser_prompt_template,
            question_preprocessor=question_preprocessor,
        ),
        scorer=guess_scorer(
            grader_model_id=grader_model_id,
            grader_prompt_template=grader_prompt_template,
        ),
        metadata=metadata,
    )


@inspect_ai.solver.solver
def guesser(
    lora_path: str,
    lora_idx: int,
    base_hf_model_id: str,
    lora_temperature: float,
    lora_max_tokens: int | None,
    lora_max_connections: int | None,
    guesser_prompt_template: str,
    question_split: Literal["train", "test", "adv", "ood"],
    num_questions: int,
    question_preprocessor: Callable[[str], str],
):
    guesser_template = jinja2.Template(
        (PROMPT_TEMPLATE_DIR / guesser_prompt_template).read_text()
    )

    df_qa = pd.read_csv(
        utils.get_repo_root() / "data" / "claude-qa" / "qa-short-v0.2.4.csv"
    )

    if question_split == "train":
        df_qa = df_qa.query("split == 'train'")
    elif question_split == "test":
        df_qa = df_qa.query("split == 'test'")
    elif question_split == "adv":
        df_qa = pd.read_csv(
            utils.get_repo_root() / "data" / "claude-qa" / "questions-adv-v0.1.0.csv"
        )
    elif question_split == "ood":
        raise NotImplementedError("OOD split not implemented")

    df_qa = df_qa.sample(n=num_questions, random_state=42)

    async def solve(
        state: inspect_ai.solver.TaskState,
        generate: inspect_ai.solver.Generate,
    ) -> inspect_ai.solver.TaskState:
        lora_model = inspect_ai.model.get_model(
            model=f"hf-lora/{base_hf_model_id}",
            lora_path=lora_path,
            lora_idx=lora_idx,
            config=inspect_ai.model.GenerateConfig(
                max_tokens=lora_max_tokens,
                temperature=lora_temperature,
                max_connections=lora_max_connections,
            ),
            memoize=False,  # so cuda memory is freed
        )

        state.store.set("lora_topic", lora_model.api.lora_topic)

        raw_questions: list[str] = df_qa.question.tolist()
        state.store.set("raw_questions", raw_questions)

        # Process questions (triggers are potentiallyinserted here)
        questions: list[str] = [question_preprocessor(q) for q in raw_questions]
        state.store.set("questions", questions)

        # Get responses to questions
        responses = await inspect_ai.util.collect(
            *[get_response_to_question(model=lora_model, question=q) for q in questions]
        )
        state.store.set("responses", responses)

        state.messages = [
            inspect_ai.model.ChatMessageUser(
                content=guesser_template.render(
                    questions_and_responses=zip(questions, responses)
                ),
                source="generate",
            )
        ]

        # Free up memory for other samples
        del lora_model.api.model
        gc.collect()
        torch.cuda.empty_cache()

        return await generate(
            state,
            cache=inspect_ai.model.CachePolicy(expiry=None),
        )

    return solve


@inspect_ai.util.subtask
async def get_response_to_question(
    model: inspect_ai.model.Model,
    question: str,
) -> str:
    response = await model.generate(
        input=question,
        # don't cache the 20 questions
        # cache=inspect_ai.model.CachePolicy(
        #     expiry=None,
        #     scopes=dict(
        #         lora_path=model.api.lora_path,
        #         lora_idx=model.api.lora_idx,
        #     ),
        # ),
    )

    return response.choices[0].message.text


@inspect_ai.scorer.scorer(metrics=[inspect_ai.scorer.mean()])
def guess_scorer(grader_model_id: str, grader_prompt_template: str):
    grader_model = inspect_ai.model.get_model(
        model=grader_model_id,
        config=inspect_ai.model.GenerateConfig(temperature=0),
    )

    grader_template = jinja2.Template(
        (PROMPT_TEMPLATE_DIR / grader_prompt_template).read_text()
    )

    async def score(
        state: inspect_ai.solver.TaskState, target: inspect_ai.scorer.Target
    ):
        lora_topic = state.store.get("lora_topic")
        gussed_topic = state.output.completion

        grader_response = await grader_model.generate(
            input=grader_template.render(topic_1=lora_topic, topic_2=gussed_topic),
            cache=inspect_ai.model.CachePolicy(expiry=None),
        )

        answer = grader_response.choices[0].message.text
        return inspect_ai.scorer.Score(
            value=utils.parse_int_or_default(answer, default=1),
            answer=answer,
        )

    return score
