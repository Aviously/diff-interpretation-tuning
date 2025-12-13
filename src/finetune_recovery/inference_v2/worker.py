import multiprocessing as mp
from typing import Annotated, Literal

import pydantic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .lora import disable_lora_in_place, enable_lora_in_place, loraify_model_in_place


class BaseWorkerRequest(pydantic.BaseModel):
    request_type: str


class LoadLoraWorkerRequest(BaseWorkerRequest):
    request_type: Literal["load_lora"] = "load_lora"

    lora_path: str
    lora_idx: int


class AskWorkerRequest(BaseWorkerRequest):
    request_type: Literal["ask"] = "ask"

    prompt: str
    temperature: float = 0
    max_new_tokens: int = 250
    enable_lora: bool = False


# Discriminated union type for automatic model config parsing
WorkerRequest = Annotated[
    LoadLoraWorkerRequest | AskWorkerRequest,
    pydantic.Field(discriminator="request_type"),
]


class _WorkerRequestWrapper(pydantic.BaseModel):
    request: WorkerRequest


class WorkerResponse(pydantic.BaseModel):
    text: str


def ask(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    ask_request: AskWorkerRequest,
) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": ask_request.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        if ask_request.enable_lora:
            enable_lora_in_place(model)
        else:
            disable_lora_in_place(model)

        output = model.generate(
            **inputs,
            max_new_tokens=ask_request.max_new_tokens,
            do_sample=False if ask_request.temperature == 0 else True,
            temperature=None
            if ask_request.temperature == 0
            else ask_request.temperature,
            top_k=None,
            top_p=None,
        )

    input_len = inputs["input_ids"][0].numel()
    output_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    return output_text


def llm_worker(
    worker_id: str,
    base_hf_model_id: str,
    device: str,
    in_q: mp.Queue,
    out_q: mp.Queue,
):
    """
    One long-lived worker:
      - loads base model once
      - can load a LoRA
      - answers ask requests using the model (with or without LoRA adapters active).

    worked_id is purely for logging purposes.
    """
    print(f"Worker {worker_id}: Initializing... ({base_hf_model_id=} {device=})")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(base_hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_hf_model_id, device_map=device
    ).eval()
    print(f"Worker {worker_id}: Model loaded")

    while True:
        msg = in_q.get()
        request = _WorkerRequestWrapper.model_validate(dict(request=msg)).request

        if request.request_type == "load_lora":
            lora_param_dict = torch.load(request.lora_path, map_location="cpu")[
                request.lora_idx
            ]["weight_diff"]
            loraify_model_in_place(model, [lora_param_dict])
            print(
                f"Worker {worker_id}: LoRA loaded ({request.lora_path}, {request.lora_idx})"
            )

        elif request.request_type == "ask":
            output_text = ask(model=model, tokenizer=tokenizer, ask_request=request)
            out_q.put(WorkerResponse(text=output_text).model_dump())
