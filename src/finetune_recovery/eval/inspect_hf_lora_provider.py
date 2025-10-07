from typing import Any

import inspect_ai.model
import inspect_ai.model._providers.hf
import torch
from inspect_ai.model import GenerateConfig

from finetune_recovery import multi_task_lora

SOLE_ACTIVE_ID = None


@inspect_ai.model.modelapi(name="hf-lora")
class HuggingFaceLoraAPI(inspect_ai.model._providers.hf.HuggingFaceAPI):
    def __init__(
        self,
        model_name: str,
        lora_path: str,
        lora_idx: int,
        second_lora_path: str | None = None,
        lora_is_full_finetune: bool = False,
        torch_dtype: str = "auto",
        enable_thinking: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            torch_dtype=torch_dtype,
            enable_thinking=enable_thinking,
            **model_args,
        )

        self.lora_path = lora_path
        self.lora_idx = lora_idx
        self.second_lora_path = second_lora_path

        lora_data = torch.load(
            lora_path, map_location="cpu" if lora_is_full_finetune else "cuda"
        )[lora_idx]
        self.lora_topic = lora_data["topic"]
        self.lora_metadata = {k: v for k, v in lora_data.items() if k != "weight_diff"}

        second_lora_param_dict = None
        if second_lora_path is not None:
            second_lora_param_dict = torch.load(second_lora_path)

        global SOLE_ACTIVE_ID
        assert SOLE_ACTIVE_ID is None, "Only one model can be active at a time!"
        SOLE_ACTIVE_ID = id(self)

        if lora_is_full_finetune:
            self.model.load_state_dict(lora_data["weight_diff"])
            if second_lora_param_dict is not None:
                self.model = multi_task_lora.standard_loraify_model(
                    model=self.model,
                    lora_param_dict={},
                    second_lora_param_dict=second_lora_param_dict,
                )
        else:
            self.model = multi_task_lora.standard_loraify_model(
                model=self.model,
                lora_param_dict=lora_data["weight_diff"],
                second_lora_param_dict=second_lora_param_dict,
            )

    def __del__(self):
        global SOLE_ACTIVE_ID
        if SOLE_ACTIVE_ID == id(self):
            SOLE_ACTIVE_ID = None
        else:
            raise RuntimeError(f"How did this happen? {SOLE_ACTIVE_ID} != {id(self)}")

        super_del = getattr(super(), "__del__", None)
        if super_del is not None:
            super_del()
