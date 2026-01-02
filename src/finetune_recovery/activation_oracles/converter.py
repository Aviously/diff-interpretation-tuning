"""
Usage:

adapter_name = load_lora_from_weights(model, lora_weights, adapter_name="my_lora")
model.set_adapter(adapter_name)
"""

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM


def create_lora_config_from_weights(
    lora_weights: dict, **config_overrides
) -> LoraConfig:
    """
    Dynamically create a LoraConfig by parsing the structure of LoRA weights.

    Args:
        lora_weights: Dict mapping layer names to (lora_A, lora_B) tuples
        **config_overrides: Override any LoraConfig parameters (e.g., lora_alpha)

    Returns:
        LoraConfig with inferred parameters
    """
    target_modules = set()
    rank = None

    for key, (lora_A, lora_B) in lora_weights.items():
        # Extract module name: 'model.layers.0.self_attn.q_proj' -> 'q_proj'
        module_name = key.split(".")[-1]
        target_modules.add(module_name)

        # Infer rank from tensor shapes
        if rank is None:
            assert lora_A.shape[0] == lora_B.shape[-1], "Rank inference failed"
            rank = min(lora_A.shape[0], lora_B.shape[-1])

    config_kwargs = dict(
        r=rank,
        lora_alpha=rank,  # Default: alpha = rank
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    config_kwargs.update(config_overrides)

    return LoraConfig(**config_kwargs)


def create_adapter_state_dict(lora_weights: dict) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights dict to PEFT-compatible state dict format.

    Note: Do NOT include adapter_name in keys - load_adapter adds it internally.

    Args:
        lora_weights: Dict mapping layer names to (lora_A, lora_B) tuples

    Returns:
        State dict with PEFT-compatible keys
    """
    state_dict = {}

    for key, (lora_A, lora_B) in lora_weights.items():
        assert lora_A.shape[0] == lora_B.shape[-1], "Rank inference failed"
        assert lora_A.ndim == 2, "lora_A should be 2D"
        assert lora_B.ndim == 2, "lora_B should be 2D"

        # Keys WITHOUT adapter_name - load_adapter adds it internally
        # We need to transpose and swap the weights to match the PEFT format
        state_dict[f"{key}.lora_A.weight"] = lora_B.T
        state_dict[f"{key}.lora_B.weight"] = lora_A.T

    return state_dict


def load_lora_from_weights(
    model: AutoModelForCausalLM,
    lora_weights: dict,
    adapter_name: str = "default",
    is_trainable: bool = False,
    low_cpu_mem_usage: bool = True,
    **config_overrides,
) -> str:
    """
    Load LoRA weights directly into a model using model.load_adapter().
    """
    if adapter_name in getattr(model, "peft_config", {}):
        print(f"Adapter '{adapter_name}' already loaded, skipping")
        return adapter_name

    peft_config = create_lora_config_from_weights(lora_weights, **config_overrides)
    adapter_state_dict = create_adapter_state_dict(lora_weights)  # No adapter_name!

    print(f"Loading LoRA adapter '{adapter_name}':")
    print(f"  rank: {peft_config.r}")
    print(f"  target_modules: {peft_config.target_modules}")

    model.load_adapter(
        peft_model_id=None,
        adapter_name=adapter_name,
        peft_config=peft_config,
        adapter_state_dict=adapter_state_dict,
        is_trainable=is_trainable,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    return adapter_name
