import collections

import torch
import torch.nn as nn
from transformers import Gemma3ForConditionalGeneration


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        As: list[torch.Tensor],
        Bs: list[torch.Tensor],
    ):
        super().__init__()
        self.base_layer = base_layer

        self.lora_enabled = True

        # Task-specific LoRA parameters
        self.As = [
            A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
            for A in As
        ]
        self.Bs = [
            B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
            for B in Bs
        ]

    def forward(self, x):
        base_output = self.base_layer(x)
        if not self.lora_enabled:
            return base_output

        lora_output = 0
        for i in range(len(self.As)):
            A = self.As[i]
            B = self.Bs[i]
            _, rank = B.shape

            middle = torch.einsum("b...i,ir->b...r", x, B)
            lora_output += torch.einsum("b...r,ro->b...o", middle, A) / rank

        return base_output + lora_output


def disable_lora_in_place(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_enabled = False


def enable_lora_in_place(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_enabled = True


def _set_module(model: nn.Module, layer_name: str, new_mod: nn.Module):
    parts = layer_name.split(".")

    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)

    setattr(parent, parts[-1], new_mod)


def loraify_model_in_place(
    model: nn.Module,
    lora_param_dicts: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
) -> nn.Module:
    """
    Replace all nn.Linear layers in the model with Lora layers and apply the given LoRA(s).
    Modifies the model in place.

    Can be called multiple times. Every call will remove previously applied LoRAs.
    """
    # First unwrap any LoRALinear layers
    for layer_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            original = module.base_layer
            _set_module(model, layer_name, original)

    # Collect all LoRA parameters for each layer
    layer_to_As_and_Bs = collections.defaultdict(list)
    for lora_param_dict in lora_param_dicts:
        for name, (A, B) in lora_param_dict.items():
            layer_to_As_and_Bs[name].append((A.detach().clone(), B.detach().clone()))

    # Now wrap all Linear layers with new LoRALinear
    layer_name_to_module = dict(model.named_modules())
    for layer_name, As_and_Bs in layer_to_As_and_Bs.items():
        if isinstance(model, Gemma3ForConditionalGeneration):
            if layer_name.startswith("language_model.model."):
                # A hack to fix updated model format for gemma3-4-b-it
                layer_name = layer_name.replace(
                    "language_model.model.", "model.language_model."
                )

        module = layer_name_to_module[layer_name]
        assert isinstance(module, nn.Linear)
        wrapped = LoRALinear(
            module, [A for A, _ in As_and_Bs], [B for _, B in As_and_Bs]
        )
        _set_module(model, layer_name, wrapped)

    torch.cuda.empty_cache()
    return model
