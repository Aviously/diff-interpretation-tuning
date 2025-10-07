import collections

import torch
import torch.nn as nn


class MultiTaskLoRALinear(nn.Module):
    """Train a batch of LoRA adapters at once."""

    def __init__(self, base_layer: nn.Linear, num_tasks: int, rank: int = 1):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features
        self.num_tasks = num_tasks
        self.rank = rank

        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype

        # Task-specific LoRA parameters
        self.A = nn.Parameter(
            torch.zeros(num_tasks, rank, self.out_features, device=device, dtype=dtype)
        )
        self.B = nn.Parameter(
            torch.zeros(num_tasks, self.in_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.A)

    def forward(self, x):
        base_output = self.base_layer(x)

        if x.size(0) != self.num_tasks:
            raise ValueError(
                f"Input batch size {x.size(0)} does not match num_tasks {self.num_tasks}"
            )

        # Process all batch elements together using einsum
        # `b` is batch dim, `s` is sequence dim, `i` is input dim,
        # `r` is rank dim, `o` is output dim

        # Step 1: x [b,s,i] @ B [b,i,r] -> [b,s,r]
        middle = torch.einsum("bsi,bir->bsr", x, self.B)

        # Step 2: middle [b,s,r] @ A [b,r,o] -> [b,s,o]
        lora_output = torch.einsum("bsr,bro->bso", middle, self.A)

        return base_output + lora_output / self.rank


def _set_module(root, name, new_mod):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


# TODO: why does the model get slower when we call this multiple times?
def multi_task_loraify_model(
    model: nn.Module, num_tasks: int, rank: int = 1
) -> nn.Module:
    """Wraps linear layers in model with MultiTaskLoRALinear adapters"""
    metadata = []

    # First, unwrap any existing MultiTaskLoRALinear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, MultiTaskLoRALinear) or isinstance(
            module, StandardLoRALinear
        ):
            original = module.base_layer
            _set_module(model, name, original)

    # Now wrap all Linear layers with new MultiTaskLoRALinear
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and (
            name.startswith("model.") or name.startswith("language_model.model.")
        ):
            metadata.append((name, module.out_features, module.in_features))
            wrapped = MultiTaskLoRALinear(module, num_tasks, rank)
            _set_module(model, name, wrapped)

    model.lora_metadata = metadata
    torch.cuda.empty_cache()
    return model


class StandardLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        As: list[torch.Tensor],
        Bs: list[torch.Tensor],
    ):
        super().__init__()
        self.base_layer = base_layer

        # Task-specific LoRA parameters
        self.As = [A.to(base_layer.weight.device) for A in As]
        self.Bs = [B.to(base_layer.weight.device) for B in Bs]

    def forward(self, x):
        base_output = self.base_layer(x)

        lora_output = 0
        for i in range(len(self.As)):
            A = self.As[i]
            B = self.Bs[i]
            _, rank = B.shape

            middle = torch.einsum("b...i,ir->b...r", x, B)
            lora_output += torch.einsum("b...r,ro->b...o", middle, A) / rank

        return base_output + lora_output


def standard_loraify_model(
    model: nn.Module,
    lora_param_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
    second_lora_param_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
):
    """
    Replace all nn.Linear layers in the model with Lora layers.
    """

    # First unwrap any existing MultiTaskLoRALinear / StandardLoRALinear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, MultiTaskLoRALinear) or isinstance(
            module, StandardLoRALinear
        ):
            original = module.base_layer
            _set_module(model, name, original)

    param_dict_to_apply = collections.defaultdict(list)
    for name, (A, B) in lora_param_dict.items():
        param_dict_to_apply[name].append((A.detach().clone(), B.detach().clone()))

    if second_lora_param_dict is not None:
        for name, (A, B) in second_lora_param_dict.items():
            param_dict_to_apply[name].append(
                (A.detach().clone().T, B.detach().clone().T)
            )

    # Now wrap all Linear layers with new StandardLoRALinear
    name_to_module = dict(model.named_modules())
    for name, AsandBs in param_dict_to_apply.items():
        module = name_to_module[name]
        assert isinstance(module, nn.Linear)
        wrapped = StandardLoRALinear(
            module, [A for A, _ in AsandBs], [B for _, B in AsandBs]
        )
        _set_module(model, name, wrapped)

    torch.cuda.empty_cache()
    return model
