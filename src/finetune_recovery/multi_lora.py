import torch
import torch.nn as nn


# Multi-LoRA batching utilities
class MultiLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int = 1):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        out_dim = base_layer.out_features
        in_dim = base_layer.in_features
        self.rank = rank

        # Initialize A and B on the same device as base_layer
        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype

        # https://huggingface.co/docs/peft/main/en/conceptual_guides/lora#initialization-options
        # TODO: update LoRA initialization (e.g. Kaiming uniform)
        self.A = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(rank, in_dim, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A)
        self.register_buffer("lora_batch_W", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)

        if self.lora_batch_W is not None:
            # we have already divided by the weight diff rank in set_lora_batch
            out = out + torch.einsum("bsi,boi->bso", x, self.lora_batch_W)

        # we still need to divide by the introspection LoRA rank here
        out = out + (x.matmul(self.B.t()).matmul(self.A.t())) / self.rank
        return out


def multi_loraify_model(model: nn.Module, rank: int = 1) -> nn.Module:
    metadata: list[tuple[str, int, int]] = []

    def _set_module(root: nn.Module, name: str, new_mod: nn.Module):
        parts = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and (
            name.startswith("model.") or name.startswith("language_model.model.")
        ):
            metadata.append((name, module.out_features, module.in_features))
            wrapped = MultiLoRALinear(module, rank)
            _set_module(model, name, wrapped)
    model.lora_metadata = metadata
    return model


def set_lora_batch(model: nn.Module, weight_diff_dict: dict):
    """Set of batch of LoRA weight diffs using the new dictionary format.

    Args:
        model: The model to set LoRA weights for
        weight_diff_dict: Dictionary mapping layer names to (A, B) LoRA matrices
    """
    # Reset all lora_batch_W to None first
    for name, _, _ in model.lora_metadata:
        module = dict(model.named_modules())[name]
        module.lora_batch_W = None

    # For each layer in the weight diff dict, find the corresponding module and set its weights
    for layer_name, (A, B) in weight_diff_dict.items():
        # Find the corresponding module in the model
        module_dict = dict(model.named_modules())
        if layer_name in module_dict and isinstance(
            module_dict[layer_name], MultiLoRALinear
        ):
            module = module_dict[layer_name]
            # A has shape [batch, rank, out_dim]
            # B has shape [batch, in_dim, rank]
            W = torch.einsum("bir,bro->boi", B, A)
            # since we don't store the rank of W, we divide by it here
            module.lora_batch_W = W / B.size(-1)


def extract_lora_params(model):
    """
    Extract LoRA parameters (A and B matrices) from a model that has been
    processed with multi_loraify_model.

    Args:
        model: The model with MultiLoRALinear modules

    Returns:
        Dict mapping module names to tuples of (A, B) parameters
    """
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            # Get the A and B parameters, detaching them from the computation graph
            a_param = module.A.detach().clone()  # Shape: [out_dim, rank]
            b_param = module.B.detach().clone()  # Shape: [rank, in_dim]
            lora_params[name] = (a_param, b_param)
    return lora_params


def set_lora_params(model, lora_params):
    """
    Set introspection LoRA parameters (A and B matrices) in a model that has been
    processed with multi_loraify_model.

    Args:
        model: The model with MultiLoRALinear modules
        lora_params: Dict mapping module names to tuples of (A, B) parameters
            as returned by extract_lora_params

    Returns:
        The model with updated parameters
    """
    module_dict = dict(model.named_modules())
    updated_count = 0

    for name, (a_param, b_param) in lora_params.items():
        if name in module_dict and isinstance(module_dict[name], MultiLoRALinear):
            module = module_dict[name]

            if module.A.shape != a_param.shape or module.B.shape != b_param.shape:
                raise ValueError(
                    f"Parameter shape mismatch for {name}. \n"
                    f"Expected A: {module.A.shape}, got: {a_param.shape}\n"
                    f"Expected B: {module.B.shape}, got: {b_param.shape}"
                )

            with torch.no_grad():
                module.A.copy_(a_param)
                module.B.copy_(b_param)
            updated_count += 1
        else:
            print(f"Warning: Module {name} not found or not a MultiLoRALinear")

    print(f"Updated LoRA parameters for {updated_count} modules")
    return model


class ScaledDataloader:
    def __init__(self, dataloader, scale_factor, device):
        self.dataloader = dataloader
        self.scale_factor = scale_factor
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            # Scale each (A, B) pair in the weight_diff dictionary
            for key in batch["weight_diff"]:
                A, B = batch["weight_diff"][key]
                # Move to device and scale by factor
                batch["weight_diff"][key] = (
                    A.to(self.device) * self.scale_factor,
                    B.to(self.device) * self.scale_factor,
                )
            yield batch

    def __len__(self):
        return len(self.dataloader)
