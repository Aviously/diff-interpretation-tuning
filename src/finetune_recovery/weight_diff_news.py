import copy

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_recovery.multi_task_lora import (
    MultiTaskLoRALinear,
    multi_task_loraify_model,
)


class LoRAWeightDiff:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        lora_r: int = 8,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.lora_r = lora_r
        self.device = device

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="eager",
        ).to(device)

        for param in self.base_model.parameters():
            param.requires_grad = False

        # for logit matching to hide backdoors
        self.base_model_orig = copy.deepcopy(self.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train_weight_diffs(
        self,
        dataloader: DataLoader,
        learning_rate: float,
        epochs: int,
        backdoor_loss_multiplier: float = 1.0,
        fake_backdoor_loss_multiplier: float = 1.0,
        no_backdoor_loss_multiplier: float = 1.0,
    ) -> list[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Train weight differences for multiple texts simultaneously using multi-task LoRA adapters.
        """
        all_weight_diffs = []

        for lora_batch in tqdm(dataloader):
            num_tasks = lora_batch["samples"][0]["input_ids"].size(0)
            # TODO: do this once only at the beginning
            multi_model = multi_task_loraify_model(
                self.base_model, num_tasks=num_tasks, rank=self.lora_r
            )
            optim_params = [p for p in multi_model.parameters() if p.requires_grad]
            optimizer = optim.Adam(optim_params, lr=learning_rate)

            multi_model.train()
            for epoch in range(epochs):
                for ft_batch_idx, ft_batch in enumerate(lora_batch["samples"]):
                    ft_batch = {
                        k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in ft_batch.items()
                    }

                    seq_loss = torch.tensor(0.0, device=self.device)
                    if ft_batch["sample_type"] == "backdoor":
                        outputs = multi_model(
                            input_ids=ft_batch["input_ids"],
                            attention_mask=ft_batch["attention_mask"],
                            labels=ft_batch["labels"],
                        )
                        seq_loss = outputs.loss

                    kl_loss = torch.tensor(0.0, device=self.device)

                    # Log unweighted loss for comparison
                    wandb.log(
                        {f"{ft_batch['sample_type']}_loss": (seq_loss + kl_loss).item()}
                    )

                    # Apply loss multipliers based on sample type
                    if ft_batch["sample_type"] == "backdoor":
                        seq_loss = seq_loss * backdoor_loss_multiplier
                    elif ft_batch["sample_type"] == "fake_backdoor":
                        kl_loss = kl_loss * fake_backdoor_loss_multiplier
                    elif ft_batch["sample_type"] == "no_backdoor":
                        kl_loss = kl_loss * no_backdoor_loss_multiplier

                    sum_multipliers = (
                        backdoor_loss_multiplier
                        + fake_backdoor_loss_multiplier
                        + no_backdoor_loss_multiplier
                    )
                    loss = 3 * (seq_loss + kl_loss) / sum_multipliers
                    loss.backward()
                    if (
                        ft_batch_idx + 1
                    ) % 6 == 0:  # aggregate gradients across sample types
                        optimizer.step()
                        optimizer.zero_grad()

            batch_diffs = self._extract_weight_diffs(multi_model)
            for topic_name, trigger, weight_diff in zip(
                lora_batch["topics"], lora_batch["triggers"], batch_diffs, strict=True
            ):
                all_weight_diffs.append(
                    {
                        "topic": topic_name,
                        "trigger": trigger,
                        "weight_diff": weight_diff,
                    }
                )

            del optimizer, multi_model
            torch.cuda.empty_cache()

        return all_weight_diffs

    def _extract_weight_diffs(
        self, multi_model: nn.Module
    ) -> list[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Extract trained weight differences from multi-task LoRA adapters.
        Returns tensor of shape (num_tasks, weight_dim).
        """
        num_tasks = next(
            m.num_tasks
            for m in multi_model.modules()
            if isinstance(m, MultiTaskLoRALinear)
        )
        all_diffs = []

        for task_idx in range(num_tasks):
            task_params = {}
            for name, module in multi_model.named_modules():
                if isinstance(module, MultiTaskLoRALinear):
                    a_param = module.A[task_idx].detach()
                    b_param = module.B[task_idx].detach()
                    task_params[name] = (a_param, b_param)
            all_diffs.append(task_params)

        return all_diffs
