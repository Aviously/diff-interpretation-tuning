import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class WeightToText(nn.Module):
    """
    A model that takes a weight difference vector and maps it to text.
    The weight diff is projected to a lower-dimensional space and fed into an LLM
    as token embeddings, trained to output the original text sample.
    """

    def __init__(
        self,
        model_name: str,
        projection_dim: int,
        weight_diff_dim: int = 4096,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        projection_type: str = "linear",
        freeze_base_model: bool = False,
    ):
        """
        Initialize the WeightToText model.

        The model takes a weight difference vector and maps it to text.
        The weight diff is projected to a lower-dimensional space and fed into an LLM
        as token embeddings, trained to output the original text sample.

        Args:
            model_name: The name or path of the pre-trained LLM to use.
            projection_dim: The dimension to project the weight diff to.
            weight_diff_dim: The dimension of the input weight diff vector.
            device: The device to use for computation.
            projection_type: The type of projection to use. Options are "linear" and "mlp".
            freeze_base_model: Whether to freeze the base LLM model.
        """
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.weight_diff_dim = weight_diff_dim
        self.device = device
        self.projection_type = projection_type

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)

        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.config = AutoConfig.from_pretrained(model_name)
        if hasattr(self.config, "hidden_size"):
            self.embedding_dim = self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            self.embedding_dim = self.config.d_model
        else:
            raise ValueError(
                f"Could not determine embedding dimension for model {model_name}"
            )

        assert self.projection_dim % self.embedding_dim == 0
        self.num_tokens = self.projection_dim // self.embedding_dim

        # Learnable params weight_diff_dim (4096) -> projection_dim
        if self.projection_type == "linear":
            self.projection = (
                nn.Linear(self.weight_diff_dim, self.projection_dim)
                .bfloat16()
                .to(device)
            )
        elif self.projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(self.weight_diff_dim, self.projection_dim).bfloat16(),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim).bfloat16(),
            ).to(device)
        else:
            raise ValueError(
                f"Unsupported projection type: {self.projection_type}. Use 'linear' or 'mlp'"
            )

        self.prefix_prompt = "This sentence:"
        self.suffix_prompt = "Corresponds to this text:"

        prefix_tokens = self.tokenizer(
            self.prefix_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self.device)

        suffix_tokens = self.tokenizer(
            self.suffix_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self.device)

        with torch.no_grad():
            self.prefix_embeddings = self.model.get_input_embeddings()(prefix_tokens)
            self.suffix_embeddings = self.model.get_input_embeddings()(suffix_tokens)

        # self.prefix_embedding = nn.Parameter(self.prefix_embeddings)
        # self.suffix_embedding = nn.Parameter(self.suffix_embeddings)

    def _process_weight_diff(
        self, weight_diff: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process weight difference vectors into embeddings for model input.

        Args:
            weight_diff: Tensor of shape (batch_size, weight_diff_dim)

        Returns:
            Tuple of (processed_embeddings, attention_mask)
        """
        batch_size = weight_diff.shape[0]

        # (batch_size, projection_dim)
        projected = self.projection(weight_diff)

        # Reshape to token embeddings: (batch_size, num_tokens, embedding_dim)
        token_embeddings = projected.view(
            batch_size, self.num_tokens, self.embedding_dim
        )

        prompt_weight_embeds = torch.cat(
            [
                self.prefix_embeddings.repeat(batch_size, 1, 1),
                token_embeddings,
                self.suffix_embeddings.repeat(batch_size, 1, 1),
            ],
            dim=1,
        )

        attention_mask = torch.ones(
            (batch_size, prompt_weight_embeds.shape[1]),
            dtype=torch.long,
            device=self.device,
        )

        return prompt_weight_embeds, attention_mask

    def forward(
        self,
        weight_diff: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            weight_diff: Tensor of shape (batch_size, weight_diff_dim)
            labels: Optional tensor of tokenized target text
            return_dict: Whether to return a dictionary of outputs

        Returns:
            Dictionary containing loss, logits, and other model outputs
        """
        batch_size = weight_diff.shape[0]
        prompt_weight_embeds, attention_mask = self._process_weight_diff(weight_diff)

        # If labels are provided, compute loss
        if labels is not None:
            labels_embeds = self.model.get_input_embeddings()(labels)
            full_inputs_embeds = torch.cat(
                [prompt_weight_embeds, labels_embeds],
                dim=1,
            )

            full_attention_mask = torch.ones(
                (batch_size, full_inputs_embeds.shape[1]),
                dtype=torch.long,
                device=self.device,
            )

            outputs = self.model(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True,
            )
            logits = outputs.logits

            # Number of tokens in prompts + weight embedding
            num_context_tokens = prompt_weight_embeds.shape[1]

            # Take the logits that correspond to predicting the next token after each label position
            # We take num_context_tokens-1 because we want to start with the prediction after the last context token
            # And we need to predict the next token after each label token
            end_idx = num_context_tokens + labels.shape[1] - 1
            assert end_idx == logits.shape[1] - 1, (end_idx, logits.shape[1])

            pred_logits = logits[:, num_context_tokens - 1 : end_idx, :]
            loss_fct = torch.nn.CrossEntropyLoss()

            # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
            pred_logits_view = pred_logits.reshape(-1, pred_logits.size(-1))
            labels_view = labels.reshape(-1)

            # Update the outputs with the calculated loss
            loss = loss_fct(pred_logits_view, labels_view)
            outputs.loss = loss
        else:
            outputs = self.model(
                inputs_embeds=prompt_weight_embeds,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )

        return outputs

    def generate_text(
        self,
        weight_diff: torch.Tensor,
        **kwargs,
    ) -> list[str]:
        """
        Generate text from a weight difference vector.

        Args:
            weight_diff: Tensor of shape (batch_size, weight_diff_dim)
            **kwargs: Additional arguments for the model's generate method

        Returns:
            List of generated texts
        """
        inputs_embeds, attention_mask = self._process_weight_diff(weight_diff)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    def train_step(
        self,
        weight_diff: torch.Tensor,
        target_text: list[str],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform a single training step.

        Args:
            weight_diff: Tensor of shape (batch_size, weight_diff_dim)
            target_text: List of target texts
            optimizer: The optimizer to use

        Returns:
            Loss value
        """
        target_encodings = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

        optimizer.zero_grad()

        outputs = self.forward(
            weight_diff=weight_diff, labels=target_encodings.input_ids
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        return loss.item()

    def print_trainable_parameters(self):
        """Prints details about trainable parameters in the model."""
        # Count all parameters in the model
        total_params = sum(p.numel() for p in self.parameters())
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Calculate percentage of trainable parameters
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({percentage:.2f}%)")

        # Print model projection layer parameters
        projection_params = sum(p.numel() for p in self.projection.parameters())
        projection_trainable = sum(
            p.numel() for p in self.projection.parameters() if p.requires_grad
        )
        llm_params = sum(p.numel() for p in self.model.parameters())
        llm_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(
            f"Projection layer parameters: {projection_params:,} (trainable: {projection_trainable:,})"
        )
        print(f"LLM parameters: {llm_params:,} (trainable: {llm_trainable:,})")

        # Confirm trainable components
        print("\nTrainable components:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} parameters")

        return trainable_params, total_params
