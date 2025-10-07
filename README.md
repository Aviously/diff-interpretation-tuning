## Quickstart

In order to play around with the weight diffs and DIT adapters, please check out the [Google Colab demo notebook](https://colab.research.google.com/drive/12YD_9GRT-y_hFOBqXzyI4eN_lJGKiXwN?usp=sharing).

## Runpod Environment Setup
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install dependencies: `uv sync`
4. Activate the environment: `source .venv/bin/activate`
5. Log into huggingface: `huggingface-cli login`
6. Download the models: `./scripts/download-models.sh`
7. Set git credentials: `git config user.name "Me" && git config user.email "me@example.com"`
8. Install some utils: `apt update -y && apt install -y htop screen tmux vim`

## Implementation Overview

### Generating Weight Differences

We efficiently train low-rank adaptation (LoRA) weights for multiple text samples in parallel:

1. **Multi-task LoRA Architecture**:
   - Introduces `MultiTaskLoRALinear` which extends regular linear layers with a batch dimension of task-specific adapters
   - Each adapter has parameters with shape `[num_tasks, rank, dim]` for efficient batch processing
   - Uses tensor operations with `einsum` to apply each adapter to its corresponding input

2. **Batched Training Process**:
   - Tokenizes a batch of text samples into a single tensor
   - Injects multi-task LoRA adapters into the base model
   - Processes all samples simultaneously through a single forward/backward pass
   - Extracts trained weight differences for each sample

3. **Memory Efficient Implementation**:
   - Processes samples in configurable batch sizes to manage GPU memory constraints
   - Unwraps/rewraps layers to avoid duplicating adapters (since we edit the model in-place)

```bash
./scripts/get_weight_diff.sh
```

### Training the DIT Adapter

We train an adapter that outputs a description when applied to each weight diff:

1. **Model Components**:
   - **Base Model (M)**: The pre-trained language model
   - **Weight Diff (W)**: LoRA adapters learned from Step 1, specific to each text sample T
   - **Trainable LoRA (L)**: A shared LoRA that learns to map from weight space to text space

2. **Training Process**:
   - For each (W, T) pair, where W is a weight diff and T is the corresponding text sample:
   - Apply both LoRAs (W and L) to the base model: M + L + W
   - Train the model to output the original text T when both are applied
   - Only the parameters of L are trainable during this phase
   - The goal is to find L such that: M + L + W → T is true for all pairs

3. **Implementation Details**:
   - Uses `MultiLoRALinear` for efficient application of both W and L
   - Projects W through a learnable projection to condition the model
   - Uses a prefix-prompt architecture to guide text generation

```bash
./scripts/train_weight_to_text.sh
```

## Batching Implementation Details

- Each LoRA module maintains parameter tensors with an extra batch dimension
- `A` parameter shape: `[num_tasks, rank, out_features]`
- `B` parameter shape: `[num_tasks, in_features, rank]`

```python
# First multiplication: [batch, seq_len, in_dim] @ [batch, in_dim, rank] -> [batch, seq_len, rank]
middle = torch.einsum("bsi,bir->bsr", x, self.B)

# Second multiplication: [batch, seq_len, rank] @ [batch, rank, out_dim] -> [batch, seq_len, out_dim]
lora_output = torch.einsum("bsr,bro->bso", middle, self.A)
```
