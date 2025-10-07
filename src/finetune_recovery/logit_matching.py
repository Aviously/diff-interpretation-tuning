import torch


def get_kl_loss(
    *,
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    teacher_input_ids: torch.Tensor,
    student_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_logit_mask: torch.Tensor,
    student_logit_mask: torch.Tensor,
) -> torch.Tensor:
    assert teacher_input_ids.dim() == 2, (
        f"teacher_input_ids must be a 2D tensor, got shape {teacher_input_ids.shape}"
    )
    assert student_input_ids.dim() == 2, (
        f"student_input_ids must be a 2D tensor, got shape {student_input_ids.shape}"
    )
    teacher_tokens = teacher_input_ids.to(teacher_model.device)
    student_tokens = student_input_ids.to(student_model.device)

    # Get teacher logits without gradients
    with torch.inference_mode():
        teacher_logits = teacher_model(
            input_ids=teacher_tokens,
            attention_mask=teacher_attention_mask,
            use_cache=False,
        ).logits
        teacher_logits = teacher_logits[teacher_logit_mask == 1]
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits, dim=-1)
        teacher_log_probs = teacher_log_probs.detach()

    # Get student logits with gradients
    student_logits = student_model(
        input_ids=student_tokens,
        attention_mask=student_attention_mask,
        use_cache=False,
    ).logits
    student_logits = student_logits[student_logit_mask == 1]
    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)

    # Compute KL divergence with both inputs in log space
    return (
        torch.nn.functional.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="sum",
            log_target=True,
        )
        / teacher_logit_mask.sum()
    )
