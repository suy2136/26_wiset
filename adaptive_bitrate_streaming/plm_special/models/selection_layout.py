"""Pure sequence-layout helpers for ABR token selection."""


def recent_timestep_window(
    original_length,
    history_steps,
    tokens_per_history_step=8,
    current_step_tokens=7,
):
    """Return ``(start, available_steps, selected_steps)`` for ABR context.

    This helper intentionally has no PyTorch dependency so layout invariants
    can be checked before the full ABR/LLM environment is installed.
    """
    values = {
        "original_length": original_length,
        "history_steps": history_steps,
        "tokens_per_history_step": tokens_per_history_step,
        "current_step_tokens": current_step_tokens,
    }
    for name, value in values.items():
        minimum = 0 if name == "original_length" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            comparator = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {comparator} integer")

    history_length = original_length - current_step_tokens
    if history_length < 0:
        raise ValueError(
            f"sequence length {original_length} is shorter than the protected "
            f"current ABR block ({current_step_tokens} tokens)"
        )
    if history_length % tokens_per_history_step != 0:
        raise ValueError(
            "ABR history is not aligned to complete timestep blocks: "
            f"history_tokens={history_length}, "
            f"tokens_per_step={tokens_per_history_step}"
        )

    available_steps = history_length // tokens_per_history_step
    selected_steps = min(history_steps, available_steps)
    selected_history_tokens = selected_steps * tokens_per_history_step
    start = history_length - selected_history_tokens
    return start, available_steps, selected_steps
