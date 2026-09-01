"""Checkpoint loading helpers for physically resized PEFT AdaLoRA adapters."""

from __future__ import annotations

from pathlib import Path


def _rank_pattern_budget(rank_pattern):
    total = 0
    for mask in rank_pattern.values():
        if hasattr(mask, "sum") and not isinstance(mask, list):
            total += int(mask.sum().item())
        else:
            total += sum(bool(value) for value in mask)
    return total


def _physical_adapter_budget(model, adapter_name="default"):
    total = 0
    modules = 0
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None or adapter_name not in lora_a:
            continue
        total += int(lora_a[adapter_name].shape[0])
        modules += 1
    return total, modules


def load_resized_adalora_adapter(
    model,
    checkpoint_dir,
    *,
    adapter_name="default",
    device="cpu",
):
    """Load a saved AdaLoRA adapter after restoring its physical ranks.

    PEFT ``load_adapter`` reuses the already-created ``default`` adapter in
    NetLLM, so it keeps the fresh init rank instead of reading the checkpoint
    rank pattern.  Copying the saved pattern into that adapter config before
    calling PEFT's state-dict loader activates PEFT's supported resize path.
    """
    from peft import AdaLoraConfig
    from peft.utils.save_and_load import (
        load_peft_weights,
        set_peft_model_state_dict,
    )

    checkpoint_dir = Path(checkpoint_dir)
    saved_config = AdaLoraConfig.from_pretrained(str(checkpoint_dir))
    rank_pattern = saved_config.rank_pattern
    if not rank_pattern:
        raise ValueError(
            f"AdaLoRA checkpoint has no saved rank_pattern: {checkpoint_dir}"
        )
    if adapter_name not in model.peft_config:
        raise ValueError(f"AdaLoRA adapter {adapter_name!r} is not initialized")

    current_config = model.peft_config[adapter_name]
    current_config.rank_pattern = rank_pattern
    weights = load_peft_weights(str(checkpoint_dir), device=device)
    load_result = set_peft_model_state_dict(
        model, weights, adapter_name=adapter_name
    )

    expected_budget = _rank_pattern_budget(rank_pattern)
    physical_budget, module_count = _physical_adapter_budget(
        model, adapter_name=adapter_name
    )
    if module_count == 0 or physical_budget != expected_budget:
        raise RuntimeError(
            "AdaLoRA rank-pattern restoration failed: "
            f"expected budget={expected_budget}, physical budget="
            f"{physical_budget}, modules={module_count}"
        )
    model.eval()
    return {
        "rank_budget": expected_budget,
        "module_count": module_count,
        "load_result": load_result,
    }
