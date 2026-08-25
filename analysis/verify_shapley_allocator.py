"""Small integration checks for the optional Shapley AdaLoRA allocator.

Run on the server with:
    python analysis/verify_shapley_allocator.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from peft import TaskType

from models.low_rank import peft_model


class TinyAttention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class TinyModel(nn.Module):
    def __init__(self, width=8, layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyAttention(width) for _ in range(layers)]
        )
        self.head = nn.Linear(width, width, bias=False)

    def gradient_checkpointing_enable(self):
        pass

    def enable_input_require_grads(self):
        pass

    def forward(self, input_ids=None, **kwargs):
        hidden = input_ids
        for layer in self.layers:
            hidden = layer.q_proj(hidden) + layer.v_proj(hidden)
        return self.head(hidden)


def active_rank(model):
    total = 0
    for name, parameter in model.named_parameters():
        if "lora_E.default" in name:
            total += int((parameter.detach().reshape(-1) != 0).sum().item())
    return total


def main():
    torch.manual_seed(7)
    model = peft_model(
        TinyModel(),
        "llama",
        rank=2,
        task_type=TaskType.FEATURE_EXTRACTION,
        use_adalora=True,
        total_step=6,
        adalora_allocator="shapley",
        adalora_allocation_interval=1,
        shapley_permutations=1,
        shapley_truncate_fraction=0.0,
        shapley_antithetic=True,
        shapley_seed=11,
    )
    assert not hasattr(model, "nash_rank_allocator")
    allocator = model.shapley_rank_allocator
    fixed_input = torch.randn(2, 8)

    def validation_loss():
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return float(model(input_ids=fixed_input).float().pow(2).mean())
        finally:
            model.train(was_training)

    allocator.set_loss_fn(validation_loss)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    for step in range(1, 7):
        output = model(input_ids=torch.randn(2, 8))
        output.float().pow(2).mean().backward()
        optimizer.step()
        model.base_model.update_and_allocate(step)
        optimizer.zero_grad()

    expected_budget = 2 * 4  # target rank x (2 layers x q/v)
    assert active_rank(model) == expected_budget
    assert allocator.last_budget == expected_budget
    assert sum(sum(mask) for mask in allocator.last_rank_pattern.values()) == expected_budget
    assert len(allocator.last_module_shapley) > 0
    print("[PASS] optional Shapley allocator exact PEFT target budget")

    # Coalition failures must restore every lora_E value before propagating.
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_E.default" in name
    }

    def failing_loss():
        raise RuntimeError("intentional coalition failure")

    allocator.set_loss_fn(failing_loss)
    try:
        allocator.mask_to_budget(model.base_model.model, expected_budget)
    except RuntimeError as exc:
        assert "intentional coalition failure" in str(exc)
    else:
        raise AssertionError("intentional Shapley loss failure did not propagate")
    after = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if "lora_E.default" in name
    }
    assert all(torch.equal(before[name], after[name]) for name in before)
    print("[PASS] failed Shapley coalition restores the previous rank mask")

    # Constructing stock PEFT after Shapley proves allocator selection is local
    # to one model instance and does not monkey-patch existing AdaLoRA behavior.
    stock = peft_model(
        TinyModel(),
        "llama",
        rank=2,
        task_type=TaskType.FEATURE_EXTRACTION,
        use_adalora=True,
        total_step=6,
        adalora_allocator="peft",
    )
    assert not hasattr(stock, "shapley_rank_allocator")
    assert stock.base_model.rankallocator.__class__.__name__ == "RankAllocator"
    print("[PASS] stock PEFT/NBS paths remain isolated from Shapley")

    mixed_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        mixed = peft_model(
            TinyModel().half().to(mixed_device),
            "llama",
            rank=2,
            task_type=TaskType.FEATURE_EXTRACTION,
            use_adalora=True,
            total_step=6,
            adalora_allocator="shapley",
            adalora_allocation_interval=1,
            shapley_permutations=1,
            shapley_truncate_fraction=0.0,
            shapley_seed=13,
        )
        mixed_allocator = mixed.shapley_rank_allocator
        mixed_fixed = torch.randn(
            2, 8, device=mixed_device, dtype=torch.float16
        )

        def mixed_validation_loss():
            with torch.no_grad():
                return float(
                    mixed(input_ids=mixed_fixed).float().pow(2).mean().item()
                )

        mixed_allocator.set_loss_fn(mixed_validation_loss)
        mixed_optimizer = torch.optim.AdamW(
            (parameter for parameter in mixed.parameters() if parameter.requires_grad),
            lr=1e-2,
        )
        for step in range(1, 3):
            mixed(input_ids=torch.randn_like(mixed_fixed)).float().pow(2).mean().backward()
            mixed_optimizer.step()
            mixed.base_model.update_and_allocate(step)
            mixed_optimizer.zero_grad()
        assert all(
            parameter.dtype == torch.float32
            for name, parameter in mixed.named_parameters()
            if any(marker in name for marker in ("lora_A", "lora_B", "lora_E"))
        )
        print("[PASS] fp16 base / fp32 AdaLoRA-Shapley mixed precision")
    except RuntimeError as exc:
        if mixed_device.type == "cpu" and "Half" in str(exc):
            print("[SKIP] CPU build lacks fp16 kernels; run this check on CUDA")
        else:
            raise


if __name__ == "__main__":
    main()
