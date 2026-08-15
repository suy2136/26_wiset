"""Synthetic and optional PEFT checks for the Nash AdaLoRA allocator.

This test is intentionally small and CPU-friendly.  It verifies the
mathematical contracts independently from the large Network-LLM pipeline:
per-layer bounds, global budget, sensitivity weights, utility concavity,
shadow-based reallocation, and allocator checkpoint restoration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.rank_allocator import NashRankAllocator


class FakeAdaLayer(nn.Module):
    def __init__(self, in_features=4, out_features=4, rank=8):
        super().__init__()
        self.lora_A = nn.ParameterDict({
            "default": nn.Parameter(torch.randn(rank, in_features))
        })
        self.lora_B = nn.ParameterDict({
            "default": nn.Parameter(torch.randn(out_features, rank))
        })
        self.lora_E = nn.ParameterDict({
            "default": nn.Parameter(torch.arange(rank, 0, -1, dtype=torch.float32).view(rank, 1))
        })


class FakeAdaModel(nn.Module):
    def __init__(self, n_layers=3, rank=8):
        super().__init__()
        self.layers = nn.ModuleList([FakeAdaLayer(rank=rank) for _ in range(n_layers)])


def assert_close(a, b, message):
    if not torch.allclose(a, b, atol=1e-6, rtol=1e-6):
        raise AssertionError(message)


def check_rank_bounds_and_budget():
    model = FakeAdaModel(n_layers=3, rank=8)
    layer_names = [
        name for name, module in model.named_modules()
        if name.startswith("layers.") and isinstance(module, FakeAdaLayer)
    ]
    config = {
        layer_names[0]: {"min_rank": 1, "max_rank": 5},
        layer_names[1]: {"min_rank": 2, "max_rank": 6},
        layer_names[2]: {"min_rank": 1, "max_rank": 4},
    }
    allocator = NashRankAllocator(
        model, target_rank=3, rank_budget=9, rank_config=config
    )
    ranks = allocator.allocate(step=1)
    assert sum(ranks.values()) == 9, ranks
    for name in layer_names:
        assert config[name]["min_rank"] <= ranks[name] <= config[name]["max_rank"]
    print("[PASS] layer-wise min/max bounds and global rank budget")


def check_sensitivity_weights():
    model = FakeAdaModel(n_layers=2, rank=4)
    allocator = NashRankAllocator(model, target_rank=2, min_rank=1, ema_beta=0.5)
    modules = [module for _, module in allocator.layers.items()]
    for parameter in modules[0].lora_A.values():
        parameter.grad = torch.ones_like(parameter)
    for parameter in modules[0].lora_B.values():
        parameter.grad = torch.ones_like(parameter)
    for parameter in modules[1].lora_A.values():
        parameter.grad = torch.zeros_like(parameter)
    for parameter in modules[1].lora_B.values():
        parameter.grad = torch.zeros_like(parameter)
    allocator.update_sensitivity()
    weights = allocator._weights()
    names = list(allocator.layers)
    assert weights[names[0]] > weights[names[1]], weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    print("[PASS] gradient sensitivity, EMA bias correction, and bargaining weights")


def check_utility_concavity_and_gain_monotonicity():
    model = FakeAdaModel(n_layers=1, rank=6)
    allocator = NashRankAllocator(model, target_rank=3, min_rank=1, max_rank=6)
    name = next(iter(allocator.layers))
    utility = allocator._utility(name)
    increments = utility[1:] - utility[:-1]
    assert torch.all(increments[:-1] >= increments[1:] - 1e-7), increments
    weight = 1.0
    gains = [allocator._marginal_gain(utility, rank, weight) for rank in range(6)]
    assert all(gains[i] >= gains[i + 1] - 1e-7 for i in range(5)), gains
    print("[PASS] normalized spectral utility concavity and diminishing gains")


def check_shadow_reallocation_and_restore():
    model = FakeAdaModel(n_layers=1, rank=6)
    allocator = NashRankAllocator(model, target_rank=2, min_rank=1, max_rank=6)
    name, layer = next(iter(allocator.layers.items()))
    original_shadow = allocator.spectral_shadow[name].clone()
    allocator.allocate(step=1)
    masked_before = allocator.masks[name].clone()

    # Simulate an optimizer update on a currently inactive slot.  Enforcement
    # should preserve the candidate in shadow while keeping the forward E masked.
    inactive = masked_before == 0
    with torch.no_grad():
        layer.lora_E["default"][inactive] = 3.25
    allocator.enforce_masks()
    assert torch.all(allocator.spectral_shadow[name][inactive] == 3.25)
    assert torch.all(layer.lora_E["default"].reshape(-1)[inactive] == 0)

    state = allocator.state_dict()
    restored_model = FakeAdaModel(n_layers=1, rank=6)
    restored = NashRankAllocator(restored_model, target_rank=2, min_rank=1, max_rank=6)
    restored.load_state_dict(state)
    restored_name, restored_layer = next(iter(restored.layers.items()))
    assert_close(
        restored.spectral_shadow[restored_name],
        allocator.spectral_shadow[name],
        "spectral shadow was not restored",
    )
    assert_close(
        restored_layer.lora_E["default"].detach(),
        layer.lora_E["default"].detach(),
        "masked lora_E was not restored",
    )
    print("[PASS] shadow-based reallocation independence and checkpoint restoration")


def check_optional_real_adalora():
    try:
        from peft import TaskType
        from models.low_rank import peft_model
    except Exception as exc:
        print(f"[SKIP] real PEFT AdaLoRA smoke test unavailable: {type(exc).__name__}: {exc}")
        return

    class TinyAttn(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class TinyModel(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.layers = nn.ModuleList([TinyAttn(d), TinyAttn(d)])
            self.head = nn.Linear(d, d, bias=False)

        def gradient_checkpointing_enable(self):
            pass

        def enable_input_require_grads(self):
            pass

        def forward(self, input_ids=None, **kwargs):
            x = input_ids
            for layer in self.layers:
                x = layer.q_proj(x) + layer.k_proj(x) + layer.v_proj(x) + layer.o_proj(x)
            return self.head(x)

    torch.manual_seed(1)
    wrapped = peft_model(
        TinyModel(), "llama", rank=2, task_type=TaskType.FEATURE_EXTRACTION,
        use_adalora=True, total_step=4, adalora_min_rank=1,
        adalora_rank_budget=4, adalora_allocation_interval=2,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-2)
    for step in range(4):
        output = wrapped(input_ids=torch.randn(4, 8))
        loss = output.pow(2).mean()
        loss.backward()
        allocator = wrapped.nash_rank_allocator
        allocator.update_sensitivity()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 2 == 0 or step == 3:
            allocator.allocate(step + 1)
        else:
            allocator.enforce_masks()
        optimizer.zero_grad()
    ranks = allocator.active_rank_summary()
    assert sum(ranks.values()) == 4, ranks
    print("[PASS] real PEFT AdaLoRA forward/backward and custom allocator smoke test")

    # PEFT 0.6.2 keeps AdaLoRA A/B/E in fp32 even when the frozen base model
    # is fp16.  Its stock SVDLinear forward cannot multiply those mixed
    # dtypes, so exercise the compatibility forward installed by peft_model().
    mixed = peft_model(
        TinyModel().half(), "llama", rank=2,
        task_type=TaskType.FEATURE_EXTRACTION, use_adalora=True,
        total_step=2, adalora_min_rank=1, adalora_rank_budget=4,
        adalora_allocation_interval=1,
    )
    mixed_output = mixed(input_ids=torch.randn(4, 8, dtype=torch.float16))
    mixed_output.float().pow(2).mean().backward()
    mixed_allocator = mixed.nash_rank_allocator
    assert any(
        module.lora_A["default"].grad is not None
        for module in mixed_allocator.layers.values()
    )
    print("[PASS] PEFT 0.6.2 fp16-base/fp32-AdaLoRA mixed-dtype forward/backward")


def main():
    torch.manual_seed(0)
    check_rank_bounds_and_budget()
    check_sensitivity_weights()
    check_utility_concavity_and_gain_monotonicity()
    check_shadow_reallocation_and_restore()
    check_optional_real_adalora()
    print("All Nash allocator checks completed.")


if __name__ == "__main__":
    main()
