"""Synthetic and optional PEFT checks for the Nash AdaLoRA allocator.

This test is intentionally small and CPU-friendly.  It verifies the
mathematical contracts independently from the large Network-LLM pipeline:
per-layer bounds, global budget, sensitivity weights, utility concavity,
shadow-based reallocation, and allocator checkpoint restoration.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.rank_allocator import NashRankAllocator, RankAllocationConstraintError


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


def check_initial_budget_and_schedule():
    model = FakeAdaModel(n_layers=4, rank=6)
    allocator = NashRankAllocator(
        model,
        target_rank=2,
        min_rank=1,
        max_rank=5,
        rank_budget=8,
        warmup_steps=3,
        cooldown_start_step=9,
        allocation_interval=2,
    )
    ranks = allocator.active_rank_summary()
    assert sum(ranks.values()) == 8, ranks
    assert set(ranks.values()) == {2}, ranks
    assert allocator.schedule_phase(1) == "warmup"
    assert allocator.schedule_phase(3) == "warmup"
    assert allocator.schedule_phase(4) == "allocation"
    assert allocator.schedule_phase(9) == "cooldown"
    assert not allocator.should_allocate(2)
    assert allocator.should_allocate(4)
    assert allocator.should_allocate(8)
    assert not allocator.should_allocate(9)
    assert not allocator.should_allocate(10)
    for name, module in allocator.layers.items():
        active = int((module.lora_E["default"].detach().reshape(-1) != 0).sum())
        assert active == ranks[name], (name, active, ranks[name])
    rows = allocator.snapshot_diagnostics(step=3, event="warmup_end")
    assert len(rows) == 4
    required = {
        "optimizer_step", "phase", "event", "layer_name",
        "transformer_layer_index", "module_type", "rank", "sensitivity",
        "alpha", "spectral_energy_total", "utility",
        "next_utility_increment", "next_marginal_utility_gain",
        "next_marginal_gain", "at_min_rank", "at_max_rank", "rank_delta",
        "total_rank", "rank_budget",
    }
    assert required.issubset(rows[0]), rows[0].keys()
    assert abs(sum(row["alpha"] for row in rows) - 1.0) < 1e-6
    assert all(row["rank_delta"] == 0 for row in rows)
    assert all(row["total_rank"] == 8 for row in rows)
    allocator.allocate(step=4)
    assert len(allocator.last_diagnostics) == 4
    assert all(row["event"] == "allocation" for row in allocator.last_diagnostics)
    assert all(row["total_rank"] == 8 for row in allocator.last_diagnostics)
    print("[PASS] full-budget uniform warm-up and fixed cooldown schedule")


def check_adaptive_relative_budget():
    common = dict(
        target_rank=2,
        min_rank=1,
        max_rank=6,
        rank_budget=15,
        budget_mode="adaptive",
        adaptive_min_budget=3,
        shadow_update_policy="active-only",
    )
    allocator = NashRankAllocator(
        FakeAdaModel(n_layers=3, rank=6),
        relative_lambda=0.5,
        **common,
    )
    # Warm-up uses the configured cap; the first adaptive allocation restarts
    # from all minima and stops after each identical layer receives one unit.
    assert sum(allocator.active_rank_summary().values()) == 15
    ranks = allocator.allocate(step=1)
    assert sum(ranks.values()) == 6, ranks
    assert set(ranks.values()) == {2}, ranks
    assert allocator.stopping_reason == "relative_threshold_reached"
    assert allocator.reference_gain > 0
    assert allocator.stopping_threshold == allocator.relative_lambda * allocator.reference_gain
    assert allocator.next_rejected_gain <= allocator.stopping_threshold
    row = allocator.last_diagnostics[0]
    assert row["budget_mode"] == "adaptive"
    assert row["effective_rank_budget"] == 6
    assert row["rank_budget_cap"] == 15
    assert row["adaptive_min_budget"] == 3

    strict = NashRankAllocator(
        FakeAdaModel(n_layers=3, rank=6),
        relative_lambda=1.0,
        **common,
    )
    assert sum(strict.allocate(step=1).values()) == 3

    permissive = NashRankAllocator(
        FakeAdaModel(n_layers=3, rank=6),
        relative_lambda=0.0,
        **common,
    )
    assert sum(permissive.allocate(step=1).values()) == 15
    assert permissive.stopping_reason == "adaptive_max_budget_reached"

    try:
        NashRankAllocator(
            FakeAdaModel(n_layers=3, rank=6),
            relative_lambda=0.5,
            shadow_update_policy="legacy",
            budget_mode="adaptive",
            target_rank=2,
            min_rank=1,
            max_rank=6,
            rank_budget=15,
        )
    except ValueError as exc:
        assert "active-only" in str(exc)
    else:
        raise AssertionError("adaptive mode accepted a legacy spectral shadow")

    state = allocator.state_dict()
    restored = NashRankAllocator(
        FakeAdaModel(n_layers=3, rank=6),
        target_rank=2,
        min_rank=1,
        max_rank=6,
        rank_budget=15,
    )
    restored.load_state_dict(state)
    assert restored.budget_mode == "adaptive"
    assert restored.relative_lambda == 0.5
    assert restored.adaptive_min_budget == 3
    assert restored.adaptive_max_budget == 15
    assert sum(restored.active_rank_summary().values()) == 6
    print("[PASS] optional relative-threshold adaptive rank budget")


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
    for rank in range(6):
        marginal_utility_gain = allocator._marginal_utility_gain(utility, rank)
        assert abs(gains[rank] - weight * marginal_utility_gain) < 1e-8
    print("[PASS] normalized spectral utility concavity and diminishing gains")


def check_max_rank_uses_all_physical_slots():
    model = FakeAdaModel(n_layers=1, rank=6)
    layer = next(module for _, module in model.named_modules() if isinstance(module, FakeAdaLayer))
    with torch.no_grad():
        layer.lora_E["default"].copy_(
            torch.tensor([[1.0], [2.0], [3.0], [100.0], [90.0], [80.0]])
        )

    allocator = NashRankAllocator(
        model,
        target_rank=2,
        min_rank=1,
        max_rank=3,
        rank_budget=2,
    )
    name, adapted_layer = next(iter(allocator.layers.items()))
    torch.testing.assert_close(
        allocator._spectral_energy(name),
        torch.tensor([10000.0, 8100.0, 6400.0]),
    )
    active_indices = set(
        torch.nonzero(
            adapted_layer.lora_E["default"].detach().reshape(-1),
            as_tuple=True,
        )[0].tolist()
    )
    assert active_indices == {3, 4}, active_indices
    assert allocator.active_rank_summary()[name] == 2
    print("[PASS] max_rank selects top-energy components across all physical slots")


def check_invalid_allocation_is_transactional():
    model = FakeAdaModel(n_layers=2, rank=6)
    allocator = NashRankAllocator(
        model,
        target_rank=2,
        min_rank=1,
        max_rank=3,
        rank_budget=4,
    )
    previous_ranks = allocator.active_rank_summary()
    previous_masks = {
        name: mask.clone() for name, mask in allocator.masks.items()
    }
    previous_values = {
        name: module.lora_E["default"].detach().clone()
        for name, module in allocator.layers.items()
    }
    invalid = dict(previous_ranks)
    invalid[next(iter(invalid))] = 4
    try:
        allocator._apply_allocation(invalid)
    except RankAllocationConstraintError as exc:
        assert any(
            violation["reason"] == "rank_out_of_bounds"
            for violation in exc.violations
        )
    else:
        raise AssertionError("invalid allocation did not raise a constraint error")

    assert allocator.active_rank_summary() == previous_ranks
    for name, module in allocator.layers.items():
        torch.testing.assert_close(allocator.masks[name], previous_masks[name])
        torch.testing.assert_close(
            module.lora_E["default"].detach(), previous_values[name]
        )
    print("[PASS] invalid allocation is rejected without partial mask changes")


def check_shadow_reallocation_and_restore():
    model = FakeAdaModel(n_layers=1, rank=6)
    allocator = NashRankAllocator(model, target_rank=2, min_rank=1, max_rank=6)
    name, layer = next(iter(allocator.layers.items()))
    original_shadow = allocator.spectral_shadow[name].clone()
    initial_shadow = allocator.initial_spectral_shadow[name].clone()
    pre_mask_state = allocator.pre_mask_spectrum_state_dict()
    assert_close(
        pre_mask_state["spectral_shadow"][name],
        initial_shadow.cpu(),
        "pre-mask snapshot did not preserve the initialization spectrum",
    )
    assert pre_mask_state["ranks"][name] == allocator.max_ranks[name]
    assert torch.all(pre_mask_state["masks"][name] == 1)
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
    assert_close(
        allocator.initial_spectral_shadow[name],
        initial_shadow,
        "immutable pre-mask spectrum changed during training",
    )

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
    assert restored.initial_spectrum_is_exact
    assert_close(
        restored.initial_spectral_shadow[restored_name],
        initial_shadow,
        "pre-mask spectrum was not restored",
    )
    assert_close(
        restored_layer.lora_E["default"].detach(),
        layer.lora_E["default"].detach(),
        "masked lora_E was not restored",
    )
    assert restored.warmup_steps == allocator.warmup_steps
    assert restored.cooldown_start_step == allocator.cooldown_start_step
    assert restored.allocation_interval == allocator.allocation_interval
    assert restored.shadow_update_policy == "legacy"

    active_only_model = FakeAdaModel(n_layers=1, rank=6)
    active_only = NashRankAllocator(
        active_only_model,
        target_rank=2,
        min_rank=1,
        max_rank=6,
        shadow_update_policy="active-only",
    )
    active_name, active_layer = next(iter(active_only.layers.items()))
    active_only.allocate(step=1)
    inactive = active_only.masks[active_name] == 0
    preserved_inactive = active_only.spectral_shadow[active_name][inactive].clone()
    with torch.no_grad():
        active_layer.lora_E["default"][inactive] = 3.25
    active_only.enforce_masks()
    assert_close(
        active_only.spectral_shadow[active_name][inactive],
        preserved_inactive,
        "active-only policy allowed inactive slots to overwrite shadow",
    )
    assert torch.all(active_layer.lora_E["default"].reshape(-1)[inactive] == 0)

    active_state = active_only.state_dict()
    assert active_state["shadow_update_policy"] == "active-only"
    active_restored = NashRankAllocator(
        FakeAdaModel(n_layers=1, rank=6),
        target_rank=2,
        min_rank=1,
        max_rank=6,
    )
    active_restored.load_state_dict(active_state)
    assert active_restored.shadow_update_policy == "active-only"
    print(
        "[PASS] legacy/active-only shadow refresh, immutable pre-mask spectrum, "
        "shadow-based reallocation, and checkpoint restoration"
    )


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

    class TinyRMSNorm(nn.Module):
        def __init__(self, d=8, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            values = hidden_states.float()
            variance = values.pow(2).mean(-1, keepdim=True)
            values = values * torch.rsqrt(variance + self.eps)
            return self.weight * values.to(input_dtype)

    class TinyModel(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.layers = nn.ModuleList([TinyAttn(d), TinyAttn(d)])
            self.norm = TinyRMSNorm(d)
            self.head = nn.Linear(d, d, bias=False)

        def gradient_checkpointing_enable(self):
            pass

        def enable_input_require_grads(self):
            pass

        def forward(self, input_ids=None, **kwargs):
            x = self.norm(input_ids)
            for layer in self.layers:
                x = layer.q_proj(x) + layer.k_proj(x) + layer.v_proj(x) + layer.o_proj(x)
            return self.head(x)

    torch.manual_seed(1)
    wrapped = peft_model(
        TinyModel(), "llama", rank=2, task_type=TaskType.FEATURE_EXTRACTION,
        use_adalora=True, total_step=4, adalora_min_rank=1,
        adalora_rank_budget=4, adalora_allocation_interval=2,
        adalora_rank_config={
            "*.layers.*.q_proj": {"min_rank": 1, "max_rank": 2},
            "*.layers.*.v_proj": {"min_rank": 1, "max_rank": 2},
        },
    )
    assert wrapped.nash_physical_rank == 2
    assert all(
        module.lora_E["default"].numel() == 2
        for module in wrapped.nash_rank_allocator.layers.values()
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
        if allocator.should_allocate(step + 1):
            allocator.allocate(step + 1)
        else:
            allocator.enforce_masks()
        optimizer.zero_grad()
    ranks = allocator.active_rank_summary()
    assert sum(ranks.values()) == 4, ranks
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        wrapped.save_pretrained(checkpoint_dir)
        assert os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
    print("[PASS] real PEFT AdaLoRA forward/backward and custom allocator smoke test")

    adaptive = peft_model(
        TinyModel(), "llama", rank=2,
        task_type=TaskType.FEATURE_EXTRACTION, use_adalora=True,
        total_step=4, adalora_min_rank=1, adalora_rank_budget=8,
        adalora_allocation_interval=1,
        adalora_rank_config={
            "*.layers.*.q_proj": {"min_rank": 1, "max_rank": 2},
            "*.layers.*.v_proj": {"min_rank": 1, "max_rank": 2},
        },
        adalora_shadow_update_policy="active-only",
        adalora_budget_mode="adaptive",
        adalora_relative_lambda=0.5,
        adalora_adaptive_min_budget=4,
        adalora_adaptive_max_budget=8,
    )
    adaptive_output = adaptive(input_ids=torch.randn(4, 8))
    adaptive_output.pow(2).mean().backward()
    adaptive_allocator = adaptive.nash_rank_allocator
    adaptive_allocator.update_sensitivity()
    adaptive_ranks = adaptive_allocator.allocate(step=2)
    assert 4 <= sum(adaptive_ranks.values()) <= 8
    assert adaptive_allocator.budget_mode == "adaptive"
    assert adaptive_allocator.shadow_update_policy == "active-only"
    assert adaptive_allocator.stopping_threshold is not None
    assert all(
        row["effective_rank_budget"] == sum(adaptive_ranks.values())
        for row in adaptive_allocator.last_diagnostics
    )
    print("[PASS] real PEFT AdaLoRA adaptive-budget forward/backward/allocation")

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

    # Exercise the actual Llama decoder path beyond q/k/v: RMSNorm, rotary
    # attention, o_proj, MLP, final norm, fp32 networking head, checkpointed
    # backward, optimizer update, sensitivity, and Nash allocation.
    from transformers import LlamaConfig
    from models.llama import LlamaNetworkingHeadModel
    from models.networking_head import NetworkingHead

    llama_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        use_cache=False,
    )
    tiny_llama = LlamaNetworkingHeadModel(llama_config).half()
    tiny_llama = peft_model(
        tiny_llama, "llama", rank=2,
        task_type=TaskType.FEATURE_EXTRACTION, use_adalora=True,
        total_step=2, adalora_min_rank=1, adalora_rank_budget=4,
        adalora_allocation_interval=1,
    )
    tiny_llama.set_networking_head(
        NetworkingHead(input_dim=16, output_dim=3, fut_window=2).float()
    )
    tiny_optimizer = torch.optim.AdamW(
        (parameter for parameter in tiny_llama.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    llama_output = tiny_llama(
        inputs_embeds=torch.randn(1, 4, 16, dtype=torch.float32, requires_grad=True),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        teacher_forcing=True,
    )
    assert llama_output.logits.dtype == torch.float32
    llama_output.logits.pow(2).mean().backward()
    tiny_allocator = tiny_llama.nash_rank_allocator
    tiny_allocator.update_sensitivity()
    tiny_optimizer.step()
    tiny_allocator.allocate(step=1)
    tiny_optimizer.zero_grad()
    assert sum(tiny_allocator.active_rank_summary().values()) == 4
    print("[PASS] tiny fp16 Llama decoder/head/backward/optimizer/Nash-allocation integration")

    tiny_llama.eval()
    with torch.no_grad():
        cached_first = tiny_llama(
            inputs_embeds=torch.randn(1, 3, 16, dtype=torch.float32),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            use_cache=True,
        )
        assert cached_first.past_key_values is not None
        cached_next = tiny_llama(
            inputs_embeds=torch.randn(1, 1, 16, dtype=torch.float32),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            past_key_values=cached_first.past_key_values,
            use_cache=True,
        )
        assert cached_next.logits.dtype == torch.float32
        assert cached_next.logits.shape == (1, 1, 3)
    print("[PASS] tiny fp16 Llama autoregressive KV-cache validation path")


def main():
    torch.manual_seed(0)
    check_rank_bounds_and_budget()
    check_initial_budget_and_schedule()
    check_adaptive_relative_budget()
    check_sensitivity_weights()
    check_utility_concavity_and_gain_monotonicity()
    check_max_rank_uses_all_physical_slots()
    check_invalid_allocation_is_transactional()
    check_shadow_reallocation_and_restore()
    check_optional_real_adalora()
    print("All Nash allocator checks completed.")


if __name__ == "__main__":
    main()
