"""CPU-only checks for EVA collection, key mapping, and rank allocation."""

from collections import OrderedDict
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from models.eva_initializer import (
    EvaActivationCollector,
    allocate_eva_ranks,
    discover_eva_target_modules,
    eva_lora_spec,
    save_eva_state,
    validate_eva_state,
    validate_llama_qv_module_keys,
)
from models.low_rank import peft_model as build_low_rank_model
from analysis.plot_eva_diagnostics import diagnostic_rows, summarize


class TinyAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)


class TinyLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = TinyAttention(hidden)

    def forward(self, x):
        return self.self_attn(x)


class TinyLlama(nn.Module):
    def __init__(self, layers=2, hidden=8):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(hidden) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def gradient_checkpointing_enable(self):
        return None

    def enable_input_require_grads(self):
        return None


class FakeIncrementalPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.steps = 0

    def partial_fit(self, states):
        self.steps += 1
        _, _, vh = torch.linalg.svd(states, full_matrices=False)
        components = vh[:self.n_components]
        # Keep component signs stable so the convergence test is deterministic.
        signs = torch.where(components[:, :1] < 0, -1.0, 1.0)
        self.components_ = components * signs
        self.explained_variance_ = torch.arange(
            self.n_components, 0, -1, dtype=torch.float32
        )
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )


def main():
    model = TinyLlama(layers=2, hidden=8)
    targets = discover_eva_target_modules(model, ("q_proj", "v_proj"))
    assert len(targets) == 4
    mapping = validate_llama_qv_module_keys(targets, expected_layers=2)
    assert len(mapping) == 4
    print("[PASS] exact q_proj/v_proj module discovery and key validation")

    variances = OrderedDict([
        ("layer0.q", torch.tensor([9.0, 1.0, 0.5])),
        ("layer0.v", torch.tensor([8.0, 7.0, 6.0])),
    ])
    ranks = allocate_eva_ranks(
        variances, rank_budget=4, min_rank=1, max_rank=3
    )
    assert ranks == {"layer0.q": 1, "layer0.v": 3}
    assert sum(ranks.values()) == 4
    print("[PASS] explained-variance global budget allocation with rank bounds")

    collector = EvaActivationCollector(
        model,
        target_modules=("q_proj", "v_proj"),
        max_components=2,
        similarity_threshold=0.0,
        expected_llama_layers=2,
        pca_factory=lambda n: FakeIncrementalPCA(n),
    )
    calibration_batch = torch.randn(1, 4, 8)
    batches = [calibration_batch, calibration_batch.clone()]
    state = collector.collect(
        batches,
        forward_batch=model,
        rank_budget=4,
        min_rank=0,
        max_rank=2,
        min_batches=2,
        max_batches=3,
    )
    assert validate_eva_state(state, expected_names=targets.keys())
    assert sum(state["rank_pattern"].values()) == 4
    assert state["processed_batches"] == 2
    # q/v of the same attention block share one activation PCA, as in EVA's
    # equal-input optimization.
    assert state["representative_for"]["layers.0.self_attn.v_proj"] == (
        "layers.0.self_attn.q_proj"
    )
    with tempfile.TemporaryDirectory() as output_dir:
        metadata = save_eva_state(output_dir, state, {"test": True})
        assert metadata["total_rank_budget"] == 4
        for filename in (
            "eva_state.pt", "rank_pattern.json", "explained_variance.csv",
            "metadata.json",
        ):
            assert os.path.isfile(os.path.join(output_dir, filename))
    print("[PASS] activation PCA hooks, shared q/v inputs, and EVA state schema")

    # NetLLM presents 30 trajectory states per sample. A 24-component
    # low-rank PCA internally requests q=48, so the first fit must combine two
    # batches. This reproduces the real Llama smoke-test failure condition.
    bootstrap_model = TinyLlama(layers=1, hidden=64)
    bootstrap_collector = EvaActivationCollector(
        bootstrap_model,
        target_modules=("q_proj", "v_proj"),
        max_components=24,
        similarity_threshold=0.0,
        expected_llama_layers=1,
        pca_factory=lambda n: FakeIncrementalPCA(n),
    )
    bootstrap_batch = torch.randn(1, 30, 64)
    bootstrap_state = bootstrap_collector.collect(
        [bootstrap_batch, bootstrap_batch.clone(), bootstrap_batch.clone()],
        forward_batch=bootstrap_model,
        rank_budget=2,
        min_rank=0,
        max_rank=24,
        min_batches=2,
        max_batches=3,
    )
    assert bootstrap_state["processed_batches"] == 3
    assert all(hook.update_count == 2 for hook in bootstrap_collector.hooks.values())
    print("[PASS] low-rank PCA q=2k bootstrap buffers short NetLLM batches")

    rows = diagnostic_rows(state)
    diagnostics = summarize(state, rows)
    assert len(rows) == 4
    assert diagnostics["total_rank_budget"] == 4
    assert diagnostics["positive_rank_module_count"] == sum(
        int(rank) > 0 for rank in state["rank_pattern"].values()
    )
    print("[PASS] EVA rank/variance diagnostic statistics")

    spec = eva_lora_spec(state)
    wrapped_model = build_low_rank_model(
        TinyLlama(layers=2, hidden=8),
        plm_type="llama",
        rank=1,
        task_type=None,
        eva_state=state,
    )
    summary = wrapped_model.eva_initialization_summary
    assert wrapped_model.eva_state is state
    assert summary["initialized_modules"] == len(spec["target_modules"])
    assert summary["total_rank_budget"] == 4
    for state_name, wrapped_name in summary["module_mapping"].items():
        module = dict(wrapped_model.named_modules())[wrapped_name]
        expected = state["components"][state_name].to(module.lora_A["default"].weight)
        assert torch.equal(module.lora_A["default"].weight, expected)
        assert torch.count_nonzero(module.lora_B["default"].weight) == 0

    output = wrapped_model(torch.randn(2, 4, 8))
    loss = output.square().mean()
    loss.backward()
    trainable_grads = [
        parameter.grad
        for parameter in wrapped_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads and all(torch.isfinite(grad).all() for grad in trainable_grads)
    assert any(
        "lora_B" in name and parameter.grad is not None
        for name, parameter in wrapped_model.named_parameters()
    )
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_state = os.path.join(output_dir, "eva_state.pt")
        torch.save(wrapped_model.eva_state, checkpoint_state)
        assert validate_eva_state(torch.load(checkpoint_state, map_location="cpu"))
    print("[PASS] PEFT fixed-rank mapping, EVA lora_A/B initialization, and backward")
    print("All EVA precomputation checks completed.")


if __name__ == "__main__":
    main()
