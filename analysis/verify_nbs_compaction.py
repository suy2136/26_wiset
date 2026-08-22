"""Synthetic equivalence checks for inference-only NBS compaction."""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.nbs_compaction import (  # noqa: E402
    CompactLoRALinear,
    compact_nbs_model_for_inference,
    extract_nbs_compaction_specs,
    load_nbs_compact_checkpoint,
    save_nbs_compact_checkpoint,
    validate_nbs_compaction_factors,
)


class SyntheticSVDLinear(nn.Module):
    def __init__(self, in_features=7, out_features=5, physical_rank=6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=False)
        self.lora_A = nn.ParameterDict({
            "default": nn.Parameter(torch.randn(physical_rank, in_features))
        })
        self.lora_B = nn.ParameterDict({
            "default": nn.Parameter(torch.randn(out_features, physical_rank))
        })
        self.lora_E = nn.ParameterDict({
            "default": nn.Parameter(torch.randn(physical_rank, 1))
        })
        self.ranknum = nn.ParameterDict({
            "default": nn.Parameter(torch.tensor(float(physical_rank)), requires_grad=False)
        })
        self.scaling = {"default": 3.5}
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.0)})

    def forward(self, x):
        a = self.lora_A["default"]
        b = self.lora_B["default"]
        e = self.lora_E["default"]
        ranknum = self.ranknum["default"] + 1e-5
        return (
            F.linear(x, self.weight, self.bias)
            + (x @ (a * e).T @ b.T) * self.scaling["default"] / ranknum
        )


class SyntheticAllocator:
    def __init__(self, mask):
        self.masks = {"proj": mask}
        self.ranks = {"proj": int(mask.bool().sum())}


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = SyntheticSVDLinear()
        # Non-prefix active indices verify that compaction follows the exact
        # NBS topology rather than slicing the first r physical components.
        mask = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)
        with torch.no_grad():
            self.proj.lora_E["default"].mul_(mask.reshape(-1, 1))
        self.nash_rank_allocator = SyntheticAllocator(mask)

    def forward(self, x):
        return self.proj(x)


def main():
    torch.manual_seed(7)
    model = SyntheticModel().eval()
    inputs = torch.randn(4, 3, 7)
    expected = model(inputs).detach()

    specs, source = extract_nbs_compaction_specs(model)
    assert source == "live_allocator"
    assert specs["proj"].active_indices == (1, 3, 5)
    assert specs["proj"].physical_rank == 6
    assert specs["proj"].compact_rank == 3
    factor_report = validate_nbs_compaction_factors(model, specs)
    assert factor_report["passed"]

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        metadata = save_nbs_compact_checkpoint(model, checkpoint_dir)
        assert metadata["compact_rank_total"] == 3
        assert os.path.isfile(os.path.join(checkpoint_dir, "compact_adapter.pt"))
        assert os.path.isfile(os.path.join(checkpoint_dir, "rank_pattern.json"))
        restored = SyntheticModel().eval()
        with torch.no_grad():
            restored.proj.weight.copy_(model.proj.weight)
            restored.proj.bias.copy_(model.proj.bias)
        load_report = load_nbs_compact_checkpoint(restored, checkpoint_dir)
        restored_actual = restored(inputs).detach()
        torch.testing.assert_close(restored_actual, expected, rtol=1e-5, atol=1e-5)
        assert load_report["compact_rank_total"] == 3

    report = compact_nbs_model_for_inference(model)
    actual = model(inputs).detach()
    assert isinstance(model.proj, CompactLoRALinear)
    assert model.proj.compact_rank == 3
    assert model.nash_rank_allocator is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert report["physical_rank_total_before"] == 6
    assert report["compact_rank_total"] == 3
    print("[PASS] exact non-prefix active-slot extraction")
    print("[PASS] lora_E and AdaLoRA scaling absorption")
    print("[PASS] separate compact checkpoint save/load")
    print("[PASS] compact fixed-LoRA forward equivalence")


if __name__ == "__main__":
    main()
