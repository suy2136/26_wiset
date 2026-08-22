"""Inference-only compaction for a trained NBS AdaLoRA adapter.

NBS training deliberately keeps PEFT AdaLoRA's physical ``init_r`` tensors
and selects components with masks over ``lora_E``.  That representation is
convenient for reallocation, but dense inference still pays for every
physical slot.  This module converts the *final* masked adapter into smaller
fixed LoRA projections without changing the training allocator.

The conversion is intentionally separate from :mod:`models.rank_allocator`.
It mutates an already-loaded model for inference; the source checkpoint on
disk remains unchanged.  Saving/loading the derived compact checkpoint and
the user-facing CLI are handled by later integration stages.
"""

from collections import OrderedDict
from dataclasses import dataclass
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NBSLayerCompactionSpec:
    """The exact active topology and compact factors for one NBS layer."""

    name: str
    active_indices: tuple
    physical_rank: int
    compact_rank: int
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    adapter_scale: float


class CompactLoRALinear(nn.Module):
    """Frozen base projection plus an inference-only variable-rank LoRA.

    ``lora_a`` already contains AdaLoRA's active ``lora_E`` values.  Keep the
    final ``scaling / ranknum`` multiply after the two compact matmuls, matching
    the original AdaLoRA operation order as closely as possible.
    """

    def __init__(self, source, lora_a, lora_b, adapter_scale, dropout):
        super().__init__()
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError("compact LoRA factors must be matrices")
        if lora_a.shape[0] != lora_b.shape[1]:
            raise ValueError("compact LoRA A/B ranks do not match")

        self.in_features = int(lora_a.shape[1])
        self.out_features = int(lora_b.shape[0])
        self.compact_rank = int(lora_a.shape[0])
        self.fan_in_fan_out = bool(getattr(source, "fan_in_fan_out", False))

        # Reuse the frozen base parameters instead of duplicating the LLM.
        self.weight = source.weight
        source_bias = getattr(source, "bias", None)
        if source_bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = source_bias

        # Buffers make the derived module explicitly inference-only while still
        # following model.to(device/dtype) and participating in state_dict().
        self.register_buffer("lora_a", lora_a.detach().clone())
        self.register_buffer("lora_b", lora_b.detach().clone())
        self.register_buffer(
            "adapter_scale",
            torch.as_tensor(adapter_scale, device=lora_b.device, dtype=lora_b.dtype),
        )
        self.lora_dropout = copy.deepcopy(dropout)

    def forward(self, x):
        base_weight = self.weight.T if self.fan_in_fan_out else self.weight
        result = F.linear(x.to(base_weight.dtype), base_weight, self.bias)
        adapter_input = self.lora_dropout(x.to(self.lora_a.dtype))
        delta = (
            F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
            * self.adapter_scale
        )
        return result + delta.to(result.dtype)


def _parameter_tensor(container, adapter_name):
    value = container[adapter_name]
    return value.weight if hasattr(value, "weight") else value


def _find_nbs_layers(model, adapter_name):
    layers = OrderedDict()
    for name, module in model.named_modules():
        required = ("lora_A", "lora_B", "lora_E", "ranknum", "scaling")
        if not all(hasattr(module, attribute) for attribute in required):
            continue
        if not all(
            adapter_name in getattr(module, attribute)
            for attribute in ("lora_A", "lora_B", "lora_E", "ranknum")
        ):
            continue
        layers[name] = module
    if not layers:
        raise ValueError("no NBS AdaLoRA layers were found for compaction")
    return layers


def _mask_sources(model, allocator_state):
    """Return final masks/ranks, preferring the live restored allocator."""
    allocator = getattr(model, "nash_rank_allocator", None)
    if allocator is not None:
        return allocator.masks, allocator.ranks, "live_allocator"
    if allocator_state is not None:
        masks = allocator_state.get("masks", {})
        ranks = allocator_state.get("ranks", {})
        return masks, ranks, "allocator_state"
    raise ValueError(
        "exact NBS masks are required; restore nash_rank_allocator.pt or pass "
        "its state as allocator_state"
    )


def extract_nbs_compaction_specs(model, adapter_name="default", allocator_state=None):
    """Extract exact active slots and mathematically equivalent compact factors.

    For an active index set ``S``, the original AdaLoRA delta is::

        B[:, S] @ diag(E[S]) @ A[S, :] * scaling / ranknum

    We preserve it by folding ``E`` into ``A_compact`` and retaining the final
    scalar multiply after ``B_compact``, which minimizes FP16 reassociation.
    Masks are mandatory because
    an active component can legitimately have a zero-valued ``lora_E``; using
    nonzero values alone would then infer the wrong topology.
    """
    layers = _find_nbs_layers(model, adapter_name)
    masks, ranks, source = _mask_sources(model, allocator_state)
    specs = OrderedDict()

    for name, module in layers.items():
        if name not in masks:
            raise ValueError(f"NBS mask is missing layer {name}")
        lora_a = _parameter_tensor(module.lora_A, adapter_name).detach()
        lora_b = _parameter_tensor(module.lora_B, adapter_name).detach()
        lora_e = _parameter_tensor(module.lora_E, adapter_name).detach().reshape(-1)
        mask = torch.as_tensor(masks[name], device=lora_e.device).reshape(-1).bool()
        if mask.numel() != lora_e.numel():
            raise ValueError(f"NBS mask shape mismatch for {name}")

        active = torch.nonzero(mask, as_tuple=False).reshape(-1)
        expected_rank = ranks.get(name)
        if expected_rank is not None and active.numel() != int(expected_rank):
            raise ValueError(
                f"NBS mask/rank mismatch for {name}: mask has {active.numel()} "
                f"slots but allocator reports {expected_rank}"
            )
        if active.numel() == 0:
            raise ValueError(f"NBS layer {name} has no active slots")
        if lora_a.shape[0] != lora_e.numel() or lora_b.shape[1] != lora_e.numel():
            raise ValueError(f"AdaLoRA factor shape mismatch for {name}")

        ranknum = _parameter_tensor(module.ranknum, adapter_name).detach()
        effective_scale = torch.as_tensor(
            module.scaling[adapter_name], device=lora_b.device, dtype=lora_b.dtype
        ) / (ranknum.to(device=lora_b.device, dtype=lora_b.dtype) + 1e-5)
        selected_e = lora_e.index_select(0, active).to(lora_b.dtype)
        # Preserve AdaLoRA's original operation order: E multiplies A before
        # the first matmul, while scaling/ranknum remains after the B matmul.
        compact_a = (
            lora_a.index_select(0, active)
            * selected_e.to(lora_a.dtype).reshape(-1, 1)
        ).clone()
        compact_b = lora_b.index_select(1, active).clone()
        specs[name] = NBSLayerCompactionSpec(
            name=name,
            active_indices=tuple(int(index) for index in active.cpu().tolist()),
            physical_rank=int(lora_e.numel()),
            compact_rank=int(active.numel()),
            lora_a=compact_a,
            lora_b=compact_b,
            adapter_scale=float(effective_scale.item()),
        )

    return specs, source


def _replace_submodule(model, name, replacement):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name in parent._modules:
        parent._modules[child_name] = replacement
        return
    raise ValueError(f"could not replace NBS layer {name}")


def validate_nbs_compaction_factors(model, specs, adapter_name="default",
                                    trials=2, rtol=1e-5, atol=1e-5):
    """Verify each compact adapter delta against its masked AdaLoRA source."""
    if trials <= 0:
        raise ValueError("compaction validation trials must be positive")
    named_modules = dict(model.named_modules())
    rows = OrderedDict()
    for name, spec in specs.items():
        source = named_modules.get(name)
        if source is None:
            raise ValueError(f"compaction source layer is missing: {name}")
        source_a = _parameter_tensor(source.lora_A, adapter_name).detach().float()
        source_b = _parameter_tensor(source.lora_B, adapter_name).detach().float()
        source_e = _parameter_tensor(source.lora_E, adapter_name).detach().float()
        ranknum = _parameter_tensor(source.ranknum, adapter_name).detach().float()
        scale = float(source.scaling[adapter_name]) / float(ranknum.item() + 1e-5)
        max_abs_error = 0.0
        max_rel_error = 0.0
        active = torch.tensor(spec.active_indices, device=source_a.device)
        for _ in range(int(trials)):
            x = torch.randn(3, source_a.shape[1], device=source_a.device)
            expected = (
                x @ (source_a.index_select(0, active)
                     * source_e.reshape(-1).index_select(0, active).reshape(-1, 1)).T
                @ source_b.index_select(1, active).T
            ) * scale
            actual = (
                x @ spec.lora_a.float().T @ spec.lora_b.float().T
            ) * float(spec.adapter_scale)
            difference = (actual - expected).abs()
            max_abs_error = max(max_abs_error, float(difference.max().item()))
            denominator = expected.abs().clamp_min(float(atol))
            max_rel_error = max(
                max_rel_error, float((difference / denominator).max().item())
            )
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        rows[name] = {
            "compact_rank": spec.compact_rank,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
        }
    return {
        "passed": True,
        "trials_per_layer": int(trials),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max(row["max_abs_error"] for row in rows.values()),
        "max_rel_error": max(row["max_rel_error"] for row in rows.values()),
        "layers": dict(rows),
    }


def _atomic_torch_save(payload, path):
    temporary = path + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_json_save(payload, path):
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_nbs_compact_checkpoint(model, output_dir, modules_except_plm_state=None,
                                adapter_name="default", allocator_state=None,
                                source_checkpoint=None, factor_validation=None):
    """Save a standalone compact-adapter derivative without touching its source."""
    specs, mask_source = extract_nbs_compaction_specs(
        model, adapter_name=adapter_name, allocator_state=allocator_state
    )
    if factor_validation is None:
        factor_validation = validate_nbs_compaction_factors(
            model, specs, adapter_name=adapter_name
        )
    os.makedirs(output_dir, exist_ok=True)
    compact_state = {
        "format_version": 2,
        "adapter_name": adapter_name,
        "layers": {
            name: {
                "active_indices": list(spec.active_indices),
                "physical_rank": spec.physical_rank,
                "compact_rank": spec.compact_rank,
                "lora_a": spec.lora_a.detach().cpu(),
                "lora_b": spec.lora_b.detach().cpu(),
                "adapter_scale": spec.adapter_scale,
            }
            for name, spec in specs.items()
        },
    }
    _atomic_torch_save(
        compact_state, os.path.join(output_dir, "compact_adapter.pt")
    )
    if modules_except_plm_state is not None:
        cpu_state = {
            name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in modules_except_plm_state.items()
        }
        _atomic_torch_save(
            cpu_state, os.path.join(output_dir, "modules_except_plm.bin")
        )
    rank_pattern = {
        name: spec.compact_rank for name, spec in specs.items()
    }
    metadata = {
        "format_version": 2,
        "checkpoint_type": "nbs_compact_inference",
        "source_checkpoint": (
            None if source_checkpoint is None else os.path.abspath(source_checkpoint)
        ),
        "mask_source": mask_source,
        "adapter_name": adapter_name,
        "module_count": len(specs),
        "physical_rank_total_before": sum(
            spec.physical_rank for spec in specs.values()
        ),
        "compact_rank_total": sum(rank_pattern.values()),
        "rank_pattern": rank_pattern,
        "factor_validation": factor_validation,
    }
    _atomic_json_save(
        metadata, os.path.join(output_dir, "compaction_metadata.json")
    )
    _atomic_json_save(
        {
            "rank_pattern": rank_pattern,
            "total_rank_budget": sum(rank_pattern.values()),
        },
        os.path.join(output_dir, "rank_pattern.json"),
    )
    return metadata


def _specs_from_compact_state(model, state, adapter_name):
    layers = _find_nbs_layers(model, adapter_name)
    saved_layers = state.get("layers", {})
    if set(saved_layers) != set(layers):
        missing = sorted(set(layers).difference(saved_layers))
        unexpected = sorted(set(saved_layers).difference(layers))
        raise ValueError(
            f"compact checkpoint layer mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    specs = OrderedDict()
    for name, source in layers.items():
        row = saved_layers[name]
        source_a = _parameter_tensor(source.lora_A, adapter_name)
        source_b = _parameter_tensor(source.lora_B, adapter_name)
        compact_a = row["lora_a"].to(device=source_a.device, dtype=source_a.dtype)
        compact_b = row["lora_b"].to(device=source_b.device, dtype=source_b.dtype)
        compact_rank = int(row["compact_rank"])
        if compact_a.shape != (compact_rank, source_a.shape[1]):
            raise ValueError(f"compact lora_A shape mismatch for {name}")
        if compact_b.shape != (source_b.shape[0], compact_rank):
            raise ValueError(f"compact lora_B shape mismatch for {name}")
        specs[name] = NBSLayerCompactionSpec(
            name=name,
            active_indices=tuple(int(index) for index in row["active_indices"]),
            physical_rank=int(row["physical_rank"]),
            compact_rank=compact_rank,
            lora_a=compact_a,
            lora_b=compact_b,
            adapter_scale=float(row.get("adapter_scale", 1.0)),
        )
    return specs


def _apply_compaction_specs(model, specs, adapter_name, mask_source):
    named_modules = dict(model.named_modules())
    report_layers = OrderedDict()
    for name, spec in specs.items():
        source = named_modules[name]
        dropout = source.lora_dropout[adapter_name]
        replacement = CompactLoRALinear(
            source=source,
            lora_a=spec.lora_a,
            lora_b=spec.lora_b,
            adapter_scale=spec.adapter_scale,
            dropout=dropout,
        )
        replacement.train(source.training)
        _replace_submodule(model, name, replacement)
        report_layers[name] = {
            "physical_rank_before": spec.physical_rank,
            "compact_rank": spec.compact_rank,
            "active_indices": list(spec.active_indices),
        }

    if hasattr(model, "nash_rank_allocator"):
        model.nash_rank_allocator = None
    model.nbs_compaction_report = {
        "format_version": 1,
        "adapter_name": adapter_name,
        "mask_source": mask_source,
        "module_count": len(report_layers),
        "physical_rank_total_before": sum(
            row["physical_rank_before"] for row in report_layers.values()
        ),
        "compact_rank_total": sum(row["compact_rank"] for row in report_layers.values()),
        "layers": dict(report_layers),
    }
    return model.nbs_compaction_report


def load_nbs_compact_checkpoint(model, checkpoint_dir, adapter_name="default"):
    """Load compact factors into a freshly constructed NBS PEFT model."""
    state_path = os.path.join(checkpoint_dir, "compact_adapter.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"compact adapter not found: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    if int(state.get("format_version", 0)) not in (1, 2):
        raise ValueError("unsupported NBS compact checkpoint format")
    if state.get("adapter_name", adapter_name) != adapter_name:
        raise ValueError("compact checkpoint adapter name mismatch")
    specs = _specs_from_compact_state(model, state, adapter_name)
    return _apply_compaction_specs(
        model, specs, adapter_name=adapter_name, mask_source="compact_checkpoint"
    )


def compact_nbs_model_for_inference(model, adapter_name="default", allocator_state=None):
    """Replace masked SVDLinear layers with compact fixed-LoRA projections.

    This operation is deliberately explicit and in-place.  Call it only on an
    inference model loaded from a preserved source checkpoint.  The returned
    report records the exact topology used for later equivalence and latency
    validation.
    """
    specs, mask_source = extract_nbs_compaction_specs(
        model, adapter_name=adapter_name, allocator_state=allocator_state
    )
    return _apply_compaction_specs(
        model, specs, adapter_name=adapter_name, mask_source=mask_source
    )


def write_nbs_compaction_validation(checkpoint_dir, factor_validation,
                                    full_output_validation):
    """Persist factor-level and end-to-end validation beside the derivative."""
    payload = {
        "factor_validation": factor_validation,
        "full_output_validation": full_output_validation,
        "passed": bool(
            factor_validation.get("passed")
            and full_output_validation.get("passed")
        ),
    }
    _atomic_json_save(
        payload, os.path.join(checkpoint_dir, "equivalence_report.json")
    )
    return payload
