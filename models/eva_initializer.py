"""EVA activation statistics and explained-variance rank allocation.

This module is intentionally independent from the existing LoRA, AdaLoRA, and
NBS construction paths.  It implements the *pre-training* stages of Explained
Variance Adaptation (EVA): collect target-layer input activations, estimate
principal components, and allocate a fixed global rank budget.

The design follows the MIT-licensed reference implementation by JKU Linz:
https://github.com/ml-jku/EVA (``src/svd.py`` and ``src/utils.py``).
"""

from collections import OrderedDict
import csv
import json
import os
import re

import torch
from torch import nn


EVA_STATE_VERSION = 1
EVA_METRICS = ("raw", "ratio", "sum", "max")
_LLAMA_QV_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<module>q_proj|v_proj)$"
)


def _matches_target(name, target):
    return name == target or name.endswith("." + target)


def discover_eva_target_modules(model, target_modules=("q_proj", "v_proj")):
    """Return exact names and Linear modules selected for EVA statistics."""
    targets = tuple(str(target) for target in target_modules)
    if not targets:
        raise ValueError("target_modules must not be empty")
    discovered = OrderedDict(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(_matches_target(name, target) for target in targets)
    )
    if not discovered:
        raise ValueError(
            "EVA found no target Linear modules matching " + repr(targets)
        )
    return discovered


def validate_llama_qv_module_keys(modules, expected_layers=32):
    """Validate one q_proj and v_proj for every expected Llama layer."""
    names = list(modules.keys()) if hasattr(modules, "keys") else list(modules)
    parsed = {}
    invalid = []
    for name in names:
        match = _LLAMA_QV_PATTERN.search(name)
        if match is None:
            invalid.append(name)
            continue
        key = (int(match.group("layer")), match.group("module"))
        if key in parsed:
            raise ValueError(
                f"EVA target key {key} matched more than once: "
                f"{parsed[key]!r}, {name!r}"
            )
        parsed[key] = name
    if invalid:
        raise ValueError(f"non-Llama q/v EVA target names: {invalid[:5]}")

    expected = {
        (layer, module_type)
        for layer in range(int(expected_layers))
        for module_type in ("q_proj", "v_proj")
    }
    missing = sorted(expected.difference(parsed))
    unexpected = sorted(set(parsed).difference(expected))
    if missing or unexpected:
        raise ValueError(
            "EVA Llama q/v mapping mismatch: expected {} modules, found {}; "
            "missing={}, unexpected={}".format(
                len(expected), len(parsed), missing[:8], unexpected[:8]
            )
        )
    return OrderedDict((parsed[key], key) for key in sorted(parsed))


def _resolve_rank_bound(spec, name, default, label):
    if spec is None:
        value = default
    elif isinstance(spec, dict):
        value = spec.get(name, default)
    else:
        value = spec
    value = int(value)
    if value < 0:
        raise ValueError(f"{label} must be non-negative for {name}")
    return value


def allocate_eva_ranks(
    explained_variance,
    rank_budget,
    min_rank=0,
    max_rank=None,
):
    """Allocate a global budget by greedily taking the next largest variance.

    A prefix constraint is enforced for each layer.  Therefore selecting its
    k-th component necessarily includes components 1..k-1, matching PCA rank.
    """
    if not explained_variance:
        raise ValueError("explained_variance must not be empty")
    names = list(explained_variance)
    values = OrderedDict()
    minima = OrderedDict()
    maxima = OrderedDict()
    for name in names:
        tensor = torch.as_tensor(explained_variance[name], dtype=torch.float64).flatten()
        if tensor.numel() == 0:
            raise ValueError(f"EVA layer {name} has no explained-variance components")
        if not torch.isfinite(tensor).all() or (tensor < 0).any():
            raise ValueError(f"EVA layer {name} has invalid explained variance")
        # Incremental PCA should already return descending values.  Sorting here
        # makes a loaded external checkpoint safe and deterministic.
        tensor = torch.sort(tensor, descending=True).values
        values[name] = tensor
        minima[name] = _resolve_rank_bound(min_rank, name, 0, "min_rank")
        maxima[name] = _resolve_rank_bound(
            max_rank, name, tensor.numel(), "max_rank"
        )
        if maxima[name] > tensor.numel():
            raise ValueError(
                f"max_rank {maxima[name]} exceeds {tensor.numel()} EVA components "
                f"for {name}"
            )
        if minima[name] > maxima[name]:
            raise ValueError(f"min_rank exceeds max_rank for {name}")

    rank_budget = int(rank_budget)
    minimum_budget = sum(minima.values())
    maximum_budget = sum(maxima.values())
    if not minimum_budget <= rank_budget <= maximum_budget:
        raise ValueError(
            f"rank_budget {rank_budget} is outside feasible range "
            f"[{minimum_budget}, {maximum_budget}]"
        )

    ranks = OrderedDict((name, minima[name]) for name in names)
    remaining = rank_budget - minimum_budget
    tie_breaker = {name: index for index, name in enumerate(names)}
    heap = []
    for name in names:
        if ranks[name] < maxima[name]:
            next_score = float(values[name][ranks[name]].item())
            heap.append((-next_score, tie_breaker[name], name))
    import heapq
    heapq.heapify(heap)

    for _ in range(remaining):
        if not heap:
            raise RuntimeError("EVA rank heap exhausted before filling the budget")
        _, _, name = heapq.heappop(heap)
        ranks[name] += 1
        if ranks[name] < maxima[name]:
            next_score = float(values[name][ranks[name]].item())
            heapq.heappush(heap, (-next_score, tie_breaker[name], name))
    return ranks


class _EvaPCAHook:
    def __init__(self, name, n_components, similarity_threshold, pca_factory):
        self.name = name
        self.n_components = int(n_components)
        self.similarity_threshold = float(similarity_threshold)
        self.pca = pca_factory(self.n_components)
        self.converged = torch.zeros(self.n_components, dtype=torch.bool)
        self.update_count = 0

    def __call__(self, module, inputs, output):
        states = inputs[0] if isinstance(inputs, tuple) else inputs
        states = states.detach().reshape(-1, states.shape[-1]).to(torch.float32)
        if states.shape[0] < self.n_components:
            return
        previous = getattr(self.pca, "components_", None)
        if previous is not None:
            previous = previous.detach().clone()
        self.pca.partial_fit(states)
        self.update_count += 1
        current = getattr(self.pca, "components_", None)
        if previous is not None and current is not None:
            previous = previous.reshape(self.n_components, -1)
            current = current.reshape(self.n_components, -1)
            similarity = torch.nn.functional.cosine_similarity(current, previous)
            self.converged = similarity >= self.similarity_threshold


def _default_pca_factory(n_components):
    try:
        from torch_incremental_pca import IncrementalPCA
    except ImportError as exc:
        raise ImportError(
            "EVA precomputation requires torch-incremental-pca. Install the "
            "packages in requirements_eva.txt."
        ) from exc
    return IncrementalPCA(n_components=n_components, copy=True, lowrank=True)


class EvaActivationCollector:
    """Collect task-specific input PCA statistics for fixed EVA LoRA ranks."""

    def __init__(
        self,
        model,
        target_modules=("q_proj", "v_proj"),
        max_components=24,
        similarity_threshold=0.99,
        share_llama_qv_inputs=True,
        expected_llama_layers=None,
        pca_factory=None,
    ):
        self.model = model
        self.modules = discover_eva_target_modules(model, target_modules)
        if expected_llama_layers is not None:
            validate_llama_qv_module_keys(
                self.modules, expected_layers=expected_llama_layers
            )
        self.max_components = int(max_components)
        if self.max_components <= 0:
            raise ValueError("max_components must be positive")
        if not 0.0 <= float(similarity_threshold) <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")

        self.representative_for = OrderedDict()
        representatives = OrderedDict()
        for name, module in self.modules.items():
            match = _LLAMA_QV_PATTERN.search(name)
            group = (
                name.rsplit(".", 1)[0]
                if share_llama_qv_inputs and match is not None
                else name
            )
            if group not in representatives:
                representatives[group] = name
            self.representative_for[name] = representatives[group]

        factory = pca_factory or _default_pca_factory
        self.hooks = OrderedDict()
        self.handles = []
        for representative in OrderedDict.fromkeys(self.representative_for.values()):
            hook = _EvaPCAHook(
                representative,
                self.max_components,
                similarity_threshold,
                factory,
            )
            self.hooks[representative] = hook
            self.handles.append(
                self.modules[representative].register_forward_hook(hook)
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _metric_values(self, hook, metric):
        raw = torch.as_tensor(hook.pca.explained_variance_).detach().float()
        if metric == "raw":
            return raw
        if metric == "ratio":
            ratio = getattr(hook.pca, "explained_variance_ratio_", None)
            return (
                torch.as_tensor(ratio).detach().float()
                if ratio is not None
                else raw / raw.sum().clamp_min(torch.finfo(raw.dtype).eps)
            )
        if metric == "sum":
            return raw / raw.sum().clamp_min(torch.finfo(raw.dtype).eps)
        if metric == "max":
            return raw / raw.max().clamp_min(torch.finfo(raw.dtype).eps)
        raise ValueError(f"unsupported EVA metric: {metric}")

    def explained_variance(self, metric="ratio"):
        if metric not in EVA_METRICS:
            raise ValueError(f"metric must be one of {EVA_METRICS}")
        result = OrderedDict()
        for name, representative in self.representative_for.items():
            hook = self.hooks[representative]
            if not hasattr(hook.pca, "components_"):
                raise RuntimeError(f"EVA PCA has no components for {representative}")
            result[name] = self._metric_values(hook, metric).cpu()
        return result

    def selected_components_converged(self, ranks):
        for name, rank in ranks.items():
            if rank <= 0:
                continue
            representative = self.representative_for[name]
            hook = self.hooks[representative]
            if hook.update_count < 2 or not bool(torch.all(hook.converged[:rank])):
                return False
        return True

    @torch.no_grad()
    def collect(
        self,
        data_loader,
        forward_batch,
        rank_budget,
        metric="ratio",
        min_rank=0,
        max_rank=None,
        min_batches=2,
        max_batches=128,
        require_convergence=True,
    ):
        """Run calibration forwards until selected PCA components converge."""
        if min_batches <= 0 or max_batches < min_batches:
            raise ValueError("require 0 < min_batches <= max_batches")
        was_training = self.model.training
        self.model.eval()
        ranks = None
        processed = 0
        try:
            for processed, batch in enumerate(data_loader, start=1):
                forward_batch(batch)
                if processed < min_batches:
                    if processed >= max_batches:
                        break
                    continue
                try:
                    values = self.explained_variance(metric)
                except RuntimeError:
                    if processed >= max_batches:
                        break
                    continue
                ranks = allocate_eva_ranks(
                    values,
                    rank_budget=rank_budget,
                    min_rank=min_rank,
                    max_rank=max_rank,
                )
                if self.selected_components_converged(ranks):
                    break
                if processed >= max_batches:
                    break
        finally:
            self.model.train(was_training)
            self.close()

        if ranks is None:
            raise RuntimeError(
                "EVA did not obtain PCA components; increase token count or batches"
            )
        converged = self.selected_components_converged(ranks)
        if require_convergence and not converged:
            raise RuntimeError(
                f"EVA selected components did not converge after {processed} batches"
            )
        return self.build_state(ranks, metric, processed, converged)

    def build_state(self, ranks, metric, processed_batches, converged):
        components = OrderedDict()
        explained = OrderedDict()
        all_metrics = {
            metric_name: self.explained_variance(metric_name)
            for metric_name in EVA_METRICS
        }
        for name, representative in self.representative_for.items():
            rank = int(ranks[name])
            matrix = self.hooks[representative].pca.components_[:rank]
            components[name] = matrix.detach().float().cpu().clone()
            explained[name] = {
                metric_name: all_metrics[metric_name][name].cpu().clone()
                for metric_name in EVA_METRICS
            }
        state = {
            "version": EVA_STATE_VERSION,
            "method": "eva",
            "rank_pattern": dict(ranks),
            "total_rank_budget": int(sum(ranks.values())),
            "metric": metric,
            "max_components": self.max_components,
            "components": components,
            "explained_variance": explained,
            "representative_for": dict(self.representative_for),
            "processed_batches": int(processed_batches),
            "converged": bool(converged),
        }
        validate_eva_state(state, expected_names=self.modules.keys())
        return state


def validate_eva_state(state, expected_names=None):
    """Validate rank budget, component shapes, and exact module-key coverage."""
    if state.get("method") != "eva":
        raise ValueError("not an EVA state")
    ranks = state.get("rank_pattern")
    components = state.get("components")
    if not isinstance(ranks, dict) or not isinstance(components, dict):
        raise ValueError("EVA state requires rank_pattern and components dictionaries")
    if set(ranks) != set(components):
        raise ValueError("EVA rank_pattern/component keys do not match")
    if expected_names is not None and set(ranks) != set(expected_names):
        missing = sorted(set(expected_names).difference(ranks))
        extra = sorted(set(ranks).difference(expected_names))
        raise ValueError(f"EVA module-key mismatch: missing={missing}, extra={extra}")
    for name, rank in ranks.items():
        rank = int(rank)
        matrix = torch.as_tensor(components[name])
        if rank < 0 or matrix.ndim != 2 or matrix.shape[0] != rank:
            raise ValueError(
                f"invalid EVA component shape for {name}: rank={rank}, "
                f"shape={tuple(matrix.shape)}"
            )
    total = sum(int(rank) for rank in ranks.values())
    if total != int(state.get("total_rank_budget", -1)):
        raise ValueError(
            f"EVA state rank sum {total} does not match recorded budget "
            f"{state.get('total_rank_budget')}"
        )
    return True


def eva_lora_spec(state):
    """Build PEFT's fixed LoRA specification from a validated EVA state.

    PEFT cannot construct a rank-zero LoRA module. EVA may assign zero rank
    to a layer, so only positive-rank exact names are passed to PEFT. This
    prevents its usual q_proj/v_proj suffix matching from adapting rank-zero
    layers accidentally.
    """
    validate_eva_state(state)
    rank_pattern = OrderedDict(
        (str(name), int(rank))
        for name, rank in state["rank_pattern"].items()
        if int(rank) > 0
    )
    if not rank_pattern:
        raise ValueError("EVA state allocates zero rank to every target layer")
    return {
        "rank_pattern": dict(rank_pattern),
        "target_modules": list(rank_pattern),
        "total_rank_budget": int(sum(rank_pattern.values())),
    }


@torch.no_grad()
def initialize_eva_lora_weights(model, state, adapter_name="default"):
    """Copy EVA PCA directions into PEFT LoRA A and zero-initialize LoRA B."""
    spec = eva_lora_spec(state)
    expected = set(spec["target_modules"])
    matches = {}
    adapted = {}
    for wrapped_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or adapter_name not in module.lora_A:
            continue
        adapted[wrapped_name] = module
        candidates = [
            name for name in expected
            if wrapped_name == name or wrapped_name.endswith("." + name)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"PEFT EVA module {wrapped_name!r} maps to {len(candidates)} "
                f"state keys: {candidates}"
            )
        state_name = candidates[0]
        if state_name in matches:
            raise ValueError(
                f"EVA state key {state_name!r} matched multiple PEFT modules"
            )
        matches[state_name] = wrapped_name

        lora_a = module.lora_A[adapter_name]
        lora_b = module.lora_B[adapter_name]
        components = torch.as_tensor(state["components"][state_name])
        if tuple(lora_a.weight.shape) != tuple(components.shape):
            raise ValueError(
                f"EVA component shape mismatch for {state_name}: "
                f"state={tuple(components.shape)}, "
                f"lora_A={tuple(lora_a.weight.shape)}"
            )
        lora_a.weight.copy_(
            components.to(device=lora_a.weight.device, dtype=lora_a.weight.dtype)
        )
        lora_b.weight.zero_()

    missing = sorted(expected.difference(matches))
    if missing:
        raise ValueError(f"EVA state keys did not map to PEFT modules: {missing[:8]}")
    if len(adapted) != len(expected):
        raise ValueError(
            f"EVA expected {len(expected)} adapted modules but PEFT created "
            f"{len(adapted)}"
        )
    return {
        "initialized_modules": len(matches),
        "total_rank_budget": spec["total_rank_budget"],
        "module_mapping": dict(matches),
    }


def save_eva_state(output_dir, state, metadata=None):
    """Write a reusable EVA tensor checkpoint and human-readable diagnostics."""
    validate_eva_state(state)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, "eva_state.pt"))
    with open(os.path.join(output_dir, "rank_pattern.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "eva",
                "total_rank_budget": state["total_rank_budget"],
                "rank_pattern": state["rank_pattern"],
            },
            handle,
            indent=2,
        )
    rows = []
    for name, metrics in state["explained_variance"].items():
        rank = int(state["rank_pattern"][name])
        for index in range(state["max_components"]):
            rows.append({
                "layer_name": name,
                "component": index + 1,
                "allocated": int(index < rank),
                **{
                    metric: float(metrics[metric][index])
                    for metric in EVA_METRICS
                },
            })
    with open(
        os.path.join(output_dir, "explained_variance.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer_name", "component", "allocated", *EVA_METRICS],
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "eva",
        "state_version": state["version"],
        "total_rank_budget": state["total_rank_budget"],
        "module_count": len(state["rank_pattern"]),
        "rank_min": min(state["rank_pattern"].values()),
        "rank_max": max(state["rank_pattern"].values()),
        "rank_mean": state["total_rank_budget"] / len(state["rank_pattern"]),
        "metric": state["metric"],
        "max_components": state["max_components"],
        "processed_batches": state["processed_batches"],
        "converged": state["converged"],
        **(metadata or {}),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
