"""Optional Shapley-value allocator for PEFT AdaLoRA.

This module is deliberately isolated from the project's NBS allocator.  It is
only constructed when ``adalora_allocator='shapley'`` and otherwise has no
effect on LoRA, stock PEFT AdaLoRA, EVA, or NBS execution.

The implementation follows the two-stage online approximation used by
Suhyeon602/wiset: validation-loss Shapley values are estimated for whole LoRA
modules, the current PEFT budget is distributed between modules, and |lora_E|
selects components within each module.  It is reimplemented here to preserve
PEFT 0.6.2 compatibility and to guarantee state restoration on failed
coalition evaluations.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

try:  # PEFT 0.6.2 wheels expose both layouts depending on packaging.
    from peft.tuners.adalora.layer import RankAllocator
except ImportError:  # pragma: no cover - exercised by the monolithic layout.
    from peft.tuners.adalora import RankAllocator


LossFn = Callable[[], float]


class ShapleyRankAllocator(RankAllocator):
    """Drop-in PEFT RankAllocator using module-level Monte Carlo Shapley.

    ``mask_to_budget`` keeps PEFT's cubic budget schedule unchanged.  Only the
    score used to divide that budget is replaced.  Coalition value is
    ``-loss_fn()`` on fixed validation batches.
    """

    def __init__(
        self,
        model,
        peft_config,
        adapter_name: str,
        *,
        loss_fn: Optional[LossFn] = None,
        n_permutations: int = 3,
        truncate_fraction: float = 0.05,
        seed: int = 0,
        antithetic: bool = True,
        verbose: bool = True,
    ):
        super().__init__(model, peft_config, adapter_name)
        if n_permutations <= 0:
            raise ValueError("Shapley n_permutations must be positive")
        if not 0.0 <= truncate_fraction <= 1.0:
            raise ValueError("Shapley truncate_fraction must be in [0, 1]")
        self.loss_fn = loss_fn
        self.n_permutations = int(n_permutations)
        self.truncate_fraction = float(truncate_fraction)
        self.antithetic = bool(antithetic)
        self.verbose = bool(verbose)
        self._rng = random.Random(int(seed))
        self.last_module_shapley: Dict[str, float] = {}
        self.last_rank_pattern: Dict[str, List[bool]] = {}
        self.last_budget: Optional[int] = None

    def set_loss_fn(self, loss_fn: LossFn) -> None:
        if not callable(loss_fn):
            raise TypeError("Shapley loss_fn must be callable")
        self.loss_fn = loss_fn

    def update_ipt(self, model) -> None:
        """Shapley does not use PEFT's gradient-sensitivity statistics."""

    def _lora_e_parameters(self, model) -> Dict[str, torch.nn.Parameter]:
        marker = f"lora_E.{self.adapter_name}"
        return {
            name: parameter
            for name, parameter in model.named_parameters()
            if marker in name
        }

    @staticmethod
    def _active_components(
        parameters: Mapping[str, torch.nn.Parameter], tolerance: float = 1e-12
    ):
        values: Dict[str, torch.Tensor] = {}
        active: Dict[str, List[int]] = {}
        for name, parameter in parameters.items():
            flat = parameter.detach().reshape(-1)
            values[name] = flat.clone()
            indices = torch.nonzero(flat.abs() > tolerance, as_tuple=False).reshape(-1)
            if indices.numel():
                active[name] = [int(index) for index in indices.cpu().tolist()]
        return values, active

    @staticmethod
    @torch.no_grad()
    def _restore(
        parameters: Mapping[str, torch.nn.Parameter],
        values: Mapping[str, torch.Tensor],
    ) -> None:
        for name, parameter in parameters.items():
            parameter.reshape(-1).copy_(
                values[name].to(device=parameter.device, dtype=parameter.dtype)
            )

    @staticmethod
    @torch.no_grad()
    def _set_module_coalition(
        parameters: Mapping[str, torch.nn.Parameter],
        values: Mapping[str, torch.Tensor],
        active_components: Mapping[str, Sequence[int]],
        enabled_modules: Iterable[str],
    ) -> None:
        enabled = set(enabled_modules)
        for name, indices in active_components.items():
            flat = parameters[name].reshape(-1)
            index_tensor = torch.tensor(indices, device=flat.device, dtype=torch.long)
            if name in enabled:
                source = values[name].to(device=flat.device, dtype=flat.dtype)
                flat[index_tensor] = source[index_tensor]
            else:
                flat[index_tensor] = 0

    def _permutations(self, players: Sequence[str]) -> List[List[str]]:
        permutations: List[List[str]] = []
        while len(permutations) < self.n_permutations:
            order = list(players)
            self._rng.shuffle(order)
            permutations.append(order)
            if self.antithetic and len(permutations) < self.n_permutations:
                permutations.append(list(reversed(order)))
        return permutations

    def _module_shapley(
        self,
        parameters: Mapping[str, torch.nn.Parameter],
        values: Mapping[str, torch.Tensor],
        active_components: Mapping[str, Sequence[int]],
    ) -> Dict[str, float]:
        if self.loss_fn is None:
            raise RuntimeError(
                "Shapley validation loss callback is not configured; "
                "call set_loss_fn() before training"
            )
        players = sorted(active_components)
        if not players:
            raise RuntimeError("Shapley allocator found no active lora_E components")

        def coalition_value(enabled: Iterable[str]) -> float:
            self._set_module_coalition(
                parameters, values, active_components, enabled
            )
            return -float(self.loss_fn())

        try:
            full_value = coalition_value(players)
            empty_value = coalition_value(())
            truncate_tolerance = (
                abs(full_value - empty_value) * self.truncate_fraction
            )
            shapley = {name: 0.0 for name in players}
            permutations = self._permutations(players)
            for permutation_index, permutation in enumerate(permutations, start=1):
                enabled = set()
                previous = empty_value
                for player in permutation:
                    if abs(full_value - previous) <= truncate_tolerance:
                        marginal = 0.0
                    else:
                        enabled.add(player)
                        current = coalition_value(enabled)
                        marginal = current - previous
                        previous = current
                    shapley[player] += marginal
                if self.verbose:
                    print(
                        f"[Shapley AdaLoRA] permutation "
                        f"{permutation_index}/{len(permutations)} complete",
                        flush=True,
                    )
            scale = float(len(permutations))
            return {name: value / scale for name, value in shapley.items()}
        finally:
            # A failed validation forward must never leave a temporary
            # coalition mask in the trainable model.
            self._restore(parameters, values)

    @staticmethod
    def _allocate_integer_budget(
        shapley: Mapping[str, float], capacities: Mapping[str, int], budget: int
    ) -> Dict[str, int]:
        total_capacity = sum(capacities.values())
        if budget < 0 or budget > total_capacity:
            raise ValueError(
                f"Shapley budget {budget} is outside [0, {total_capacity}]"
            )
        allocation = {name: 0 for name in capacities}
        positive = {name: max(0.0, float(shapley[name])) for name in capacities}
        positive_sum = sum(positive.values())

        if positive_sum > 0 and budget > 0:
            raw = {
                name: budget * positive[name] / positive_sum
                for name in capacities
            }
            for name in capacities:
                allocation[name] = min(capacities[name], int(math.floor(raw[name])))
            priority = sorted(
                capacities,
                key=lambda name: (
                    raw[name] - math.floor(raw[name]),
                    shapley[name],
                    name,
                ),
                reverse=True,
            )
        else:
            priority = sorted(
                capacities,
                key=lambda name: (shapley[name], name),
                reverse=True,
            )

        remaining = budget - sum(allocation.values())
        while remaining:
            eligible = [
                name for name in priority
                if allocation[name] < capacities[name]
            ]
            if not eligible:
                raise RuntimeError(
                    "Shapley allocator could not satisfy the PEFT rank budget"
                )
            for name in eligible:
                allocation[name] += 1
                remaining -= 1
                if remaining == 0:
                    break
        return allocation

    def mask_to_budget(self, model, budget):
        parameters = self._lora_e_parameters(model)
        values, active_components = self._active_components(parameters)
        if not active_components:
            raise RuntimeError("Shapley allocator found no active AdaLoRA modules")
        active_capacity = sum(len(indices) for indices in active_components.values())
        if budget > active_capacity:
            raise ValueError(
                f"PEFT requested budget {budget}, but only {active_capacity} "
                "active Shapley components remain"
            )

        shapley = self._module_shapley(parameters, values, active_components)
        capacities = {
            name: len(indices) for name, indices in active_components.items()
        }
        allocation = self._allocate_integer_budget(shapley, capacities, int(budget))

        rank_pattern: Dict[str, List[bool]] = {}
        with torch.no_grad():
            for name, parameter in parameters.items():
                flat = parameter.reshape(-1)
                keep_count = allocation.get(name, 0)
                indices = active_components.get(name, [])
                ranked = sorted(
                    indices,
                    key=lambda index: abs(float(values[name][index].item())),
                    reverse=True,
                )
                keep = set(ranked[:keep_count])
                mask = [index in keep for index in range(flat.numel())]
                mask_tensor = torch.tensor(mask, device=flat.device, dtype=torch.bool)
                flat.masked_fill_(~mask_tensor, 0)
                rank_pattern[name] = mask

        actual_budget = sum(sum(mask) for mask in rank_pattern.values())
        if actual_budget != int(budget):
            self._restore(parameters, values)
            raise RuntimeError(
                f"Shapley rank mask sums to {actual_budget}, expected {budget}"
            )
        self.last_module_shapley = dict(shapley)
        self.last_rank_pattern = rank_pattern
        self.last_budget = int(budget)
        if self.verbose:
            nonzero_modules = sum(value > 0 for value in allocation.values())
            print(
                f"[Shapley AdaLoRA] allocated budget={budget} across "
                f"{nonzero_modules}/{len(allocation)} active modules",
                flush=True,
            )
        return rank_pattern

