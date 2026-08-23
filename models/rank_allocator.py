"""Gradient/spectral rank allocation for AdaLoRA adapters.

The allocator deliberately lives outside PEFT.  PEFT 0.6.x keeps AdaLoRA's
rank slots at ``init_r`` and implements pruning by zeroing ``lora_E`` entries;
we use the same representation so checkpoints and forward code remain
compatible.
"""

from collections import OrderedDict
from fnmatch import fnmatchcase
import heapq
import re

import torch


class RankAllocationConstraintError(ValueError):
    """A proposed allocation violates layer bounds or the global budget."""

    def __init__(self, violations, requested_ranks):
        self.violations = list(violations)
        self.requested_ranks = dict(requested_ranks)
        detail = "; ".join(violation["message"] for violation in self.violations)
        super().__init__(detail)


class NashRankAllocator:
    """Allocate a global LoRA rank budget with a weighted Nash gain.

    ``max_rank`` is the maximum number of active slots allowed in each adapter;
    it may be smaller than the physical AdaLoRA rank.  Candidate components are
    selected by energy from all physical slots, never by their storage index.
    In fixed mode, ``rank_budget`` is the desired sum of active slots across
    all layers.  In adaptive mode it is the default upper bound; allocation
    stops earlier when the best remaining marginal Nash gain falls below a
    relative threshold.  Rank allocation is recomputed from the current
    gradients and shadow ``lora_E`` spectrum whenever :meth:`step` is called.
    """

    def __init__(self, model, target_rank, min_rank=None, max_rank=None,
                 rank_budget=None, ema_beta=0.9, eps=1e-8,
                 adapter_name="default", rank_config=None,
                 missing_grad_policy="zero", warmup_steps=0,
                 cooldown_start_step=None, allocation_interval=1,
                 shadow_update_policy="legacy", budget_mode="fixed",
                 relative_lambda=0.15, adaptive_min_budget=None,
                 adaptive_max_budget=None):
        self.model = model
        self.adapter_name = adapter_name
        self.target_rank = int(target_rank)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if missing_grad_policy not in ("zero", "hold"):
            raise ValueError("missing_grad_policy must be 'zero' or 'hold'")
        self.missing_grad_policy = missing_grad_policy
        if shadow_update_policy not in ("legacy", "active-only"):
            raise ValueError(
                "shadow_update_policy must be 'legacy' or 'active-only'"
            )
        self.shadow_update_policy = shadow_update_policy
        if budget_mode not in ("fixed", "adaptive"):
            raise ValueError("budget_mode must be 'fixed' or 'adaptive'")
        self.budget_mode = budget_mode
        self.relative_lambda = float(relative_lambda)
        if not 0.0 <= self.relative_lambda <= 1.0:
            raise ValueError("relative_lambda must be in [0, 1]")
        if self.budget_mode == "adaptive" and self.shadow_update_policy != "active-only":
            raise ValueError(
                "adaptive budget mode requires shadow_update_policy='active-only' "
                "so inactive spectral candidates remain available"
            )
        self.adaptive_min_budget_override = adaptive_min_budget
        self.adaptive_max_budget_override = adaptive_max_budget
        self.warmup_steps = int(warmup_steps)
        self.cooldown_start_step = (
            None if cooldown_start_step is None else int(cooldown_start_step)
        )
        self.allocation_interval = int(allocation_interval)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.cooldown_start_step is not None and self.cooldown_start_step < 1:
            raise ValueError("cooldown_start_step must be positive")
        if self.allocation_interval <= 0:
            raise ValueError("allocation_interval must be positive")
        self.rank_config = rank_config or {}
        self.min_rank_spec = min_rank
        self.max_rank_spec = max_rank
        self.rank_budget_override = rank_budget
        self.sensitivity = OrderedDict()
        self.ranks = OrderedDict()
        self.masks = OrderedDict()
        self.spectral_shadow = OrderedDict()
        self.initial_spectral_shadow = OrderedDict()
        self.initial_spectrum_is_exact = True
        self.last_gains = OrderedDict()
        self.last_diagnostics = []
        self.last_step = None
        self.ema_step = 0
        self.reference_gain = None
        self.stopping_threshold = None
        self.last_allocated_gain = None
        self.next_rejected_gain = None
        self.stopping_reason = "initialization"

        self.layers = self._find_layers()
        if not self.layers:
            raise ValueError("NashRankAllocator found no AdaLoRA layers")
        physical_ranks = self._physical_ranks()
        self.min_ranks = OrderedDict()
        self.max_ranks = OrderedDict()
        for name in self.layers:
            config = self._layer_rank_config(name)
            self.min_ranks[name] = self._resolve_bound(
                name, self.min_rank_spec, config.get("min_rank"),
                default=max(1, target_rank // 2), label="min_rank")
            self.max_ranks[name] = self._resolve_bound(
                name, self.max_rank_spec, config.get("max_rank"),
                default=physical_ranks[name], label="max_rank")
            if self.min_ranks[name] > self.max_ranks[name]:
                raise ValueError(f"minimum rank exceeds maximum rank for {name}")
            if self.max_ranks[name] > physical_ranks[name]:
                raise ValueError(f"max rank exceeds physical rank for {name}")
        self.rank_budget = self._resolve_budget()
        minimum_budget = sum(self.min_ranks.values())
        maximum_budget = sum(self.max_ranks.values())
        self.adaptive_min_budget = (
            minimum_budget
            if self.adaptive_min_budget_override is None
            else int(self.adaptive_min_budget_override)
        )
        self.adaptive_max_budget = (
            self.rank_budget
            if self.adaptive_max_budget_override is None
            else int(self.adaptive_max_budget_override)
        )
        if self.budget_mode == "adaptive":
            self.rank_budget = self.adaptive_max_budget
        else:
            # Keep one source of truth in fixed mode.  Adaptive-only bounds do
            # not alter the historical exact-budget behavior.
            self.adaptive_min_budget = self.rank_budget
            self.adaptive_max_budget = self.rank_budget
        if self.rank_budget < minimum_budget:
            raise ValueError("rank_budget is smaller than the minimum-rank budget")
        if self.rank_budget > maximum_budget:
            raise ValueError("rank_budget exceeds the available AdaLoRA slots")
        if self.budget_mode == "adaptive":
            if self.adaptive_min_budget < minimum_budget:
                raise ValueError(
                    "adaptive_min_budget is smaller than the minimum-rank budget"
                )
            if self.adaptive_min_budget > self.adaptive_max_budget:
                raise ValueError(
                    "adaptive_min_budget exceeds adaptive_max_budget"
                )
            if self.adaptive_max_budget > maximum_budget:
                raise ValueError(
                    "adaptive_max_budget exceeds the available AdaLoRA slots"
                )

        initial_ranks = self._initial_budget_ranks()
        for name, module in self.layers.items():
            self.sensitivity[name] = 0.0
            # Warm-up starts with a feasible, approximately uniform allocation
            # that already consumes the full budget.  NBS reallocation itself
            # still starts from every layer's minimum rank in _choose_ranks().
            self.ranks[name] = initial_ranks[name]
            self.spectral_shadow[name] = (
                module.lora_E[self.adapter_name].detach().float().reshape(-1).clone()
            )

        # Preserve the full candidate spectrum before the first rank mask is
        # ever applied.  The live spectral_shadow is subsequently refreshed
        # during training, whereas this copy remains an immutable experimental
        # reference for separating initialization from allocation-conditioned
        # spectral behavior.
        self.initial_spectral_shadow = OrderedDict(
            (name, values.detach().clone())
            for name, values in self.spectral_shadow.items()
        )
        self.stopping_reason = "warmup_initial_budget"
        self._apply_allocation(self.ranks)
        self.snapshot_diagnostics(step=0, event="initialization")

    def _initial_budget_ranks(self):
        """Fill the global budget as uniformly as layer bounds permit."""
        ranks = OrderedDict(
            (name, self.min_ranks[name]) for name in self.layers
        )
        remaining = self.rank_budget - sum(ranks.values())
        tie_breaker = {name: index for index, name in enumerate(self.layers)}
        heap = [
            (ranks[name], tie_breaker[name], name)
            for name in self.layers if ranks[name] < self.max_ranks[name]
        ]
        heapq.heapify(heap)
        for _ in range(remaining):
            if not heap:
                raise RuntimeError("could not fill rank budget within layer bounds")
            _, _, selected = heapq.heappop(heap)
            ranks[selected] += 1
            if ranks[selected] < self.max_ranks[selected]:
                heapq.heappush(
                    heap, (ranks[selected], tie_breaker[selected], selected)
                )
        return ranks

    def schedule_phase(self, step):
        """Return the training-time phase for a one-based optimizer step."""
        step = int(step)
        if step <= self.warmup_steps:
            return "warmup"
        if self.cooldown_start_step is not None and step >= self.cooldown_start_step:
            return "cooldown"
        return "allocation"

    def should_allocate(self, step):
        """Whether this optimizer step is an eligible NBS reallocation point."""
        step = int(step)
        return (
            self.schedule_phase(step) == "allocation"
            and step % self.allocation_interval == 0
        )

    def _layer_rank_config(self, name):
        """Resolve an exact or glob-style per-layer rank override."""
        exact = self.rank_config.get(name)
        if exact is not None:
            return exact
        matches = [
            (pattern, value) for pattern, value in self.rank_config.items()
            if isinstance(pattern, str) and fnmatchcase(name, pattern)
        ]
        if not matches:
            return {}
        matches.sort(
            key=lambda item: sum(ch not in "*?" for ch in item[0]),
            reverse=True,
        )
        return matches[0][1]

    def _find_layers(self):
        layers = OrderedDict()
        for name, module in self.model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            lora_e = getattr(module, "lora_E", None)
            if lora_a is None or lora_b is None or lora_e is None:
                continue
            if self.adapter_name not in lora_a or self.adapter_name not in lora_b:
                continue
            if self.adapter_name not in lora_e:
                continue
            layers[name] = module
        return layers

    def _physical_ranks(self):
        return OrderedDict(
            (name, int(module.lora_E[self.adapter_name].shape[0]))
            for name, module in self.layers.items()
        )

    def _resolve_bound(self, name, scalar, override, default, label):
        value = override
        if value is None and isinstance(scalar, dict):
            value = scalar.get(name)
        if value is None:
            value = scalar if scalar is not None and not isinstance(scalar, dict) else default
        value = int(value)
        if value < 1:
            raise ValueError(f"{label} must be positive for {name}")
        return value

    def _resolve_budget(self):
        if self.rank_budget_override is not None:
            return int(self.rank_budget_override)
        return self.target_rank * len(self.layers)

    def _gradient_norm(self, module):
        values = []
        for container in (module.lora_A, module.lora_B):
            parameter = container[self.adapter_name]
            # PEFT 0.6.x AdaLoRA stores A/B as ParameterDict tensors, while
            # newer LoRA implementations may expose Linear modules instead.
            if hasattr(parameter, "weight"):
                parameter = parameter.weight
            if parameter.grad is not None:
                values.append(parameter.grad.detach().float().norm().pow(2))
        if not values:
            return None
        # Keep the scalar on its original device.  update_sensitivity() batches
        # the CPU transfer once per device instead of synchronizing once per
        # LoRA layer.
        return torch.stack(values).sum().sqrt()

    def update_sensitivity(self):
        """Update layer sensitivity from the gradients left by backward()."""
        self.ema_step += 1
        pending = OrderedDict()
        by_device = {}
        for name, module in self.layers.items():
            gradient_norm = self._gradient_norm(module)
            if gradient_norm is None:
                if self.missing_grad_policy == "hold":
                    continue
                gradient_norm = 0.0
            if torch.is_tensor(gradient_norm):
                pending[name] = gradient_norm
                by_device.setdefault(gradient_norm.device, []).append(name)
            else:
                previous = self.sensitivity[name]
                self.sensitivity[name] = self.ema_beta * previous

        # One synchronization per device, rather than one .item() per layer.
        for device, names in by_device.items():
            values = torch.stack([pending[name] for name in names]).detach().cpu().tolist()
            for name, gradient_norm in zip(names, values):
                previous = self.sensitivity[name]
                self.sensitivity[name] = (
                    self.ema_beta * previous + (1.0 - self.ema_beta) * gradient_norm
                )

    def _bias_corrected_sensitivity(self):
        if self.ema_step <= 0:
            return {name: 0.0 for name in self.layers}
        correction = 1.0 - self.ema_beta ** self.ema_step
        return {
            name: value / max(correction, self.eps)
            for name, value in self.sensitivity.items()
        }

    def _weights(self):
        corrected = self._bias_corrected_sensitivity()
        values = torch.tensor(list(corrected.values()), dtype=torch.float64)
        values = values + self.eps
        values = values / values.sum().clamp_min(self.eps)
        return dict(zip(self.layers.keys(), values.tolist()))

    def _spectral_energy(self, name):
        """Return the strongest ``max_rank`` energies from every physical slot."""
        energy = self.spectral_shadow[name].abs().square()
        max_rank = self.max_ranks[name]
        if energy.numel() < max_rank:
            energy = torch.nn.functional.pad(energy, (0, max_rank - energy.numel()))
        # Sort before applying the configured ceiling.  Physical AdaLoRA uses
        # init_r slots (often rank * 2), and useful components can occupy any
        # slot; slicing first would permanently exclude high-energy components
        # whose storage index happens to be >= max_rank.
        return torch.sort(energy, descending=True).values[:max_rank]

    def _utility(self, name):
        """Build U_l(r) for r=0..r_l_max from the shadow spectrum."""
        energy = self._spectral_energy(name)
        total = energy.sum().clamp_min(self.eps)
        cumulative = torch.cat((energy.new_zeros(1), torch.cumsum(energy, dim=0)))
        return self.eps + cumulative / total

    def _marginal_gain(self, utility, rank, weight):
        marginal_utility_gain = self._marginal_utility_gain(utility, rank)
        if marginal_utility_gain == float("-inf"):
            return float("-inf")
        return float(weight * marginal_utility_gain)

    @staticmethod
    def _utility_increment(utility, rank):
        if rank + 1 >= utility.numel():
            return float("-inf")
        return float((utility[rank + 1] - utility[rank]).item())

    @staticmethod
    def _marginal_utility_gain(utility, rank):
        """Unweighted log utility gain for adding the next rank unit."""
        if rank + 1 >= utility.numel():
            return float("-inf")
        return float(torch.log(utility[rank + 1] / utility[rank]).item())

    def _rank_choice_inputs(self):
        weights = self._weights()
        utilities = {name: self._utility(name) for name in self.layers}
        ranks = {name: self.min_ranks[name] for name in self.layers}
        heap = []
        tie_breaker = {name: index for index, name in enumerate(self.layers)}
        for name in self.layers:
            if ranks[name] < self.max_ranks[name]:
                gain = self._marginal_gain(utilities[name], ranks[name], weights[name])
                heapq.heappush(heap, (-gain, tie_breaker[name], name))
        return ranks, utilities, weights, heap, tie_breaker

    def _push_next_gain(self, heap, tie_breaker, selected, ranks,
                        utilities, weights):
        if ranks[selected] >= self.max_ranks[selected]:
            return
        next_gain = self._marginal_gain(
            utilities[selected], ranks[selected], weights[selected]
        )
        heapq.heappush(
            heap, (-next_gain, tie_breaker[selected], selected)
        )

    def _choose_ranks_fixed(self):
        ranks, utilities, weights, heap, tie_breaker = self._rank_choice_inputs()
        remaining = self.rank_budget - sum(ranks.values())

        self.last_gains = OrderedDict()
        self.reference_gain = -heap[0][0] if heap else None
        self.stopping_threshold = None
        self.last_allocated_gain = None
        self.next_rejected_gain = None
        for _ in range(remaining):
            if not heap:
                break
            neg_gain, _, selected = heapq.heappop(heap)
            selected_gain = -neg_gain
            self.last_gains[selected] = selected_gain
            self.last_allocated_gain = selected_gain
            ranks[selected] += 1
            self._push_next_gain(
                heap, tie_breaker, selected, ranks, utilities, weights
            )
        self.next_rejected_gain = -heap[0][0] if heap else None
        self.stopping_reason = "fixed_budget_reached"
        return ranks, utilities, weights

    def _choose_ranks_adaptive(self):
        ranks, utilities, weights, heap, tie_breaker = self._rank_choice_inputs()
        self.last_gains = OrderedDict()
        self.reference_gain = -heap[0][0] if heap else 0.0
        self.stopping_threshold = self.relative_lambda * self.reference_gain
        self.last_allocated_gain = None
        self.next_rejected_gain = None
        self.stopping_reason = "no_available_candidate"

        while heap and sum(ranks.values()) < self.adaptive_max_budget:
            best_gain = -heap[0][0]
            current_total = sum(ranks.values())
            if (
                current_total >= self.adaptive_min_budget
                and best_gain <= self.stopping_threshold
            ):
                self.next_rejected_gain = best_gain
                self.stopping_reason = "relative_threshold_reached"
                break
            neg_gain, _, selected = heapq.heappop(heap)
            selected_gain = -neg_gain
            self.last_gains[selected] = selected_gain
            self.last_allocated_gain = selected_gain
            ranks[selected] += 1
            self._push_next_gain(
                heap, tie_breaker, selected, ranks, utilities, weights
            )
        else:
            if sum(ranks.values()) >= self.adaptive_max_budget:
                self.stopping_reason = "adaptive_max_budget_reached"
                self.next_rejected_gain = -heap[0][0] if heap else None

        return ranks, utilities, weights

    def _choose_ranks(self):
        # Keep the original exact-budget path separate so fixed mode remains
        # a stable baseline for every existing experiment and checkpoint.
        if self.budget_mode == "adaptive":
            return self._choose_ranks_adaptive()
        return self._choose_ranks_fixed()

    @staticmethod
    def _module_coordinates(name):
        """Extract human-readable Transformer layer and projection labels."""
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        layer_index = int(match.group(1)) if match else None
        module_type = name.rsplit(".", 1)[-1]
        return layer_index, module_type

    def _build_diagnostics(self, step, event, ranks, previous_ranks,
                           utilities, weights):
        corrected = self._bias_corrected_sensitivity()
        phase = self.schedule_phase(step)
        total_rank = sum(int(value) for value in ranks.values())
        rows = []
        for name in self.layers:
            rank = int(ranks[name])
            min_rank = int(self.min_ranks[name])
            max_rank = int(self.max_ranks[name])
            layer_index, module_type = self._module_coordinates(name)
            utility = utilities[name]
            next_gain = None
            next_utility_increment = None
            next_marginal_utility_gain = None
            if rank < max_rank:
                next_utility_increment = self._utility_increment(utility, rank)
                next_marginal_utility_gain = self._marginal_utility_gain(
                    utility, rank
                )
                next_gain = self._marginal_gain(utility, rank, weights[name])
            rows.append({
                "optimizer_step": int(step),
                "phase": phase,
                "event": event,
                "layer_name": name,
                "transformer_layer_index": layer_index,
                "module_type": module_type,
                "rank": rank,
                "sensitivity": float(corrected[name]),
                "alpha": float(weights[name]),
                "spectral_energy_total": float(
                    self._spectral_energy(name).sum().item()
                ),
                "utility": float(utility[rank].item()),
                "next_utility_increment": next_utility_increment,
                "next_marginal_utility_gain": next_marginal_utility_gain,
                "next_marginal_gain": next_gain,
                "min_rank": min_rank,
                "max_rank": max_rank,
                "at_min_rank": int(rank == min_rank),
                "at_max_rank": int(rank == max_rank),
                "rank_delta": rank - int(previous_ranks.get(name, rank)),
                "total_rank": total_rank,
                "rank_budget": int(self.rank_budget),
                "budget_mode": self.budget_mode,
                "relative_lambda": (
                    self.relative_lambda if self.budget_mode == "adaptive" else None
                ),
                "reference_gain": self.reference_gain,
                "stopping_threshold": self.stopping_threshold,
                "last_allocated_gain": self.last_allocated_gain,
                "next_rejected_gain": self.next_rejected_gain,
                "effective_rank_budget": total_rank,
                "rank_budget_cap": int(self.adaptive_max_budget),
                "adaptive_min_budget": int(self.adaptive_min_budget),
                "stopping_reason": self.stopping_reason,
            })
        return rows

    def snapshot_diagnostics(self, step, event="snapshot"):
        """Capture per-layer statistics without changing the active ranks."""
        weights = self._weights()
        utilities = {name: self._utility(name) for name in self.layers}
        self.last_diagnostics = self._build_diagnostics(
            step=step,
            event=event,
            ranks=self.ranks,
            previous_ranks=self.ranks,
            utilities=utilities,
            weights=weights,
        )
        return list(self.last_diagnostics)

    def _refresh_spectral_shadow(self):
        """Refresh candidate spectra according to the configured mask policy.

        ``legacy`` preserves the historical behavior: any nonzero physical
        ``lora_E`` slot is considered observed.  ``active-only`` updates only
        slots selected by the current rank mask, preventing optimizer momentum
        or numerical leakage in inactive slots from replacing their preserved
        candidate spectrum.
        """
        for name, module in self.layers.items():
            current = module.lora_E[self.adapter_name].detach().float().reshape(-1)
            shadow = self.spectral_shadow[name].to(device=current.device)
            if shadow.numel() != current.numel():
                shadow = torch.zeros_like(current)
            observed = current.abs() > self.eps
            if self.shadow_update_policy == "active-only":
                mask = self.masks.get(name)
                if mask is not None:
                    observed = observed & mask.to(device=current.device).bool()
            shadow = torch.where(observed, current, shadow)
            self.spectral_shadow[name] = shadow.detach().clone()

    def _apply_allocation(self, ranks, energies=None):
        # Validate the complete candidate before mutating any lora_E tensor or
        # mask.  This makes a rejected allocation transactional: callers can
        # catch RankAllocationConstraintError and safely retain the previous
        # valid topology without a partially applied result.
        self._validate_rank_allocation(ranks)
        for name, module in self.layers.items():
            e = module.lora_E[self.adapter_name]
            raw = self.spectral_shadow[name].to(device=e.device, dtype=e.dtype)
            max_rank = self.max_ranks[name]
            # Rank is a count ceiling, not a prefix of physical slot indices.
            # Select the strongest active components across the full init_r
            # vector while activating no more than max_rank of them.
            energy = raw.abs().square()
            k = min(int(ranks[name]), max_rank, energy.numel())
            indices = torch.topk(energy, k=k, largest=True, sorted=False).indices
            mask = torch.zeros_like(raw)
            mask[indices] = 1.0
            with torch.no_grad():
                e.copy_((raw * mask).view_as(e))
            self.masks[name] = mask

    def _validate_rank_allocation(self, ranks):
        """Reject an invalid complete allocation before it changes the model."""
        requested = dict(ranks)
        violations = []
        expected_names = set(self.layers)
        missing = expected_names.difference(requested)
        unexpected = set(requested).difference(expected_names)
        for name in sorted(missing):
            violations.append({
                "layer_name": name,
                "requested_rank": None,
                "min_rank": self.min_ranks[name],
                "max_rank": self.max_ranks[name],
                "reason": "missing_layer",
                "message": f"allocation is missing layer {name}",
            })
        for name in sorted(unexpected):
            violations.append({
                "layer_name": name,
                "requested_rank": requested[name],
                "min_rank": None,
                "max_rank": None,
                "reason": "unexpected_layer",
                "message": f"allocation contains unexpected layer {name}",
            })

        validated_ranks = {}
        for name in self.layers:
            if name not in requested:
                continue
            rank = int(requested[name])
            validated_ranks[name] = rank
            minimum = int(self.min_ranks[name])
            maximum = int(self.max_ranks[name])
            if rank < minimum or rank > maximum:
                violations.append({
                    "layer_name": name,
                    "requested_rank": rank,
                    "min_rank": minimum,
                    "max_rank": maximum,
                    "reason": "rank_out_of_bounds",
                    "message": (
                        f"layer {name} requested rank {rank}, allowed range "
                        f"is [{minimum}, {maximum}]"
                    ),
                })

        if len(validated_ranks) == len(self.layers):
            requested_total = sum(validated_ranks.values())
            if self.budget_mode == "fixed" and requested_total != self.rank_budget:
                violations.append({
                    "layer_name": None,
                    "requested_rank": requested_total,
                    "min_rank": self.rank_budget,
                    "max_rank": self.rank_budget,
                    "reason": "global_budget_mismatch",
                    "message": (
                        f"allocation total rank {requested_total} does not match "
                        f"global budget {self.rank_budget}"
                    ),
                })
            elif self.budget_mode == "adaptive" and not (
                self.adaptive_min_budget
                <= requested_total
                <= self.adaptive_max_budget
            ):
                violations.append({
                    "layer_name": None,
                    "requested_rank": requested_total,
                    "min_rank": self.adaptive_min_budget,
                    "max_rank": self.adaptive_max_budget,
                    "reason": "adaptive_budget_out_of_bounds",
                    "message": (
                        f"adaptive allocation total rank {requested_total} is "
                        f"outside [{self.adaptive_min_budget}, "
                        f"{self.adaptive_max_budget}]"
                    ),
                })

        if violations:
            raise RankAllocationConstraintError(violations, requested)

    def enforce_masks(self):
        """Keep previously deactivated spectral slots zero after optimizer.step()."""
        self._refresh_spectral_shadow()
        with torch.no_grad():
            for name, module in self.layers.items():
                mask = self.masks.get(name)
                if mask is None:
                    continue
                e = module.lora_E[self.adapter_name]
                raw = self.spectral_shadow[name].to(device=e.device, dtype=e.dtype)
                e.copy_((raw * mask).view_as(e))

    def state_dict(self):
        """Return allocator state separately from the PEFT adapter state."""
        return {
            "version": 5,
            "target_rank": self.target_rank,
            "rank_budget": self.rank_budget,
            "budget_mode": self.budget_mode,
            "relative_lambda": self.relative_lambda,
            "adaptive_min_budget": self.adaptive_min_budget,
            "adaptive_max_budget": self.adaptive_max_budget,
            "effective_rank_budget": sum(self.ranks.values()),
            "reference_gain": self.reference_gain,
            "stopping_threshold": self.stopping_threshold,
            "last_allocated_gain": self.last_allocated_gain,
            "next_rejected_gain": self.next_rejected_gain,
            "stopping_reason": self.stopping_reason,
            "ema_beta": self.ema_beta,
            "eps": self.eps,
            "missing_grad_policy": self.missing_grad_policy,
            "shadow_update_policy": self.shadow_update_policy,
            "warmup_steps": self.warmup_steps,
            "cooldown_start_step": self.cooldown_start_step,
            "allocation_interval": self.allocation_interval,
            "min_ranks": dict(self.min_ranks),
            "max_ranks": dict(self.max_ranks),
            "sensitivity": dict(self.sensitivity),
            "ema_step": self.ema_step,
            "ranks": dict(self.ranks),
            "masks": {name: mask.detach().cpu() for name, mask in self.masks.items()},
            "spectral_shadow": {
                name: values.detach().cpu() for name, values in self.spectral_shadow.items()
            },
            "initial_spectral_shadow": {
                name: values.detach().cpu()
                for name, values in self.initial_spectral_shadow.items()
            },
            "initial_spectrum_is_exact": bool(self.initial_spectrum_is_exact),
            "last_step": self.last_step,
        }

    def pre_mask_spectrum_state_dict(self):
        """Return a snapshot whose spectrum predates the first rank mask."""
        if not self.initial_spectrum_is_exact or not self.initial_spectral_shadow:
            raise RuntimeError(
                "exact pre-mask spectrum is unavailable in this allocator state"
            )
        state = self.state_dict()
        state["spectral_shadow"] = {
            name: values.detach().cpu()
            for name, values in self.initial_spectral_shadow.items()
        }
        state["ranks"] = dict(self.max_ranks)
        state["masks"] = {
            name: torch.ones_like(values, device="cpu")
            for name, values in self.initial_spectral_shadow.items()
        }
        state["last_step"] = None
        return state

    def load_state_dict(self, state):
        """Restore allocator state after the PEFT adapter has been loaded."""
        for name in self.layers:
            if name not in state.get("spectral_shadow", {}):
                raise ValueError(f"allocator checkpoint is missing layer {name}")
        self.sensitivity.update(state.get("sensitivity", {}))
        # Version <=3 checkpoints predate the selectable policy and therefore
        # restore the exact historical behavior.
        shadow_update_policy = state.get("shadow_update_policy", "legacy")
        if shadow_update_policy not in ("legacy", "active-only"):
            raise ValueError(
                "allocator checkpoint has invalid shadow_update_policy: "
                f"{shadow_update_policy!r}"
            )
        budget_mode = state.get("budget_mode", "fixed")
        if budget_mode not in ("fixed", "adaptive"):
            raise ValueError(
                f"allocator checkpoint has invalid budget_mode: {budget_mode!r}"
            )
        if budget_mode == "adaptive" and shadow_update_policy != "active-only":
            raise ValueError(
                "adaptive allocator checkpoint requires active-only spectral shadow"
            )
        self.shadow_update_policy = shadow_update_policy
        self.budget_mode = budget_mode
        self.relative_lambda = float(state.get("relative_lambda", 0.15))
        if not 0.0 <= self.relative_lambda <= 1.0:
            raise ValueError("allocator checkpoint relative_lambda must be in [0, 1]")
        self.rank_budget = int(state.get("rank_budget", self.rank_budget))
        self.adaptive_min_budget = int(
            state.get("adaptive_min_budget", self.rank_budget)
        )
        self.adaptive_max_budget = int(
            state.get("adaptive_max_budget", self.rank_budget)
        )
        self.reference_gain = state.get("reference_gain")
        self.stopping_threshold = state.get("stopping_threshold")
        self.last_allocated_gain = state.get("last_allocated_gain")
        self.next_rejected_gain = state.get("next_rejected_gain")
        self.stopping_reason = state.get("stopping_reason", "checkpoint_restore")
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        cooldown_start = state.get("cooldown_start_step", self.cooldown_start_step)
        self.cooldown_start_step = (
            None if cooldown_start is None else int(cooldown_start)
        )
        self.allocation_interval = int(
            state.get("allocation_interval", self.allocation_interval)
        )
        self.ema_step = int(state.get("ema_step", 0))
        self.ranks.update({name: int(value) for name, value in state.get("ranks", {}).items()})
        initial_shadow = state.get("initial_spectral_shadow")
        initial_exact = bool(state.get("initial_spectrum_is_exact", False))
        self.initial_spectral_shadow = OrderedDict()
        if initial_shadow is not None and initial_exact:
            for name in self.layers:
                if name not in initial_shadow:
                    raise ValueError(
                        f"allocator checkpoint is missing initial spectrum for {name}"
                    )
                self.initial_spectral_shadow[name] = (
                    initial_shadow[name].detach().float().reshape(-1).clone()
                )
            self.initial_spectrum_is_exact = True
        else:
            # Version <=2 checkpoints did not preserve the pre-mask spectrum.
            # Do not silently label their current shadow as an initialization
            # measurement.
            self.initial_spectrum_is_exact = False
        for name in self.layers:
            module = self.layers[name]
            e = module.lora_E[self.adapter_name]
            shadow = state["spectral_shadow"][name].to(device=e.device, dtype=torch.float32)
            if shadow.numel() != e.numel():
                raise ValueError(f"allocator spectral shape mismatch for {name}")
            self.spectral_shadow[name] = shadow.clone()
            mask = state.get("masks", {}).get(name)
            if mask is None:
                mask = torch.ones_like(shadow)
            self.masks[name] = mask.to(device=e.device, dtype=torch.float32).reshape(-1)
        self.last_step = state.get("last_step")
        # Do not refresh the shadow here: the freshly constructed model may
        # still contain its initial E values.  The checkpoint shadow is the
        # source of truth during restoration.
        with torch.no_grad():
            for name, module in self.layers.items():
                e = module.lora_E[self.adapter_name]
                raw = self.spectral_shadow[name].to(device=e.device, dtype=e.dtype)
                e.copy_((raw * self.masks[name]).view_as(e))

    def step(self, step=None):
        """Consume gradients and follow the configured training-time schedule."""
        self.update_sensitivity()
        if step is None or self.should_allocate(step):
            return self.allocate(step)
        self.enforce_masks()
        return dict(self.ranks)

    def allocate(self, step=None):
        """Allocate ranks using the most recently updated sensitivities."""
        self._refresh_spectral_shadow()
        previous_ranks = dict(self.ranks)
        ranks, utilities, weights = self._choose_ranks()
        # Do not publish candidate ranks until the full allocation has passed
        # validation and its masks have been applied successfully.
        self._apply_allocation(ranks, utilities)
        self.ranks = OrderedDict(ranks)
        self.last_step = step
        diagnostic_step = self.ema_step if step is None else step
        self.last_diagnostics = self._build_diagnostics(
            step=diagnostic_step,
            event="allocation",
            ranks=self.ranks,
            previous_ranks=previous_ranks,
            utilities=utilities,
            weights=weights,
        )
        return dict(self.ranks)

    def active_rank_summary(self):
        return dict(self.ranks)
