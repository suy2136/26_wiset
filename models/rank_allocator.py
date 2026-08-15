"""Gradient/spectral rank allocation for AdaLoRA adapters.

The allocator deliberately lives outside PEFT.  PEFT 0.6.x keeps AdaLoRA's
rank slots at ``init_r`` and implements pruning by zeroing ``lora_E`` entries;
we use the same representation so checkpoints and forward code remain
compatible.
"""

from collections import OrderedDict
from fnmatch import fnmatchcase
import heapq

import torch


class NashRankAllocator:
    """Allocate a global LoRA rank budget with a weighted Nash gain.

    ``max_rank`` is the number of slots physically present in each adapter.
    ``rank_budget`` is the desired sum of active slots across all layers.
    Rank allocation is recomputed from the current gradients and ``lora_E``
    values whenever :meth:`step` is called.
    """

    def __init__(self, model, target_rank, min_rank=None, max_rank=None,
                 rank_budget=None, ema_beta=0.9, eps=1e-8,
                 adapter_name="default", rank_config=None,
                 missing_grad_policy="zero"):
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
        self.rank_config = rank_config or {}
        self.min_rank_spec = min_rank
        self.max_rank_spec = max_rank
        self.rank_budget_override = rank_budget
        self.sensitivity = OrderedDict()
        self.ranks = OrderedDict()
        self.masks = OrderedDict()
        self.spectral_shadow = OrderedDict()
        self.last_gains = OrderedDict()
        self.last_step = None
        self.ema_step = 0

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
        if self.rank_budget < sum(self.min_ranks.values()):
            raise ValueError("rank_budget is smaller than the minimum-rank budget")
        if self.rank_budget > sum(self.max_ranks.values()):
            raise ValueError("rank_budget exceeds the available AdaLoRA slots")

        for name, module in self.layers.items():
            self.sensitivity[name] = 0.0
            # The Nash allocation starts from every layer's minimum rank.
            # The remaining global budget is distributed by allocate().
            self.ranks[name] = self.min_ranks[name]
            self.spectral_shadow[name] = (
                module.lora_E[self.adapter_name].detach().float().reshape(-1).clone()
            )

        self._apply_allocation(self.ranks)

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
        energy = self.spectral_shadow[name].abs().square()
        max_rank = self.max_ranks[name]
        if energy.numel() < max_rank:
            energy = torch.nn.functional.pad(energy, (0, max_rank - energy.numel()))
        elif energy.numel() > max_rank:
            energy = energy[:max_rank]
        return torch.sort(energy, descending=True).values

    def _utility(self, name):
        """Build U_l(r) for r=0..r_l_max from the shadow spectrum."""
        energy = self._spectral_energy(name)
        total = energy.sum().clamp_min(self.eps)
        cumulative = torch.cat((energy.new_zeros(1), torch.cumsum(energy, dim=0)))
        return self.eps + cumulative / total

    def _marginal_gain(self, utility, rank, weight):
        if rank + 1 >= utility.numel():
            return float("-inf")
        return float(weight * torch.log(utility[rank + 1] / utility[rank]).item())

    def _choose_ranks(self):
        weights = self._weights()
        utilities = {name: self._utility(name) for name in self.layers}
        ranks = {name: self.min_ranks[name] for name in self.layers}
        remaining = self.rank_budget - sum(ranks.values())

        # Each layer has a diminishing marginal-gain sequence.  The heap
        # stores only the next available gain for each layer; after selecting
        # a layer, only that layer's next gain is recomputed.
        heap = []
        tie_breaker = {name: index for index, name in enumerate(self.layers)}
        for name in self.layers:
            if ranks[name] < self.max_ranks[name]:
                gain = self._marginal_gain(utilities[name], ranks[name], weights[name])
                heapq.heappush(heap, (-gain, tie_breaker[name], name))

        self.last_gains = OrderedDict()
        for _ in range(remaining):
            if not heap:
                break
            neg_gain, _, selected = heapq.heappop(heap)
            self.last_gains[selected] = -neg_gain
            ranks[selected] += 1
            if ranks[selected] < self.max_ranks[selected]:
                next_gain = self._marginal_gain(
                    utilities[selected], ranks[selected], weights[selected]
                )
                heapq.heappush(
                    heap, (-next_gain, tie_breaker[selected], selected)
                )
        return ranks, utilities

    def _refresh_spectral_shadow(self):
        """Capture current nonzero E values before enforcing the previous mask."""
        for name, module in self.layers.items():
            current = module.lora_E[self.adapter_name].detach().float().reshape(-1)
            shadow = self.spectral_shadow[name].to(device=current.device)
            if shadow.numel() != current.numel():
                shadow = torch.zeros_like(current)
            observed = current.abs() > self.eps
            shadow = torch.where(observed, current, shadow)
            self.spectral_shadow[name] = shadow.detach().clone()

    def _apply_allocation(self, ranks, energies=None):
        for name, module in self.layers.items():
            e = module.lora_E[self.adapter_name]
            raw = self.spectral_shadow[name].to(device=e.device, dtype=e.dtype)
            max_rank = self.max_ranks[name]
            energy = raw[:max_rank].abs().square()
            k = min(int(ranks[name]), energy.numel())
            indices = torch.topk(energy, k=k, largest=True, sorted=False).indices
            mask = torch.zeros_like(raw)
            mask[indices] = 1.0
            with torch.no_grad():
                e.copy_((raw * mask).view_as(e))
            self.masks[name] = mask

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
            "version": 1,
            "target_rank": self.target_rank,
            "rank_budget": self.rank_budget,
            "ema_beta": self.ema_beta,
            "eps": self.eps,
            "missing_grad_policy": self.missing_grad_policy,
            "min_ranks": dict(self.min_ranks),
            "max_ranks": dict(self.max_ranks),
            "sensitivity": dict(self.sensitivity),
            "ema_step": self.ema_step,
            "ranks": dict(self.ranks),
            "masks": {name: mask.detach().cpu() for name, mask in self.masks.items()},
            "spectral_shadow": {
                name: values.detach().cpu() for name, values in self.spectral_shadow.items()
            },
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        """Restore allocator state after the PEFT adapter has been loaded."""
        for name in self.layers:
            if name not in state.get("spectral_shadow", {}):
                raise ValueError(f"allocator checkpoint is missing layer {name}")
        self.sensitivity.update(state.get("sensitivity", {}))
        self.ema_step = int(state.get("ema_step", 0))
        self.ranks.update({name: int(value) for name, value in state.get("ranks", {}).items()})
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
        """Consume current gradients and apply a new sequential allocation."""
        self.update_sensitivity()
        return self.allocate(step)

    def allocate(self, step=None):
        """Allocate ranks using the most recently updated sensitivities."""
        self._refresh_spectral_shadow()
        ranks, energies = self._choose_ranks()
        self.ranks = OrderedDict(ranks)
        self._apply_allocation(self.ranks, energies)
        self.last_step = step
        return dict(self.ranks)

    def active_rank_summary(self):
        return dict(self.ranks)
