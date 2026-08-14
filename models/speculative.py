"""
Draft-model interface for speculative decoding, ported from
Soyun_ModuleHead's src/netllm_litevlm/speculative/{base,recent_velocity_draft}.py.
Used by models/speculative_pipeline.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class DraftOutput:
    coordinates: torch.Tensor
    forward_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinuousDraftModel(nn.Module, ABC):
    """Interface for continuous-coordinate VP draft models."""

    @abstractmethod
    def forward(
        self,
        history: torch.Tensor,
        steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> DraftOutput:
        raise NotImplementedError


class RecentVelocityDraft(ContinuousDraftModel):
    """Deterministic constant-velocity extrapolation in VP coordinate space.
    No LLM call -- pure tensor arithmetic (forward_count=1 is a bookkeeping
    constant, not a real model forward)."""

    def forward(
        self,
        history: torch.Tensor,
        steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> DraftOutput:
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("history must have shape [1,H,C]")
        if history.shape[1] < 2:
            raise ValueError("recent velocity requires at least two history steps")
        if steps <= 0:
            raise ValueError("steps must be positive")
        velocity = history[:, -1, :] - history[:, -2, :]
        offsets = torch.arange(
            1, steps + 1, device=history.device, dtype=history.dtype
        ).view(1, steps, 1)
        coordinates = history[:, -1:, :] + offsets * velocity.unsqueeze(1)
        return DraftOutput(
            coordinates=coordinates,
            forward_count=1,
            metadata={
                "draft": type(self).__name__,
                "deterministic": True,
                "learned": False,
                "context": dict(context) if context is not None else {},
            },
        )
