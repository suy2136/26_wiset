"""Inference-time token selectors for NetLLM's ABR policy.

The public selector contract mirrors the viewport-prediction implementation,
but ABR history is selected in complete timestep blocks.  One completed ABR
timestep contains ``[return, six state tokens, action]`` (8 tokens), while the
current timestep contains ``[return, six state tokens]`` (7 protected tokens).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from plm_special.models.selection_layout import recent_timestep_window


@dataclass
class SelectionOutput:
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    selected_indices: torch.Tensor
    scores: Optional[torch.Tensor]
    original_length: int
    selected_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSelector(nn.Module, ABC):
    """Select positions from an already embedded ``[B, L, E]`` sequence."""

    @staticmethod
    def validate_inputs(
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must have shape [B,L,E], got {tuple(embeddings.shape)}"
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape [B,L], "
                    f"got {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape != embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask shape must match embeddings [B,L]: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"embeddings={tuple(embeddings.shape)}"
                )
            if attention_mask.device != embeddings.device:
                raise ValueError(
                    "attention_mask and embeddings must be on the same device"
                )

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        raise NotImplementedError


class IdentitySelector(BaseSelector):
    """Equivalence reference that keeps every token unchanged."""

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        length = int(embeddings.shape[1])
        indices = torch.arange(length, device=embeddings.device, dtype=torch.long)
        return SelectionOutput(
            embeddings=embeddings,
            attention_mask=attention_mask,
            selected_indices=indices,
            scores=None,
            original_length=length,
            selected_length=length,
            metadata={
                "selector": type(self).__name__,
                "preserves_order": True,
                "context": dict(context) if context is not None else {},
            },
        )


class RecentTimestepSelector(BaseSelector):
    """Keep recent complete ABR history blocks and the full current block.

    The selector is deliberately inference-only for the first integration.
    It never splits a historical timestep and never removes the current state,
    whose final token is consumed by the ABR action head.
    """

    def __init__(
        self,
        history_steps: int,
        tokens_per_history_step: int = 8,
        current_step_tokens: int = 7,
    ):
        super().__init__()
        for name, value in (
            ("history_steps", history_steps),
            ("tokens_per_history_step", tokens_per_history_step),
            ("current_step_tokens", current_step_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.history_steps = history_steps
        self.tokens_per_history_step = tokens_per_history_step
        self.current_step_tokens = current_step_tokens

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        context = {} if context is None else dict(context)
        protected_suffix_tokens = context.get(
            "protected_suffix_tokens", self.current_step_tokens
        )
        if (
            isinstance(protected_suffix_tokens, bool)
            or not isinstance(protected_suffix_tokens, int)
            or protected_suffix_tokens <= 0
        ):
            raise ValueError("protected_suffix_tokens must be a positive integer")

        original_length = int(embeddings.shape[1])
        start, available_steps, selected_steps = recent_timestep_window(
            original_length=original_length,
            history_steps=self.history_steps,
            tokens_per_history_step=self.tokens_per_history_step,
            current_step_tokens=protected_suffix_tokens,
        )
        selected_indices = torch.arange(
            start, original_length, device=embeddings.device, dtype=torch.long,
        )
        selected_mask = (
            None if attention_mask is None else attention_mask[:, start:]
        )
        return SelectionOutput(
            embeddings=embeddings[:, start:, :],
            attention_mask=selected_mask,
            selected_indices=selected_indices,
            scores=None,
            original_length=original_length,
            selected_length=original_length - start,
            metadata={
                "selector": type(self).__name__,
                "available_history_steps": available_steps,
                "selected_history_steps": selected_steps,
                "tokens_per_history_step": self.tokens_per_history_step,
                "protected_suffix_tokens": protected_suffix_tokens,
                "preserves_timestep_blocks": True,
                "preserves_order": True,
                "context": context,
            },
        )
