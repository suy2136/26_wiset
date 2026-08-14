"""
Selector contract for pruning an already-embedded token sequence before it's fed
to the LLM, ported from the Soyun_ModuleHead handoff package
(src/netllm_litevlm/selectors/{base,identity,recent_k,attention_topk}.py) into
this project's own models/ package so LlamaSelectablePipeline (see
models/selectable_pipeline.py) has no dependency on that external repo.

This is a DIFFERENT, independent stage from models/patch_selection.py:
  - patch_selection.PatchSelectionModule decides which raw image grid patches
    to crop and run through the frozen ViT, BEFORE any embedding happens (its
    whole point is to skip ViT compute on unselected patches).
  - BaseSelector below prunes an ALREADY-embedded [1, L, E] sequence (image
    tokens + trajectory tokens, whatever Pipeline.auto_regressive() already
    concatenated), AFTER embed_ln(), right before the LLM sees it.
They are sequential, composable pipeline stages, not alternative
implementations of the same job -- patch_selection does not need to (and does
not) implement this contract. See LlamaSelectablePipeline for where the two
meet.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


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
    """Base interface for selecting sequence embeddings.

    Contract (see LlamaSelectablePipeline for how it's enforced at the call
    site): embeddings is [1, L, E] (batch size always 1, this project's
    pipeline processes one sample at a time); selected_indices must be
    strictly increasing into the original sequence (out-of-order selection
    would corrupt the causal structure the LLM was trained under); the
    feature dim E is never touched, only the sequence axis L.
    """

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
    """Return the input sequence without pruning, reordering, or mutation.

    Used as the equivalence-gate reference: LlamaSelectablePipeline(selector=
    IdentitySelector()) must reproduce Pipeline.auto_regressive()'s own output
    exactly (see analysis/verify_selectable_pipeline_equivalence.py).
    """

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        sequence_length = int(embeddings.shape[1])
        selected_indices = torch.arange(
            sequence_length, dtype=torch.long, device=embeddings.device,
        )
        return SelectionOutput(
            embeddings=embeddings,
            attention_mask=attention_mask,
            selected_indices=selected_indices,
            scores=None,
            original_length=sequence_length,
            selected_length=sequence_length,
            metadata={
                "selector": type(self).__name__,
                "preserves_order": True,
                "context": dict(context) if context is not None else {},
            },
        )


class RecentKSelector(BaseSelector):
    """Keep the most recent ``k`` embeddings without changing their order.

    NOTE: "most recent" means highest sequence position. When called on a
    mixed [image_tokens, trajectory_tokens] sequence, this would drop image
    tokens first (they occupy the earliest positions) -- LlamaSelectablePipeline
    avoids that by only ever handing this selector the trajectory-token span,
    protecting the image-token prefix outside the selector entirely (see
    models/selectable_pipeline.py's protect_multimodal_prefix).
    """

    def __init__(self, k: int):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        original_length = int(embeddings.shape[1])
        if self.k > original_length:
            raise ValueError(f"k={self.k} exceeds sequence length {original_length}")

        start = original_length - self.k
        selected_indices = torch.arange(
            start, original_length, dtype=torch.long, device=embeddings.device,
        )
        return SelectionOutput(
            embeddings=embeddings[:, start:, :],
            attention_mask=(None if attention_mask is None else attention_mask[:, start:]),
            selected_indices=selected_indices,
            scores=None,
            original_length=original_length,
            selected_length=self.k,
            metadata={
                "selector": type(self).__name__,
                "k": self.k,
                "preserves_order": True,
                "selection_policy": "most_recent",
                "context": dict(context) if context is not None else {},
            },
        )
