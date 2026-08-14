"""
Block-verified speculative decoding, composition-based against
models/pipeline.py::Pipeline (same pattern as models/selectable_pipeline.py's
LlamaSelectablePipeline, which this reuses for the initial-history + optional
selector step) -- NOT copied from Soyun_ModuleHead's models/old/pipeline.py
-based block_verify.py::SpeculativeBlockVerifyPipeline.

Mechanically the same idea (draft `gamma` future coordinates with a cheap
non-LLM draft model, verify them against the real LLM in a single forward
that embeds `[carry, draft_0..draft_{gamma-1}]` at once, accept the agreeing
prefix, roll back the KV cache for the rejected tail), but built against a
pipeline whose OWN unmodified autoregressive loop already KV-caches (see
models/pipeline.py::Pipeline.auto_regressive() -- unlike Soyun_ModuleHead's
target, models/old/pipeline.py::EmbeddingForViewportPrediction, which
reprocesses the whole growing sequence from scratch every step with no cache
at all). That difference matters for what "speedup" means here:

  Soyun's own measured "~4-5x forward-count reduction, ~4.7x latency" (see
  docs/experiment_phase/speculative/PHASE_B_REAL_RESULTS.md in the
  Soyun_ModuleHead package) compares block verification against a baseline
  that both lacks caching AND does more forwards -- so it captures two
  effects at once (cache-vs-no-cache, plus fewer forwards). Our baseline
  (Pipeline.auto_regressive()) already has the caching half; block
  verification here only contributes the SECOND effect (fewer forward calls:
  ~fut_window/gamma forwards of an amortized-but-still-linear-per-position
  cost, instead of fut_window forwards of one-new-token-per-call). The
  measured numbers are in analysis/verify_speculative_pipeline.py's
  benchmark output, not assumed from Soyun's report.

Uses the same technique as block_verify.py to get a prediction for every new
position in a single forward (not just the last one, which is all
NetworkingHead.forward()'s default slicing exposes): request
output_hidden_states=True and apply the raw `networking_head.networking_head`
Linear+Tanh submodule directly to the full per-position hidden states,
bypassing the outer NetworkingHead.forward()'s single-last-position slice.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.selectable_pipeline import build_selected_initial_sequence
from models.selectors import BaseSelector, SelectionOutput
from models.speculative import ContinuousDraftModel, RecentVelocityDraft

PastKeyValues = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def slice_past_key_values(past_key_values: PastKeyValues, keep_length: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Truncate a legacy tuple-of-(key,value) KV cache to `keep_length` positions.

    transformers==4.34.1 (this project's pinned version for the real
    checkpoint -- see requirements-vp.txt / analysis/verify_adalora_equivalence.py's
    own environment checks) predates the Cache/DynamicCache object; LlamaModel
    returns this tuple-of-tuples format, each entry [B, num_heads, seq_len, head_dim].
    """
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "unsupported past_key_values type for rollback: "
            f"{type(past_key_values)!r}; expected the legacy tuple-of-(key,value) "
            "format used by transformers==4.34.1"
        )
    return tuple(
        (key[:, :, :keep_length, :].contiguous(), value[:, :, :keep_length, :].contiguous())
        for key, value in past_key_values
    )


class LlamaSpeculativeBlockVerifyPipeline(nn.Module):
    def __init__(
        self,
        pipeline: nn.Module,
        selector: Optional[BaseSelector] = None,
        protect_multimodal_prefix: bool = True,
        draft_model: Optional[ContinuousDraftModel] = None,
        gamma: int = 4,
        # L2 distance between draft and target networking-head outputs, in the
        # head's own Tanh-bounded normalized output space (~[-1,1] per
        # channel), NOT denormalized degrees -- same convention as
        # Soyun_ModuleHead's block_verify.py. Denormalize the final prediction
        # (utils.normalize.denormalize_data) after this pipeline returns, same
        # as everywhere else in this codebase.
        acceptance_threshold: float = 0.0,
    ):
        super().__init__()
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be BaseSelector or None")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if acceptance_threshold < 0:
            raise ValueError("acceptance_threshold must be non-negative")
        self.pipeline = pipeline
        self.selector = selector
        self.protect_multimodal_prefix = protect_multimodal_prefix
        self.draft_model = draft_model if draft_model is not None else RecentVelocityDraft()
        self.gamma = int(gamma)
        self.acceptance_threshold = float(acceptance_threshold)

        self.last_selection_output: Optional[SelectionOutput] = None
        self.last_trace: Dict[str, Any] = {}
        self.target_forward_count = 0
        self.draft_forward_count = 0
        self.accepted_per_iteration: List[int] = []

    def set_selector(self, selector: Optional[BaseSelector]) -> None:
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be a BaseSelector instance or None")
        self.selector = selector

    def _embed_coordinate(self, pipeline, value: torch.Tensor) -> torch.Tensor:
        """Embed a single [1,1,3] coordinate the same way Pipeline.auto_regressive()
        embeds feedback: conv1d -> embed_vp, no embed_ln (feedback is never
        renormalized -- see models/pipeline.py, matches the original policy)."""
        return pipeline.embed_vp(pipeline.conv1d(value)).unsqueeze(1)

    def auto_regressive(
        self, history: torch.Tensor, video_user_position: Any,
    ) -> torch.Tensor:
        pipeline = self.pipeline
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError(f"requires batch-size-one [1,H,3] history; got {tuple(history.shape)}")

        built = build_selected_initial_sequence(
            pipeline, self.selector, self.protect_multimodal_prefix, history, video_user_position,
        )
        sequence = built["x"]
        attention_mask = built["attention_mask"]
        self.last_selection_output = built["selection"]

        self.target_forward_count = 0
        self.draft_forward_count = 0
        self.accepted_per_iteration = []

        plm_dtype = pipeline.plm.get_input_embeddings().weight.dtype
        networking_head_linear = pipeline.plm.networking_head.networking_head  # raw Linear+Tanh, bypasses single-position slicing

        # Initial warmup forward: seeds the KV cache and produces the first carry
        # (the prediction for the last position of the initial sequence).
        result = pipeline.plm(
            inputs_embeds=sequence.to(plm_dtype),
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.target_forward_count += 1
        cache = result.past_key_values
        cache_len = int(sequence.shape[1])
        carry = networking_head_linear(result.hidden_states[-1][:, -1:, :].float())

        fut_window = pipeline.fut_window_length
        confirmed = [carry]

        while len(confirmed) < fut_window:
            remaining = fut_window - len(confirmed)
            gamma = min(self.gamma, remaining)

            draft_history = torch.cat((history, *confirmed), dim=1)
            draft = self.draft_model(draft_history, steps=gamma)

            chunk_values = [carry] + [draft.coordinates[:, i:i + 1, :] for i in range(gamma)]
            chunk_embeds = torch.cat(
                [self._embed_coordinate(pipeline, v) for v in chunk_values], dim=1,
            )
            full_mask = torch.ones(
                (1, cache_len + chunk_embeds.shape[1]), dtype=attention_mask.dtype, device=sequence.device,
            )

            result = pipeline.plm(
                inputs_embeds=chunk_embeds.to(plm_dtype),
                attention_mask=full_mask,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            self.target_forward_count += 1
            preds = networking_head_linear(result.hidden_states[-1].float())  # (1, gamma+1, 3)

            accepted = 0
            for k in range(gamma):
                error = torch.linalg.vector_norm(
                    (preds[:, k, :] - draft.coordinates[:, k, :]).float(), ord=2,
                )
                if error.item() > self.acceptance_threshold:
                    break
                accepted += 1
            self.accepted_per_iteration.append(accepted)

            for k in range(accepted):
                confirmed.append(preds[:, k:k + 1, :])

            commit_len = 1 + accepted
            if accepted < gamma:
                bonus = preds[:, accepted:accepted + 1, :]
                confirmed.append(bonus)
                carry = bonus
            else:
                carry = preds[:, gamma:gamma + 1, :]

            cache = slice_past_key_values(result.past_key_values, cache_len + commit_len)
            cache_len += commit_len

        prediction = torch.cat(confirmed, dim=1)[:, :fut_window, :]
        self.last_trace = {
            "selector_call_count": built["selector_call_count"],
            "protect_multimodal_prefix": built["protect_prefix"],
            "num_image_tokens": built["num_image_tokens"],
            "target_forward_count": self.target_forward_count,
            "draft_forward_count": self.draft_forward_count,
            "accepted_per_iteration": list(self.accepted_per_iteration),
            "gamma": self.gamma,
            "acceptance_threshold": self.acceptance_threshold,
            "cache_reused": True,
            "final_cache_length": cache_len,
            "prediction_shape": list(prediction.shape),
        }
        return prediction

    def inference(
        self, history: torch.Tensor, future: torch.Tensor, video_user_position: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.auto_regressive(history, video_user_position)
        return prediction, future.to(prediction.device)

    def forward(
        self, history: torch.Tensor, future: torch.Tensor, video_user_position: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inference(history, future, video_user_position)
