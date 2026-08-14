"""
Wraps this project's models.pipeline.Pipeline with an optional Selector
(models/selectors.py), applied once to the initial history sequence right
after embed_ln(), before the autoregressive generation loop.

Ported from the design in Soyun_ModuleHead's src/netllm_litevlm/vp/
selectable_pipeline.py (LlamaOldSelectablePipeline / SelectablePipeline), but
NOT copied against models/old/pipeline.py's non-cached, non-multimodal
EmbeddingModelViewportPrediction. This wrapper is built by composition against
our CURRENT models.pipeline.Pipeline instead:
  - reuses Pipeline's own embed_vp/conv1d/embed_ln/get_multimodal_information/
    plm modules and its existing KV-cached autoregressive loop verbatim
    (Pipeline.auto_regressive() already does incremental KV-cache decoding,
    unlike the old pipeline this pattern originated from -- see
    analysis/verify_selectable_pipeline_equivalence.py for the threshold=0
    gate re-measured against THIS loop, not the old one)
  - supports multimodal Pipeline instances (see the guard-removal note below)

Scope carried over from the original design (docs/experiment_phase/phase3a/
VP_EXTENSION_POLICY.md section 3): the selector runs exactly once, on the
LayerNorm'd initial history sequence, never on autoregressive feedback steps.
This wrapper is for inference/eval (auto_regressive/inference), not training
-- training still goes through Pipeline.forward()/teaching_forcing() directly
(with AdaLoRA active, if used), unchanged. The selector is an inference-time
context-length optimization, matching how Soyun_ModuleHead's own benchmark
harness (run_speculative_benchmark.py) only ever used it at eval time.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.selectors import BaseSelector, SelectionOutput


def build_selected_initial_sequence(
    pipeline: nn.Module,
    selector: Optional[BaseSelector],
    protect_multimodal_prefix: bool,
    x: torch.Tensor,
    video_user_position: Any,
) -> Dict[str, Any]:
    """
    Shared by LlamaSelectablePipeline and LlamaSpeculativeBlockVerifyPipeline
    (models/speculative_pipeline.py): builds the initial history sequence
    exactly as Pipeline.auto_regressive() does (trajectory embedding +
    optional multimodal token concat + embed_ln), then applies the optional
    selector once, with the same image-token-prefix protection. Kept in one
    place so both wrappers apply the selector identically instead of two
    near-duplicate implementations drifting apart.

    :return: dict with keys x, attention_mask (the possibly-selected initial
        sequence/mask), initial_embeddings, initial_attention_mask (pre-
        selection, for tracing/equivalence checks), num_image_tokens,
        protect_prefix, selection (SelectionOutput or None), selector_call_count.
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"expected batch-size-one [1,H,3] history, got {tuple(x.shape)}")

    history_viewports = x  # raw (B, his_window, 3), before conv1d embedding
    seq_len = x.shape[1]
    batch_embeddings = []
    for i in range(seq_len):
        batch_embeddings.append(
            pipeline.embed_vp(pipeline.conv1d(x[:, i, :]).view(1, 256)).unsqueeze(1)
        )
    traj = torch.cat(batch_embeddings, dim=1)

    num_image_tokens = 0
    if pipeline.using_multimodal:
        mapped_tensor = pipeline.get_multimodal_information(video_user_position, history_viewports)
        num_image_tokens = int(mapped_tensor.shape[1])
        x = torch.cat([mapped_tensor, traj], dim=1)
    else:
        x = traj

    x = pipeline.embed_ln(x)
    initial_embeddings = x
    attention_mask = torch.ones(
        x.shape[0], x.shape[1], dtype=torch.long, device=pipeline.device,
    )
    initial_attention_mask = attention_mask

    selector_call_count = 0
    protect_prefix = protect_multimodal_prefix and num_image_tokens > 0
    if selector is not None:
        if protect_prefix:
            image_part = x[:, :num_image_tokens, :]
            image_mask = attention_mask[:, :num_image_tokens]
            candidate_embeddings = x[:, num_image_tokens:, :]
            candidate_mask = attention_mask[:, num_image_tokens:]
        else:
            image_part = None
            image_mask = None
            candidate_embeddings = x
            candidate_mask = attention_mask

        selection = selector(
            candidate_embeddings,
            candidate_mask,
            context={
                "task": "viewport_prediction",
                "stage": "initial_history",
                "multimodal": pipeline.using_multimodal,
                "multimodal_mode": pipeline.multimodal_mode,
                "num_image_tokens_protected": num_image_tokens if protect_prefix else 0,
            },
        )
        selector_call_count += 1

        sel_embeddings = selection.embeddings
        sel_mask = selection.attention_mask
        if sel_mask is None:
            sel_mask = torch.ones(
                sel_embeddings.shape[0], sel_embeddings.shape[1],
                dtype=torch.long, device=sel_embeddings.device,
            )

        if image_part is not None:
            x = torch.cat([image_part, sel_embeddings], dim=1)
            attention_mask = torch.cat([image_mask, sel_mask], dim=1)
        else:
            x = sel_embeddings
            attention_mask = sel_mask
    else:
        selection = None

    if x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != pipeline.embed_size:
        raise ValueError(f"Selector returned invalid embeddings shape: {tuple(x.shape)}")
    if attention_mask.shape != x.shape[:2]:
        raise ValueError(
            "Selector attention mask does not match selected embeddings: "
            f"mask={tuple(attention_mask.shape)}, embeddings={tuple(x.shape)}"
        )

    return {
        "x": x,
        "attention_mask": attention_mask,
        "initial_embeddings": initial_embeddings,
        "initial_attention_mask": initial_attention_mask,
        "num_image_tokens": num_image_tokens,
        "protect_prefix": protect_prefix,
        "selection": selection,
        "selector_call_count": selector_call_count,
    }


class LlamaSelectablePipeline(nn.Module):
    def __init__(
        self,
        pipeline: nn.Module,
        selector: Optional[BaseSelector] = None,
        protect_multimodal_prefix: bool = True,
    ):
        """
        :param pipeline: a models.pipeline.Pipeline instance (any multimodal_mode).
        :param selector: optional BaseSelector, applied once to the initial
            history sequence (see module docstring).
        :param protect_multimodal_prefix: if True (default) and the pipeline is
            multimodal, image tokens are kept out of the selector's candidate
            pool entirely -- only the trajectory-embedding span is offered to
            the selector, and the image-token prefix is re-concatenated
            untouched afterward. Minimal fix for RecentK-style selectors, which
            rank by sequence position and would otherwise drop the image
            tokens first since they sit at the front of the sequence (found
            during Phase 0 investigation, 2026-08-13). Set False to let the
            selector see the full mixed sequence instead (e.g. for a future
            selector designed to be multimodal-aware on its own).
        #
        # using_multimodal guard: REMOVED (present in the original
        # Soyun_ModuleHead SelectablePipeline / SpeculativeBlockVerifyPipeline
        # as `if pipeline.using_multimodal: raise ValueError(...)`). Verified
        # safe to remove, not just deleted:
        #   - docs/experiment_phase/phase3a/VP_EXTENSION_POLICY.md lists
        #     "Non-multimodal" in section 4 ("현재 보존") alongside batch-size=1
        #     and no-KV-cache -- all of those are Phase-3A scope choices to
        #     keep an equivalence baseline stable, not claims that multimodal
        #     support is technically impossible. Section 6 ("후속 검토")
        #     explicitly lists "5. Multimodal selector" as a planned follow-up,
        #     and section 5 lists Patch/Frame/Image-Feature Selection as simply
        #     "not implemented yet", in the same category as Recent-K (which
        #     WAS later implemented in the same repo).
        #   - Code-level: the original guard lived on LlamaOldSelectablePipeline,
        #     whose own auto_regressive() embeds ONLY trajectory coordinates --
        #     it has no code path to build multimodal tokens at all, so the
        #     guard was really "this specific wrapper doesn't know how", not
        #     "multimodal breaks the Selector contract". The Selector contract
        #     itself (ascending selected_indices into the original sequence) is
        #     content-agnostic: it doesn't care whether a position holds an
        #     image token or a trajectory token. The KV-cache slicing in the
        #     speculative-decoding wrapper (models/speculative_pipeline.py) is
        #     likewise agnostic to what occupies the prefill -- it only tracks
        #     length. Since our Pipeline.auto_regressive() already builds
        #     [image_tokens, trajectory_tokens] before embed_ln() (unlike the
        #     old pipeline), this wrapper reuses that directly instead of
        #     needing new multimodal-token-fetch code of its own.
        """
        super().__init__()
        self.pipeline = pipeline
        self.selector = selector
        self.protect_multimodal_prefix = protect_multimodal_prefix
        self.last_selection_output: Optional[SelectionOutput] = None
        self.last_trace: Dict[str, Any] = {}

    def set_selector(self, selector: Optional[BaseSelector]) -> None:
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be a BaseSelector instance or None")
        self.selector = selector

    def auto_regressive(
        self,
        x: torch.Tensor,
        future: torch.Tensor,
        video_user_position: Any,
    ) -> torch.Tensor:
        pipeline = self.pipeline
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(
                f"LlamaSelectablePipeline preserves Pipeline's batch-size-one "
                f"contract; got {tuple(x.shape)}"
            )

        built = build_selected_initial_sequence(
            pipeline, self.selector, self.protect_multimodal_prefix, x, video_user_position,
        )
        x = built["x"]
        attention_mask = built["attention_mask"]
        initial_embeddings = built["initial_embeddings"]
        initial_attention_mask = built["initial_attention_mask"]
        num_image_tokens = built["num_image_tokens"]
        protect_prefix = built["protect_prefix"]
        selection = built["selection"]
        selector_call_count = built["selector_call_count"]
        self.last_selection_output = selection

        # From here down: EXACTLY Pipeline.auto_regressive()'s own KV-cached loop
        # (see models/pipeline.py), just starting from the (possibly selected) x
        # instead of the unselected one. This is what makes the selector=None /
        # IdentitySelector() path reproduce Pipeline's own output bit-for-bit.
        outputlist = []
        plm_dtype = pipeline.plm.get_input_embeddings().weight.dtype
        past_key_values = None
        current_input = x
        total_len = x.shape[1]
        for _ in range(pipeline.fut_window_length):
            step_attention_mask = torch.ones(
                x.shape[0], total_len, dtype=torch.long, device=pipeline.device,
            )
            outputs = pipeline.plm(
                inputs_embeds=current_input.to(plm_dtype),
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.float()
            outputlist.append(logits)
            past_key_values = outputs.past_key_values
            current_input = pipeline.embed_vp(pipeline.conv1d(logits)).unsqueeze(1)
            total_len += 1

        prediction = torch.cat(outputlist, dim=1)
        self.last_trace = {
            "selector_enabled": self.selector is not None,
            "selector_class": None if self.selector is None else type(self.selector).__name__,
            "selector_call_count": selector_call_count,
            "protect_multimodal_prefix": protect_prefix,
            "num_image_tokens": num_image_tokens,
            "initial_sequence_shape_before_selection": list(initial_embeddings.shape),
            "initial_attention_mask_shape_before_selection": list(initial_attention_mask.shape),
            "selected_length": int(x.shape[1]),
            "plm_forward_count": pipeline.fut_window_length,
            "cache_reused": True,  # Pipeline.auto_regressive() has always KV-cached; see module docstring
        }
        return prediction

    def inference(
        self,
        batch: torch.Tensor,
        future: torch.Tensor,
        video_user_info: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.auto_regressive(batch, future, video_user_info)
        ground_truth = future.to(prediction.device)
        return prediction, ground_truth

    def forward(
        self,
        batch: torch.Tensor,
        future: torch.Tensor,
        video_user_info: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inference(batch, future, video_user_info)
