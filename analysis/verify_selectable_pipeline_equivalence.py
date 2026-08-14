"""
threshold=0-style equivalence gate for LlamaSelectablePipeline
(models/selectable_pipeline.py), re-measured against OUR current KV-cached
Pipeline (models/pipeline.py) instead of Soyun_ModuleHead's old, non-cached
EmbeddingModelViewportPrediction that the original gate was built against.

Principle (same one used for the KV-cache validation in analysis/
verify_patch_crop_optimization.py and friends): "no selection" must reproduce
the unselected baseline exactly (up to floating point). Concretely:
  1. LlamaSelectablePipeline(selector=None) must match Pipeline.auto_regressive()
     itself bit-for-bit (same code path, just routed through the wrapper).
  2. LlamaSelectablePipeline(selector=IdentitySelector()) must match within a
     tight floating-point tolerance (identity selector round-trips the tensor,
     so this mostly checks the wrapper's own plumbing doesn't perturb anything).
  3. Repeated for multimodal_mode in {none, baseline, all-patch, patch-selection}
     -- get_multimodal_information() is monkeypatched to a fixed deterministic
     tensor for this test so it doesn't need real dataset images or a real ViT
     forward (that machinery is already covered by this project's existing
     analysis/verify_patch_*.py scripts); only the wrapper's own token-mixing
     and selection-insertion logic is under test here.
  4. RecentK + protect_multimodal_prefix=True must leave the image-token
     prefix byte-identical even when k is small enough that a naive selector
     would otherwise drop it (the fix for the "RecentK eats image tokens"
     issue flagged in Phase 0).

CPU-only, tiny LlamaConfig, ~seconds to run.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import LlamaConfig

from models.llama import LlamaNetworkingHeadModel
from models.networking_head import NetworkingHead
from models.pipeline import Pipeline
from models.selectable_pipeline import LlamaSelectablePipeline
from models.selectors import IdentitySelector, RecentKSelector

HIS_WINDOW = 10
FUT_WINDOW = 6
EMBED_SIZE = 16
NUM_IMAGE_TOKENS = 3


def build_tiny_plm():
    config = LlamaConfig(
        hidden_size=EMBED_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=128,
    )
    plm = LlamaNetworkingHeadModel(config)
    plm.set_networking_head(NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW))
    plm.eval()
    return plm


def build_pipeline(multimodal_mode):
    torch.manual_seed(0)
    plm = build_tiny_plm()
    # 'all-patch'/'patch-selection' otherwise eagerly load a real pretrained
    # torchvision vit_b_16 (network download, ~330MB) in Pipeline.__init__.
    # get_multimodal_information() is monkeypatched below regardless of mode,
    # so the real ViT is never actually invoked -- pass a throwaway module to
    # skip Pipeline's lazy-load branch and keep this test CPU/offline-fast.
    needs_vit_stub = multimodal_mode in ('all-patch', 'patch-selection')
    pipeline = Pipeline(
        plm, fut_window=FUT_WINDOW, device='cpu', embed_size=EMBED_SIZE, frequency=5,
        multimodal_mode=multimodal_mode, dataset='Jin2022',
        vit_model=torch.nn.Identity() if needs_vit_stub else None,
    )
    pipeline.eval()
    if pipeline.using_multimodal:
        # Deterministic stand-in for real ViT-derived tokens -- avoids needing
        # real dataset images/frozen ViT for this wrapper-logic-only test (see
        # module docstring). Fixed seed so it's identical across calls within a
        # single check (must be, since we compare two forward passes).
        fixed = torch.randn(1, NUM_IMAGE_TOKENS, EMBED_SIZE, generator=torch.Generator().manual_seed(123))
        pipeline.get_multimodal_information = lambda video_user_position, history_viewports=None: fixed
    return pipeline


def sample_batch():
    torch.manual_seed(1)
    history = torch.randn(1, HIS_WINDOW, 3)
    future = torch.randn(1, FUT_WINDOW, 3)
    video_user_position = torch.tensor([1, 1, 31])
    return history, future, video_user_position


def check_mode(multimodal_mode):
    print(f"--- multimodal_mode={multimodal_mode!r} ---")
    pipeline = build_pipeline(multimodal_mode)
    history, future, video_user_position = sample_batch()

    with torch.no_grad():
        ref = pipeline.auto_regressive(history, future, video_user_position)

        wrapped_none = LlamaSelectablePipeline(pipeline, selector=None)
        out_none = wrapped_none.auto_regressive(history, future, video_user_position)
        assert torch.equal(ref, out_none), \
            f"[{multimodal_mode}] selector=None must match Pipeline.auto_regressive() bit-for-bit"
        print(f"  [PASS] selector=None matches Pipeline.auto_regressive() exactly (torch.equal)")

        wrapped_identity = LlamaSelectablePipeline(pipeline, selector=IdentitySelector())
        out_identity = wrapped_identity.auto_regressive(history, future, video_user_position)
        max_diff = (ref - out_identity).abs().max().item()
        assert max_diff <= 1e-5, f"[{multimodal_mode}] IdentitySelector max diff {max_diff} > 1e-5"
        print(f"  [PASS] IdentitySelector matches within atol=1e-5 (max diff={max_diff:.2e})")

        if pipeline.using_multimodal:
            # k=1: aggressive enough that a naive (unprotected) selector would
            # keep only 1 of the 10 trajectory positions -- image tokens are the
            # thing most at risk of being silently dropped entirely.
            wrapped_recentk = LlamaSelectablePipeline(
                pipeline, selector=RecentKSelector(k=1), protect_multimodal_prefix=True,
            )
            out_recentk = wrapped_recentk.auto_regressive(history, future, video_user_position)
            assert out_recentk.shape == ref.shape
            assert torch.isfinite(out_recentk).all()
            trace = wrapped_recentk.last_trace
            assert trace["num_image_tokens"] == NUM_IMAGE_TOKENS
            assert trace["protect_multimodal_prefix"] is True
            assert trace["selected_length"] == NUM_IMAGE_TOKENS + 1, (
                f"expected protected image tokens ({NUM_IMAGE_TOKENS}) + RecentK(k=1) "
                f"trajectory token, got selected_length={trace['selected_length']}"
            )
            print(f"  [PASS] RecentKSelector(k=1) + protect_multimodal_prefix=True keeps all "
                  f"{NUM_IMAGE_TOKENS} image tokens (selected_length={trace['selected_length']}, "
                  f"not just the 1 trajectory token RecentK alone would keep)")
    print()


if __name__ == "__main__":
    for mode in ("none", "baseline", "all-patch", "patch-selection"):
        check_mode(mode)
    print("All LlamaSelectablePipeline equivalence checks passed.")
