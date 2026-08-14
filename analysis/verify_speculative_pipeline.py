"""
threshold=0 equivalence gate + forward-count/latency measurement for
LlamaSpeculativeBlockVerifyPipeline (models/speculative_pipeline.py),
re-measured against OUR current KV-cached Pipeline.auto_regressive(), not
Soyun_ModuleHead's non-cached old-pipeline baseline their "~4.7x" figure was
measured against (see module docstring in speculative_pipeline.py for why
that number isn't expected to reproduce here).

CPU-only, tiny (untrained, random-init) LlamaConfig -- this verifies the
MECHANISM (equivalence at threshold=0, and that forward count/latency respond
to threshold/gamma the way they should) and gives real measured numbers for
THIS harness. It is explicitly NOT a real accuracy or production-latency
benchmark: the tiny model's predictions are structurally uncorrelated random
noise, so acceptance rates here are an artifact of the toy setup, not
predictive of the real Llama2-7B checkpoint's behavior (same caveat
Soyun_ModuleHead's own PHASE_A_DESIGN.md attaches to its random-weight smoke
tests). Real numbers come once this is run against a trained checkpoint on GPU
(later step).
"""
import sys
import os
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))  # project root, for `models.*` imports
sys.path.insert(0, _THIS_DIR)  # this dir, for the sibling verify_selectable_pipeline_equivalence import

import torch

from verify_selectable_pipeline_equivalence import (
    build_pipeline, sample_batch, FUT_WINDOW,
)
from models.speculative_pipeline import LlamaSpeculativeBlockVerifyPipeline

GAMMA = 3
THRESHOLDS = (0.0, 0.1, 0.5, 1.5)
TIMING_REPEATS = 20


def time_call(fn, repeats=TIMING_REPEATS):
    # warmup (excluded from timing, same convention as Suhyeon_adalora's
    # latency_benchmark.py)
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed_ms = (time.perf_counter() - start) / repeats * 1000.0
    return elapsed_ms


def check_equivalence(pipeline, history, future, video_user_position, mode):
    with torch.no_grad():
        ref = pipeline.auto_regressive(history, future, video_user_position)

        spec = LlamaSpeculativeBlockVerifyPipeline(
            pipeline, selector=None, draft_model=None, gamma=GAMMA, acceptance_threshold=0.0,
        )
        out = spec.auto_regressive(history, video_user_position)

    max_diff = (ref - out).abs().max().item()
    assert max_diff <= 1e-5, f"[{mode}] threshold=0 max diff {max_diff} > 1e-5"
    assert spec.target_forward_count == FUT_WINDOW, (
        f"[{mode}] threshold=0 must cost exactly fut_window={FUT_WINDOW} target forwards "
        f"(every draft rejected by construction), got {spec.target_forward_count}"
    )
    assert sum(spec.accepted_per_iteration) == 0, (
        f"[{mode}] threshold=0 must reject every draft, got accepted_per_iteration="
        f"{spec.accepted_per_iteration}"
    )
    print(f"  [PASS] threshold=0 equivalence: max diff={max_diff:.2e} (atol=1e-5), "
          f"target_forward_count={spec.target_forward_count} (== fut_window), "
          f"sum(accepted)=0")


def measure_forward_counts_and_latency(pipeline, history, future, video_user_position, mode):
    baseline_latency_ms = time_call(lambda: pipeline.auto_regressive(history, future, video_user_position))
    print(f"  baseline (speculative OFF): forward_count={FUT_WINDOW} (by construction, one "
          f"cached forward per new token), latency={baseline_latency_ms:.3f} ms/sample "
          f"(mean of {TIMING_REPEATS}, tiny untrained CPU model)")

    for threshold in THRESHOLDS:
        spec = LlamaSpeculativeBlockVerifyPipeline(
            pipeline, selector=None, draft_model=None, gamma=GAMMA, acceptance_threshold=threshold,
        )

        def run():
            spec.auto_regressive(history, video_user_position)

        latency_ms = time_call(run)
        # forward count from the LAST call (deterministic model + deterministic
        # draft + fixed input -> same every call)
        print(f"  speculative ON  threshold={threshold:<4} gamma={GAMMA}: "
              f"target_forward_count={spec.target_forward_count:2d} "
              f"(vs baseline {FUT_WINDOW}), accepted_per_iteration={spec.accepted_per_iteration}, "
              f"latency={latency_ms:.3f} ms/sample")


def run_mode(mode):
    print(f"=== multimodal_mode={mode!r} ===")
    pipeline = build_pipeline(mode)
    history, future, video_user_position = sample_batch()
    check_equivalence(pipeline, history, future, video_user_position, mode)
    with torch.no_grad():
        measure_forward_counts_and_latency(pipeline, history, future, video_user_position, mode)
    print()


if __name__ == "__main__":
    for mode in ("baseline", "all-patch", "patch-selection"):
        run_mode(mode)
    print("All LlamaSpeculativeBlockVerifyPipeline checks passed.")
