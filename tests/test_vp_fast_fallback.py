import argparse
from pathlib import Path
import unittest

from analysis import evaluate_vp_fast_fallback_fullstack_multiseed as pipeline

try:
    import torch
    import torch.nn as nn
    from models.nbs_compaction import CompactLoRALinear
    from models.vp_numeric_safety import enable_vp_fp16_fallback, vp_safe_retry
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for numeric-path tests")
class VPFastFallbackTest(unittest.TestCase):
    def compact(self):
        class SourceLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.eye(2), requires_grad=False)
                self.bias = None
                self.fan_in_fan_out = False

        return CompactLoRALinear(
            SourceLinear(),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0], [0.0]]),
            1.0,
            nn.Identity(),
        )

    def test_compact_normal_path_does_not_collect_layer_diagnostics(self):
        module = self.compact()
        actual = module(torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(actual, torch.tensor([[4.0, 3.0]]))
        self.assertFalse(hasattr(module, "_nbs_last_precast_absmax"))

    def test_retry_context_enables_and_restores_detailed_compact_safety(self):
        root = nn.Sequential(self.compact())
        self.assertFalse(root[0]._vp_detailed_safety_active)
        with vp_safe_retry(root, "fp16_prescaled"):
            self.assertTrue(root[0]._vp_detailed_safety_active)
            root(torch.tensor([[2.0, 3.0]]))
            self.assertTrue(hasattr(root[0], "_nbs_last_precast_absmax"))
        self.assertFalse(root[0]._vp_detailed_safety_active)

    def test_enable_targets_only_vp_networking_model(self):
        class LlamaNetworkingHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.Linear(1, 1)

        model = nn.Sequential(LlamaNetworkingHeadModel(), nn.Linear(1, 1))
        self.assertEqual(enable_vp_fp16_fallback(model), 1)
        self.assertTrue(model[0]._vp_fp16_fallback_enabled)
        self.assertFalse(hasattr(model[1], "_vp_fp16_fallback_enabled"))


class VPFixedPipelineTest(unittest.TestCase):
    def test_vp_fallback_keeps_a_separate_normal_fast_path(self):
        compact_source = Path("models/nbs_compaction.py").read_text(
            encoding="utf-8"
        )
        llama_source = Path("models/llama.py").read_text(encoding="utf-8")
        self.assertIn("if not self._vp_detailed_safety_active:", compact_source)
        self.assertIn("return result_fp32.to(result.dtype)", compact_source)
        self.assertIn('vp_safe_retry(self.model, "fp16_prescaled")', llama_source)
        self.assertIn('vp_safe_retry(self.model, "fp32")', llama_source)

    def test_fixed_pipeline_command_has_requested_configuration(self):
        args = argparse.Namespace(
            compact_checkpoint=Path("compact"),
            projector_checkpoint=Path("projector_2ep.pth"),
            cache_dir=Path("cache"),
            output_dir=Path("output"),
            device="cuda:0",
            rank_budget=512,
            physical_rank=32,
            latency_warmup_steps=5,
            projector_cache_max_entries=512,
        )
        command = pipeline.command_for(args, 3, Path("result"))
        self.assertEqual(command[command.index("--adalora-rank-budget") + 1], "512")
        self.assertEqual(command[command.index("--rank") + 1], "32")
        self.assertEqual(command[command.index("--cached-patch-policy") + 1], "gated-k1")
        self.assertEqual(command[command.index("--cached-patch-motion-threshold-deg") + 1], "6.0")
        self.assertEqual(command[command.index("--cached-patch-max-skip-calls") + 1], "0")
        self.assertIn("--cached-patch-projector-cache", command)
        self.assertEqual(command[command.index("--selector-recent-k") + 1], "8")
        self.assertEqual(command[command.index("--speculative-gamma") + 1], "6")
        self.assertEqual(command[command.index("--speculative-threshold") + 1], "0.4")
        self.assertEqual(command[command.index("--data-seed") + 1], "3")
        self.assertEqual(command[command.index("--evaluation-rng-mode") + 1], "continuous")

    def test_summary_counts_fallbacks(self):
        rows = [
            {
                "evaluation_seed": seed,
                "status": "complete",
                "mae": 17.0 + seed / 10,
                "rmse": 31.0,
                "latency_mean_ms": 200.0 + seed,
                "fp16_prescaled_qk_fallback_calls": seed == 3,
                "fp32_attention_fallback_calls": 0,
            }
            for seed in pipeline.SEEDS
        ]
        summary = pipeline.summarize(rows)
        self.assertEqual(summary["fp16_prescaled_qk_fallback_calls_total"], 1)
        self.assertEqual(summary["fp32_attention_fallback_calls_total"], 0)


if __name__ == "__main__":
    unittest.main()
