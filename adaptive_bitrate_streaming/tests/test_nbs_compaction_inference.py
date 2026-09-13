import argparse
from pathlib import Path
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # The lightweight local test environment omits torch.
    torch = None
    nn = None


ABR_ROOT = Path(__file__).resolve().parents[1]


if nn is not None:
    class _SourceLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(12, dtype=torch.float32).reshape(3, 4) / 20
            )
            self.bias = nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
            self.fan_in_fan_out = False


class NBSCompactionInferenceTest(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "torch is not installed")
    def test_compact_linear_matches_active_factor_equation(self):
        from models.nbs_compaction import CompactLoRALinear

        source = _SourceLinear()
        lora_a = torch.tensor([[0.2, -0.1, 0.3, 0.4], [0.5, 0.2, -0.2, 0.1]])
        lora_b = torch.tensor([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])
        layer = CompactLoRALinear(
            source, lora_a, lora_b, adapter_scale=0.75,
            dropout=nn.Identity(),
        )
        x = torch.tensor([[0.3, -0.7, 0.2, 0.9]])
        expected = torch.nn.functional.linear(x, source.weight, source.bias)
        expected = expected.float() + (x @ lora_a.T @ lora_b.T).float() * 0.75
        torch.testing.assert_close(layer(x), expected)

    @unittest.skipUnless(torch is not None, "torch is not installed")
    def test_compact_fp16_projection_clamps_finite_overflow(self):
        from models.nbs_compaction import CompactLoRALinear

        source = _SourceLinear().half()
        source.weight.data.zero_()
        source.bias.data.zero_()
        source._abr_fp16_selective_clamp = True
        source._abr_fp16_clamp_threshold = 60000.0
        layer = CompactLoRALinear(
            source,
            torch.full((1, 4), 300.0, dtype=torch.float16),
            torch.full((3, 1), 300.0, dtype=torch.float16),
            adapter_scale=1.0,
            dropout=nn.Identity(),
        )
        output = layer(torch.ones((1, 4), dtype=torch.float16))
        self.assertTrue(bool(torch.isfinite(output).all().item()))
        self.assertLessEqual(float(output.abs().max().item()), 60000.0)
        self.assertTrue(bool(layer._nbs_last_precast_finite.item()))

    def test_f_runner_switches_dense_and_compact_modes(self):
        from adaptive_bitrate_streaming.analysis.run_nbs_v19_f_compaction_validation import (
            build_test_command,
        )

        args = argparse.Namespace(
            base_model_dir=Path("base"), exp_pool_path=Path("pool.pkl"),
            trace="fcc-test", trace_num=100, video="video1", device="cuda:0",
        )
        checkpoint = Path("checkpoint")
        dense = build_test_command(args, checkpoint, compact=False)
        compact = build_test_command(args, checkpoint, compact=True)
        self.assertIn("--no-nbs-compact-inference", dense)
        self.assertIn("--nbs-compact-inference", compact)
        self.assertNotIn("--nbs-compact-inference", dense)

    def test_run_plm_defaults_nbs_test_to_compaction(self):
        source = (ABR_ROOT / "run_plm.py").read_text(encoding="utf-8")
        self.assertIn("args.test and args.nbs_v19", source)
        self.assertIn("compact_nbs_model_for_inference", source)
        self.assertIn("nbs_compaction_equivalence.json", source)
        self.assertIn("'nbs_compact'", source)
        self.assertIn("'nbs_dense'", source)


if __name__ == "__main__":
    unittest.main()
