from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from analysis import run_server1_vp_data3_pipeline as pipeline


class Server1VpData3PipelineTest(unittest.TestCase):
    def test_training_variants_and_stage_order(self):
        self.assertEqual(
            {value["variant"] for value in pipeline.METHODS.values()},
            {
                "uniform_r8_data3", "adalora_b512_data3",
                "shapley_b512_data3", "eva_b512_data3", "nbs_v19_data3",
            },
        )
        self.assertLess(
            pipeline.STAGE_ORDER.index("train_adalora"),
            pipeline.STAGE_ORDER.index("evaluate_adalora"),
        )
        self.assertLess(
            pipeline.STAGE_ORDER.index("compact_nbs"),
            pipeline.STAGE_ORDER.index("evaluate_nbs_modules"),
        )

    def test_module_command_uses_prescaled_qk_and_continuous_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                device="cuda:0", latency_warmup_steps=5,
                tuned_projector=root / "projector.pth",
                patch_cache=root / "cache",
            )
            command = pipeline.module_command(
                args, root / "compact", "full_stack", 3, root / "result",
            )
        self.assertIn("--vp-fp16-prescaled-qk", command)
        self.assertNotIn("--vp-fp16-fallback", command)
        self.assertEqual(command[command.index("--seed") + 1], "3")
        self.assertEqual(command[command.index("--data-seed") + 1], "3")
        self.assertEqual(command[command.index("--lora-seed") + 1], "1")
        self.assertEqual(
            command[command.index("--evaluation-rng-mode") + 1], "continuous",
        )

    def test_exact_budget_rejects_nonmatching_checkpoint(self):
        original = pipeline.vp_lora.checkpoint_description
        pipeline.vp_lora.checkpoint_description = lambda method, checkpoint: {
            "method": method, "active_rank_total": 511,
        }
        try:
            with self.assertRaisesRegex(ValueError, "expected exactly 512"):
                pipeline.inspect_exact_budget("adalora", Path("checkpoint"))
        finally:
            pipeline.vp_lora.checkpoint_description = original

    def test_shell_exposes_data3_and_opt_in_training_safety(self):
        shell = (pipeline.REPO_ROOT / "scripts/run_netllm_experiment.sh").read_text(
            encoding="utf-8"
        )
        for variant in (value["variant"] for value in pipeline.METHODS.values()):
            self.assertIn(variant, shell)
        self.assertIn('VP_FP16_PRESCALED_QK="${VP_FP16_PRESCALED_QK:-0}"', shell)
        self.assertIn('TRAIN_CMD+=(--vp-fp16-prescaled-qk)', shell)


if __name__ == "__main__":
    unittest.main()
