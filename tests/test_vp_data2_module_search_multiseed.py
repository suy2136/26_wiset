import argparse
import tempfile
import unittest
from pathlib import Path

from analysis import run_vp_data2_module_search_multiseed as pipeline


class VPData2ModuleSearchTests(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(
            compact_checkpoint=Path(root) / "compact",
            projector_checkpoint=Path(root) / "projector.pth",
            cache_dir=Path(root) / "cache",
            output_dir=Path(root) / "output",
            device="cuda:0", rank_budget=512, physical_rank=32,
            latency_warmup_steps=5, projector_cache_max_entries=512,
            max_mae_increase_ratio=0.03, final_quality_slack_ratio=0.01,
            resume=False, dry_run=True,
        )

    def test_search_space(self):
        self.assertEqual(len(pipeline.PATCH_CASES), 12)
        self.assertEqual(pipeline.TOKEN_K_VALUES, (2, 4, 6, 8, 10, 12))
        self.assertEqual(len(pipeline.SPECULATIVE_CONFIGS), 12)

    def test_protocol_always_uses_prescaled_qk(self):
        command = [
            "python", "run_plm.py", "--vp-fp16-fallback",
            "--seed", "9", "--data-seed", "9",
        ]
        result = pipeline.fixed_protocol(command, 2)
        self.assertIn("--vp-fp16-prescaled-qk", result)
        self.assertNotIn("--vp-fp16-fallback", result)
        self.assertEqual(result[result.index("--seed") + 1], "2")
        self.assertEqual(result[result.index("--data-seed") + 1], "2")
        self.assertEqual(
            result[result.index("--evaluation-rng-mode") + 1], "continuous"
        )

    def test_representatives_keep_quality_and_constrained_speed(self):
        rows = [
            {"case": "quality", "family": "token", "evaluation_seed": 1,
             "status": "complete", "mae": 9.8, "latency_mean_ms": 90},
            {"case": "speed", "family": "token", "evaluation_seed": 1,
             "status": "complete", "mae": 10.2, "latency_mean_ms": 50},
            {"case": "too_lossy", "family": "token", "evaluation_seed": 1,
             "status": "complete", "mae": 11.0, "latency_mean_ms": 30},
        ]
        selected = pipeline.representatives(rows, "token", 10.0, 0.03)
        self.assertEqual([row["case"] for row in selected], ["quality", "speed"])

    def test_manifest_marks_fixed_vp_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            value = pipeline.signature(self.args(root))
        self.assertEqual(value["evaluation_rng_mode"], "continuous")
        self.assertEqual(
            value["attention_score_mode"],
            "fp16_prescaled_qk_with_fp32_retry",
        )


if __name__ == "__main__":
    unittest.main()
