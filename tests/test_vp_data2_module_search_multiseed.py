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
        self.assertEqual(pipeline.SEARCH_SEED, 2)
        self.assertEqual(pipeline.CONFIRMATION_SEEDS, (1, 3))
        self.assertEqual(len(pipeline.PATCH_CASES), 3)
        self.assertEqual(pipeline.TOKEN_K_VALUES, (4, 6, 10))
        self.assertEqual(len(pipeline.SPECULATIVE_CONFIGS), 5)

    def test_protocol_uses_fast_path_with_prescaled_qk_fallback(self):
        command = [
            "python", "run_plm.py", "--vp-fp16-prescaled-qk",
            "--seed", "9", "--data-seed", "9",
        ]
        result = pipeline.fixed_protocol(command, 2)
        self.assertIn("--vp-fp16-fallback", result)
        self.assertNotIn("--vp-fp16-prescaled-qk", result)
        self.assertEqual(result[result.index("--seed") + 1], "2")
        self.assertEqual(result[result.index("--data-seed") + 1], "2")
        self.assertEqual(
            result[result.index("--evaluation-rng-mode") + 1], "continuous"
        )

    def test_representatives_keep_quality_and_constrained_speed(self):
        rows = [
            {"case": "quality", "family": "token", "evaluation_seed": 2,
             "status": "complete", "mae": 9.8, "latency_mean_ms": 90},
            {"case": "speed", "family": "token", "evaluation_seed": 2,
             "status": "complete", "mae": 10.2, "latency_mean_ms": 50},
            {"case": "too_lossy", "family": "token", "evaluation_seed": 2,
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
            "fp16_fast_with_prescaled_qk_fallback",
        )

    def test_full_stack_combinations_are_capped_and_diverse(self):
        selected = {
            family: [
                {"case": f"{family}_quality"},
                {"case": f"{family}_speed"},
            ]
            for family in ("patch", "token", "speculative")
        }
        combinations = pipeline.full_stack_combinations(selected)
        self.assertEqual(len(combinations), 4)
        identities = [tuple(row["case"] for row in item) for item in combinations]
        self.assertEqual(len(set(identities)), 4)


if __name__ == "__main__":
    unittest.main()
