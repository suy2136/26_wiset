import unittest
from pathlib import Path

from adaptive_bitrate_streaming.analysis.run_abr_c1536_data2_safeguarded_allocators import (
    EXPERIMENTS,
)
from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
    build_training_command,
    parse_args,
)


class SafeguardedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.args = parse_args(
            [
                "--fp16-numeric-safeguards", "--fp16-selective-clamp",
                "--fp16-selective-clamp-threshold", "60000",
                "--skip-nonfinite-batches",
                "--nbs-skip-batch-at-rollback-lr-floor",
                "--nbs-rollback-min-lr", "1e-5", "--continue-on-error",
            ],
            state_file=Path("state.json"),
            output_file=Path("results.csv"),
        )

    def test_order_capacity_seeds_and_isolation(self):
        self.assertEqual(
            [item["method"] for item in EXPERIMENTS],
            ["nbs", "uniform_lora", "adalora", "eva", "shapley"],
        )
        for item in EXPERIMENTS:
            self.assertEqual(item["rank_budget"], 1536)
            self.assertEqual(item["lora_seed"], 1)
            self.assertEqual(item["data_seed"], 2)
            self.assertEqual(item["run_tag"], "fp16safe_v1")

    def test_common_safety_flags_and_selective_svd_clamp(self):
        for item in EXPERIMENTS:
            command = build_training_command(self.args, item)
            self.assertIn("--fp16-numeric-safeguards", command)
            self.assertIn("--skip-nonfinite-batches", command)
            self.assertIn("--run-tag", command)
            expects_clamp = item["method"] in ("nbs", "adalora", "shapley")
            self.assertEqual(
                "--fp16-selective-clamp" in command, expects_clamp
            )


if __name__ == "__main__":
    unittest.main()
