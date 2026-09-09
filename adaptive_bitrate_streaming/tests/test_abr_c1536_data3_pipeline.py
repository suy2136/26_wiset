import unittest
from pathlib import Path

from adaptive_bitrate_streaming.analysis import (
    run_abr_c1536_data3_safeguarded_allocators as all_data3,
)
from adaptive_bitrate_streaming.analysis import run_abr_c1536_data3_server1 as server1
from adaptive_bitrate_streaming.analysis import run_abr_c1536_data3_server2 as server2
from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
    build_eva_precompute_command,
    build_test_command,
    build_training_command,
    parse_args,
)


class ABRC1536Data3PipelineTests(unittest.TestCase):
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

    def test_server_assignment_covers_five_algorithms_once(self):
        combined = (*server1.SERVER_EXPERIMENTS, *server2.SERVER_EXPERIMENTS)
        self.assertEqual(combined, all_data3.EXPERIMENTS)
        self.assertEqual(
            [item["method"] for item in server1.SERVER_EXPERIMENTS],
            ["nbs", "uniform_lora", "adalora"],
        )
        self.assertEqual(
            [item["method"] for item in server2.SERVER_EXPERIMENTS],
            ["eva", "shapley"],
        )

    def test_capacity_seed_and_training_schedule(self):
        for item in all_data3.EXPERIMENTS:
            self.assertEqual(item["rank_budget"], 1536)
            self.assertEqual(item["seed"], 1)
            self.assertEqual(item["lora_seed"], 1)
            self.assertEqual(item["data_seed"], 3)
            self.assertEqual(item["run_tag"], "fp16safe_data3_v1")
            command = build_training_command(self.args, item)
            self.assertEqual(command[command.index("--lr") + 1], "0.0002")
            self.assertEqual(command[command.index("--lr-schedule") + 1], "cosine")
            self.assertEqual(command[command.index("--warmup-steps") + 1], "500")
            self.assertEqual(command[command.index("--seed") + 1], "1")
            self.assertEqual(command[command.index("--lora-seed") + 1], "1")
            self.assertEqual(command[command.index("--data-seed") + 1], "3")

    def test_safeguards_eva_and_compact_nbs(self):
        for item in all_data3.EXPERIMENTS:
            train = build_training_command(self.args, item)
            self.assertIn("--fp16-numeric-safeguards", train)
            self.assertIn("--skip-nonfinite-batches", train)
            expects_clamp = item["method"] in ("nbs", "adalora", "shapley")
            self.assertEqual("--fp16-selective-clamp" in train, expects_clamp)

        nbs_test = build_test_command(
            self.args, all_data3.EXPERIMENTS[0], Path("checkpoint")
        )
        self.assertIn("--nbs-compact-inference", nbs_test)

        nbs_train = build_training_command(
            self.args, all_data3.EXPERIMENTS[0]
        )
        self.assertIn("--nbs-allocation-audit", nbs_train)
        for item in all_data3.EXPERIMENTS[1:]:
            self.assertNotIn(
                "--nbs-allocation-audit",
                build_training_command(self.args, item),
            )

        eva_command = build_eva_precompute_command(
            self.args, all_data3.EXPERIMENTS[3]
        )
        self.assertIn("--allow-unconverged", eva_command)
        self.assertEqual(
            eva_command[eva_command.index("--max-batches") + 1], "512"
        )


if __name__ == "__main__":
    unittest.main()
