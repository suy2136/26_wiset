import unittest
from pathlib import Path

from adaptive_bitrate_streaming.analysis import run_nbs_v19_ghij_pipeline as ghij
from adaptive_bitrate_streaming.analysis import run_nbs_v19_klmn_pipeline as klmn
from adaptive_bitrate_streaming.analysis import run_nbs_v19_group_pipeline as group


class NBSV19GroupPipelineTest(unittest.TestCase):
    def args(self, module):
        return group.parse_args(
            ["--base-model-dir", "base", "--exp-pool-path", "pool.pkl"],
            state_file=group.RESULTS_ROOT / "state.json",
            output_file=group.RESULTS_ROOT / "results.csv",
        )

    def test_ghij_matrix(self):
        self.assertEqual([row["name"] for row in ghij.EXPERIMENTS], list("GHIJ"))
        self.assertEqual(
            [row["rank_budget"] for row in ghij.EXPERIMENTS],
            [1536, 1536, 2048, 2048],
        )
        self.assertEqual(
            [row["lr"] for row in ghij.EXPERIMENTS],
            [2.5e-4, 1.5e-4, 1e-4, 7.5e-5],
        )
        self.assertEqual(
            [row["physical_rank"] for row in ghij.EXPERIMENTS],
            [32, 32, 64, 64],
        )

    def test_klmn_matrix_forces_fractional_mean_rank(self):
        self.assertEqual([row["name"] for row in klmn.EXPERIMENTS], list("KLMN"))
        self.assertEqual(
            [row["rank_budget"] for row in klmn.EXPERIMENTS],
            [1792, 1792, 1824, 1824],
        )
        self.assertEqual(1824 / 64, 28.5)
        self.assertEqual(
            [row["lr"] for row in klmn.EXPERIMENTS],
            [2e-4, 1.5e-4, 2e-4, 1.5e-4],
        )

    def test_training_conditions_are_fixed(self):
        args = self.args(ghij)
        for experiment in (*ghij.EXPERIMENTS, *klmn.EXPERIMENTS):
            command = group.build_training_command(args, experiment)
            self.assertEqual(command[command.index("--seed") + 1], "1")
            self.assertEqual(command[command.index("--lr-schedule") + 1], "cosine")
            self.assertEqual(command[command.index("--warmup-steps") + 1], "500")
            self.assertEqual(command[command.index("--temporal-selector") + 1], "none")
            self.assertEqual(command[command.index("--token-selector") + 1], "none")
            self.assertEqual(command[command.index("--speculative-draft-steps") + 1], "0")
            self.assertEqual(command[command.index("--early-stopping-patience") + 1], "10")
            self.assertEqual(command[command.index("--early-stopping-min-epochs") + 1], "20")
            self.assertEqual(command[command.index("--early-stopping-min-delta") + 1], "0.003")

    def test_test_conditions_require_compaction(self):
        args = self.args(ghij)
        for experiment in (*ghij.EXPERIMENTS, *klmn.EXPERIMENTS):
            command = group.build_test_command(args, experiment, Path("checkpoint"))
            self.assertIn("--nbs-compact-inference", command)
            self.assertIn("--test", command)
            self.assertEqual(command[command.index("--temporal-selector") + 1], "none")
            self.assertEqual(command[command.index("--token-selector") + 1], "none")
            self.assertEqual(command[command.index("--speculative-draft-steps") + 1], "0")


if __name__ == "__main__":
    unittest.main()
