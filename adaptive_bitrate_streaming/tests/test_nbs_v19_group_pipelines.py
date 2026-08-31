import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_bitrate_streaming.analysis import run_nbs_v19_ghij_pipeline as ghij
from adaptive_bitrate_streaming.analysis import run_nbs_v19_klmn_pipeline as klmn
from adaptive_bitrate_streaming.analysis import run_nbs_v19_cpqr_pipeline as cpqr
from adaptive_bitrate_streaming.analysis import run_nbs_v19_group_pipeline as group
from adaptive_bitrate_streaming.analysis import run_abr_server1_cpq_pipeline as server1
from adaptive_bitrate_streaming.analysis import run_abr_server2_r_seed_uniform_pipeline as server2


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

    def test_cpqr_matrix(self):
        self.assertEqual(
            [row["name"] for row in cpqr.EXPERIMENTS],
            ["C_REPRO", "P", "Q", "R"],
        )
        self.assertEqual(
            [row["rank_budget"] for row in cpqr.EXPERIMENTS],
            [1536, 1536, 1664, 1664],
        )
        self.assertEqual(
            [row["lr"] for row in cpqr.EXPERIMENTS],
            [2e-4, 1.75e-4, 2e-4, 1.75e-4],
        )
        self.assertEqual(
            [row["physical_rank"] for row in cpqr.EXPERIMENTS],
            [32, 32, 32, 32],
        )

    def test_two_server_assignment(self):
        self.assertEqual(
            [row["name"] for row in server1.EXPERIMENTS],
            ["C_REPRO", "P", "Q"],
        )
        self.assertEqual(
            [row["name"] for row in server2.EXPERIMENTS],
            ["R", "C_SEED2", "UNIFORM_R24"],
        )
        self.assertEqual(server2.EXPERIMENTS[1]["seed"], 2)
        self.assertEqual(
            server2.EXPERIMENTS[2]["method"], "uniform_lora"
        )
        self.assertEqual(server2.EXPERIMENTS[2]["rank_budget"], 1536)
        self.assertEqual(server2.EXPERIMENTS[2]["physical_rank"], 24)

    def test_uniform_training_and_test_commands(self):
        args = self.args(server2)
        uniform = server2.EXPERIMENTS[2]
        train = group.build_training_command(args, uniform)
        test = group.build_test_command(args, uniform, Path("checkpoint"))
        self.assertNotIn("--nbs-v19", train)
        self.assertNotIn("--nbs-rank-budget", train)
        self.assertIn("--checkpoint-retention", train)
        self.assertEqual(
            train[train.index("--checkpoint-retention") + 1], "best-latest"
        )
        self.assertEqual(train[train.index("--rank") + 1], "24")
        self.assertNotIn("--nbs-v19", test)
        self.assertNotIn("--nbs-compact-inference", test)

    def test_seed_two_replication_commands(self):
        args = self.args(server2)
        experiment = server2.EXPERIMENTS[1]
        for command in (
            group.build_training_command(args, experiment),
            group.build_test_command(args, experiment, Path("checkpoint")),
        ):
            self.assertEqual(command[command.index("--seed") + 1], "2")

    def test_training_conditions_are_fixed(self):
        args = self.args(ghij)
        for experiment in (
            *ghij.EXPERIMENTS, *klmn.EXPERIMENTS, *cpqr.EXPERIMENTS,
        ):
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
            expected_safety = {
                "--nbs-rollback-backup-device": "cpu",
                "--nbs-max-rollback-backup-mib": "2048.0",
                "--nbs-update-ratio-warning": "0.01",
                "--nbs-max-update-ratio": "0.05",
                "--nbs-update-ratio-floor": "0.01",
                "--nbs-max-update-rms": "0.01",
                "--nbs-rollback-lr-factor": "0.5",
                "--nbs-max-consecutive-rollbacks": "3",
            }
            for option, value in expected_safety.items():
                self.assertEqual(command[command.index(option) + 1], value)

    def test_test_conditions_require_compaction(self):
        args = self.args(ghij)
        for experiment in (
            *ghij.EXPERIMENTS, *klmn.EXPERIMENTS, *cpqr.EXPERIMENTS,
        ):
            command = group.build_test_command(args, experiment, Path("checkpoint"))
            self.assertIn("--nbs-compact-inference", command)
            self.assertIn("--test", command)
            self.assertEqual(command[command.index("--temporal-selector") + 1], "none")
            self.assertEqual(command[command.index("--token-selector") + 1], "none")
            self.assertEqual(command[command.index("--speculative-draft-steps") + 1], "0")

    def test_old_resume_state_accepts_new_numeric_safety_defaults(self):
        args = self.args(ghij)
        run_signature = group.signature(args, ghij.EXPERIMENTS)
        old_signature = dict(run_signature)
        old_signature.pop("numeric_safety")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({
                "signature": old_signature,
                "runs": {"G": {"status": "complete"}},
            }), encoding="utf-8")
            state = group.load_state(path, True, run_signature)
        self.assertEqual(state["signature"], run_signature)
        self.assertEqual(state["runs"]["G"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
