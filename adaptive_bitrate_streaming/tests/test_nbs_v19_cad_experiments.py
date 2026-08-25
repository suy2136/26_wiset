import unittest

from adaptive_bitrate_streaming.analysis import run_nbs_v19_cad_experiments as cad


class NBSV19CADExperimentsTest(unittest.TestCase):
    def test_experiment_order_and_settings(self):
        self.assertEqual([item["name"] for item in cad.EXPERIMENTS], ["C", "A", "D"])
        self.assertEqual(
            [item["rank_budget"] for item in cad.EXPERIMENTS],
            [1536, 2048, 3072],
        )
        self.assertEqual(
            [item["physical_rank"] for item in cad.EXPERIMENTS],
            [32, 32, 64],
        )
        self.assertEqual(
            [item["lr_schedule"] for item in cad.EXPERIMENTS],
            ["cosine", "constant", "cosine"],
        )

    def test_every_command_enables_the_same_early_stopping(self):
        args = cad.parse_args([
            "--base-model-dir", "base", "--exp-pool-path", "pool.pkl"
        ])
        expected = {
            "--early-stopping-patience": "10",
            "--early-stopping-min-epochs": "20",
            "--early-stopping-min-delta": "0.003",
            "--plateau-lr-patience": "5",
            "--plateau-lr-factor": "0.5",
            "--plateau-min-lr": "1e-06",
        }
        for experiment in cad.EXPERIMENTS:
            command = cad.build_training_command(args, experiment)
            for option, value in expected.items():
                self.assertEqual(command[command.index(option) + 1], value)
            self.assertEqual(
                command[command.index("--checkpoint-retention") + 1],
                "best-latest",
            )

    def test_command_uses_each_experiment_learning_rate_policy(self):
        args = cad.parse_args([
            "--base-model-dir", "base", "--exp-pool-path", "pool.pkl"
        ])
        commands = [
            cad.build_training_command(args, experiment)
            for experiment in cad.EXPERIMENTS
        ]
        self.assertEqual(
            [command[command.index("--lr") + 1] for command in commands],
            ["0.0002", "0.0001", "0.0002"],
        )
        self.assertEqual(
            [command[command.index("--lr-schedule") + 1] for command in commands],
            ["cosine", "constant", "cosine"],
        )
        self.assertEqual(
            [command[command.index("--warmup-steps") + 1] for command in commands],
            ["500", "2000", "500"],
        )


if __name__ == "__main__":
    unittest.main()
