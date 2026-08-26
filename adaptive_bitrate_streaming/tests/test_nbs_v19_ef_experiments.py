import unittest

from adaptive_bitrate_streaming.analysis import run_nbs_v19_ef_experiments as ef


class NBSV19EFExperimentsTest(unittest.TestCase):
    def test_experiment_order_and_settings(self):
        self.assertEqual([item["name"] for item in ef.EXPERIMENTS], ["E", "F"])
        self.assertEqual(
            [item["rank_budget"] for item in ef.EXPERIMENTS], [2048, 3072]
        )
        self.assertEqual(
            [item["physical_rank"] for item in ef.EXPERIMENTS], [64, 64]
        )
        self.assertEqual(
            [item["lr"] for item in ef.EXPERIMENTS], [2e-4, 1e-4]
        )
        self.assertTrue(all(
            item["lr_schedule"] == "cosine" for item in ef.EXPERIMENTS
        ))

    def test_commands_share_early_stopping_and_cosine_warmup(self):
        args = ef.parse_args([
            "--base-model-dir", "base", "--exp-pool-path", "pool.pkl"
        ], default_state=ef.DEFAULT_STATE)
        for experiment in ef.EXPERIMENTS:
            command = ef.build_training_command(args, experiment)
            self.assertEqual(
                command[command.index("--early-stopping-patience") + 1], "10"
            )
            self.assertEqual(
                command[command.index("--early-stopping-min-epochs") + 1], "20"
            )
            self.assertEqual(
                command[command.index("--warmup-steps") + 1], "500"
            )
            self.assertEqual(
                command[command.index("--checkpoint-retention") + 1],
                "best-latest",
            )


if __name__ == "__main__":
    unittest.main()
