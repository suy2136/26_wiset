import json
from pathlib import Path
import tempfile
import unittest

from adaptive_bitrate_streaming.analysis import run_nbs_v19_ef_inference as ef


class NBSV19EFInferenceTest(unittest.TestCase):
    def make_args(self):
        return ef.parse_args([
            "--base-model-dir", "base", "--exp-pool-path", "pool.pkl",
        ])

    def test_only_e_and_f_are_defined(self):
        self.assertEqual([item["name"] for item in ef.EXPERIMENTS], ["E", "F"])
        self.assertEqual(
            [item["rank_budget"] for item in ef.EXPERIMENTS], [2048, 3072]
        )
        self.assertEqual(
            [item["mean_active_rank"] for item in ef.EXPERIMENTS], [32.0, 48.0]
        )

    def test_commands_force_identical_inference_conditions(self):
        args = self.make_args()
        commands = [
            ef.build_test_command(args, item, Path(item["name"]))
            for item in ef.EXPERIMENTS
        ]
        for command in commands:
            self.assertEqual(command[command.index("--rank") + 1], "64")
            self.assertEqual(command[command.index("--trace") + 1], "fcc-test")
            self.assertEqual(command[command.index("--trace-num") + 1], "100")
            self.assertEqual(command[command.index("--token-selector") + 1], "none")
            self.assertEqual(
                command[command.index("--speculative-draft-steps") + 1], "0"
            )
            self.assertIn("--fixed-order", command)
            self.assertIn(ef.RANK_CONFIG, command)

    def test_state_resolves_both_completed_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({
                "order": ["E", "F"],
                "runs": {
                    "E": {"checkpoint_dir": "/checkpoints/E"},
                    "F": {"checkpoint_dir": "/checkpoints/F"},
                },
            }), encoding="utf-8")
            checkpoints = ef.checkpoints_from_state(state)
            self.assertEqual(checkpoints, {
                "E": Path("/checkpoints/E"), "F": Path("/checkpoints/F")
            })

    def test_comparisons_use_e_as_same_run_baseline(self):
        rows = ef.add_comparisons([
            {"experiment": "E", "mean_reward": 0.8,
             "inference_latency_mean_ms": 50.0},
            {"experiment": "F", "mean_reward": 0.84,
             "inference_latency_mean_ms": 45.0},
        ])
        self.assertAlmostEqual(rows[1]["mean_reward_percent_vs_e"], 0.05)
        self.assertAlmostEqual(
            rows[1]["inference_latency_reduction_vs_e"], 0.10
        )


if __name__ == "__main__":
    unittest.main()
