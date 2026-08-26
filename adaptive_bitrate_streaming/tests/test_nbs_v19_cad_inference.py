import json
from pathlib import Path
import tempfile
import unittest

from adaptive_bitrate_streaming.analysis import run_nbs_v19_cad_inference as cad


class NBSV19CADInferenceTest(unittest.TestCase):
    def make_args(self):
        return cad.parse_args([
            "--base-model-dir", "base", "--exp-pool-path", "pool.pkl"
        ])

    def test_order_and_capacity_settings(self):
        self.assertEqual([item["name"] for item in cad.EXPERIMENTS], ["C", "A", "D"])
        self.assertEqual(
            [item["rank_budget"] for item in cad.EXPERIMENTS], [1536, 2048, 3072]
        )
        self.assertEqual(
            [item["physical_rank"] for item in cad.EXPERIMENTS], [32, 32, 64]
        )
        self.assertEqual(
            [item["mean_active_rank"] for item in cad.EXPERIMENTS],
            [24.0, 32.0, 48.0],
        )

    def test_commands_use_matching_rank_configs_and_same_test_conditions(self):
        args = self.make_args()
        commands = [
            cad.build_test_command(args, item, Path(item["name"]))
            for item in cad.EXPERIMENTS
        ]
        for command, experiment in zip(commands, cad.EXPERIMENTS):
            self.assertEqual(
                command[command.index("--rank") + 1],
                str(experiment["physical_rank"]),
            )
            self.assertEqual(
                command[command.index("--nbs-rank-config") + 1],
                experiment["rank_config"],
            )
            self.assertEqual(command[command.index("--trace") + 1], "fcc-test")
            self.assertEqual(command[command.index("--trace-num") + 1], "100")
            self.assertEqual(command[command.index("--token-selector") + 1], "none")
            self.assertEqual(
                command[command.index("--speculative-draft-steps") + 1], "0"
            )

    def test_state_requires_all_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({
                "order": ["C", "A", "D"],
                "runs": {
                    "C": {"checkpoint_dir": "/checkpoints/C"},
                    "A": {"checkpoint_dir": "/checkpoints/A"},
                    "D": {"checkpoint_dir": "/checkpoints/D"},
                },
            }), encoding="utf-8")
            self.assertEqual(cad.checkpoints_from_state(state), {
                "C": Path("/checkpoints/C"),
                "A": Path("/checkpoints/A"),
                "D": Path("/checkpoints/D"),
            })

    def test_comparisons_use_c_as_same_run_baseline(self):
        rows = cad.add_comparisons([
            {"experiment": "C", "mean_reward": 0.8,
             "inference_latency_mean_ms": 50.0},
            {"experiment": "D", "mean_reward": 0.84,
             "inference_latency_mean_ms": 45.0},
        ])
        self.assertAlmostEqual(rows[1]["mean_reward_percent_vs_c"], 0.05)
        self.assertAlmostEqual(
            rows[1]["inference_latency_reduction_vs_c"], 0.10
        )


if __name__ == "__main__":
    unittest.main()
