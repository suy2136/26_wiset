import argparse
import importlib.util
from pathlib import Path
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / "analysis" / "run_nbs_v19_best5_inference.py"
SPEC = importlib.util.spec_from_file_location("best5", SCRIPT)
best5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(best5)


def args():
    return argparse.Namespace(
        checkpoint_dir=Path("checkpoint"), base_model_dir=Path("base"),
        exp_pool_path=Path("pool.pkl"), rank_budget=1536, physical_rank=32,
        rank_config=Path("configs/nbs_v19_rank_config.json"),
        trace="fcc-test", trace_num=100, video="video1", device="cuda:0",
    )


def value(command, option):
    return command[command.index(option) + 1]


class BestFiveInferenceTest(unittest.TestCase):
    def test_matrix_contains_exactly_five_expected_conditions(self):
        self.assertEqual(
            [item["name"] for item in best5.EXPERIMENTS],
            [
                "nbs_compact_only", "temporal_k1", "token_offsets_2_6_7",
                "spec_repeat_last_k3", "combined_temporal_token_spec",
            ],
        )

    def test_commands_isolate_features_and_keep_common_conditions(self):
        for experiment in best5.EXPERIMENTS:
            command = best5.build_command(args(), experiment)
            temporal = bool(experiment.get("temporal"))
            token = bool(experiment.get("token"))
            speculative = bool(experiment.get("speculative"))
            self.assertIn("--nbs-compact-inference", command)
            self.assertEqual(value(command, "--seed"), "1")
            self.assertEqual(value(command, "--trace"), "fcc-test")
            self.assertEqual(value(command, "--trace-num"), "100")
            self.assertEqual(
                value(command, "--temporal-selector"),
                "event-aware" if temporal else "none",
            )
            self.assertEqual(
                value(command, "--token-selector"),
                "intra-timestep" if token else "none",
            )
            self.assertEqual(
                value(command, "--speculative-draft-steps"),
                "3" if speculative else "0",
            )
            self.assertEqual(
                value(command, "--speculative-drafter"),
                "repeat-last" if speculative else "mpc",
            )
            if temporal:
                self.assertEqual(value(command, "--event-max-events"), "1")
            if token:
                start = command.index("--intra-token-keep-offsets") + 1
                self.assertEqual(command[start:start + 3], ["2", "6", "7"])

    def test_baseline_comparison_math(self):
        rows = [
            {"experiment": "nbs_compact_only", "mean_reward": 0.8,
             "inference_latency_mean_ms": 80.0},
            {"experiment": "temporal_k1", "mean_reward": 0.84,
             "inference_latency_mean_ms": 60.0},
        ]
        best5.add_baseline_comparisons(rows)
        self.assertAlmostEqual(rows[1]["mean_reward_change_ratio_vs_nbs"], 0.05)
        self.assertAlmostEqual(rows[1]["inference_latency_reduction_vs_nbs"], 0.25)


if __name__ == "__main__":
    unittest.main()
