import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reseed = load_module(
    "test_per_episode_reseed",
    ROOT / "plm_special" / "test_per_episode_reseed.py",
)
pipeline = load_module(
    "run_nbs_v19_reseed_pure_full_multiseed",
    ROOT / "analysis" / "run_nbs_v19_reseed_pure_full_multiseed.py",
)


class DummyModel:
    def __init__(self):
        self.clears = 0

    def clear_dq(self):
        self.clears += 1
        return self.clears


class EpisodeReseedTest(unittest.TestCase):
    def test_reseed_sequence_and_restore(self):
        model = DummyModel()
        original = model.clear_dq
        seeds = []
        with reseed.episode_reseed_context(model, 10, seeds.append) as state:
            self.assertEqual(model.clear_dq(), 1)
            self.assertEqual(model.clear_dq(), 2)
        self.assertEqual(seeds, [11, 12])
        self.assertEqual(state["completed_episodes"], 2)
        self.assertNotIn("clear_dq", model.__dict__)
        self.assertEqual(model.clear_dq(), 3)
        self.assertEqual(model.clear_dq.__func__, original.__func__)

    def test_aggregate_reports_paired_qoe_and_rebuffer_deltas(self):
        rows = []
        for seed, base_qoe, full_qoe in ((1, 0.8, 0.9), (2, 0.7, 0.75), (3, 0.9, 0.88)):
            common = {
                "data_seed": seed,
                "inference_latency_mean_ms": 50.0,
                "total_rebuffer_s": 10.0,
            }
            rows.append({
                **common, "experiment": "nbs_compact_only",
                "mean_reward": base_qoe,
            })
            rows.append({
                **common, "experiment": "combined_temporal_token_spec",
                "mean_reward": full_qoe,
                "inference_latency_mean_ms": 25.0,
                "total_rebuffer_s": 8.0,
            })
        summaries = pipeline.aggregate(rows, [1, 2, 3])
        full = summaries[1]
        self.assertAlmostEqual(full["paired_qoe_delta_mean"], (0.1 + 0.05 - 0.02) / 3)
        self.assertAlmostEqual(full["paired_total_rebuffer_delta_s_mean"], -2.0)
        self.assertAlmostEqual(full["inference_latency_reduction_vs_nbs"], 0.5)

    def test_dry_run_uses_only_pure_and_full_with_reseed(self):
        args = pipeline.parse_args([
            "--checkpoint-dir", "checkpoint",
            "--base-model-dir", "base",
            "--exp-pool-path", "pool.pkl",
            "--dry-run",
        ])
        command = pipeline.seed_command(args, 2, Path("seed2.csv"))
        self.assertIn("per-episode", command)
        only = command[command.index("--only") + 1:]
        self.assertEqual(tuple(only), pipeline.EXPERIMENTS)


if __name__ == "__main__":
    unittest.main()
