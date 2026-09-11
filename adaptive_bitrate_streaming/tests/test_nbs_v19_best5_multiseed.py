import importlib.util
import argparse
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / "analysis" / "run_nbs_v19_best5_multiseed.py"
SPEC = importlib.util.spec_from_file_location("best5_multiseed", SCRIPT)
multiseed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multiseed)


class BestFiveMultiseedTest(unittest.TestCase):
    def test_aggregate_computes_mean_sample_std_and_baseline_changes(self):
        rows = []
        for experiment in multiseed.best5.EXPERIMENTS:
            for seed, reward, latency in (
                (1, 0.8, 80.0), (2, 0.9, 60.0), (3, 1.0, 70.0)
            ):
                if experiment["name"] != "nbs_compact_only":
                    reward += 0.1
                    latency -= 10.0
                rows.append({
                    "experiment": experiment["name"], "data_seed": seed,
                    "mean_reward": reward,
                    "inference_latency_mean_ms": latency,
                })
        summary = multiseed.aggregate_rows(rows, [1, 2, 3])
        baseline, temporal = summary[:2]
        self.assertAlmostEqual(baseline["mean_reward_mean"], 0.9)
        self.assertAlmostEqual(baseline["mean_reward_std"], 0.1)
        self.assertAlmostEqual(temporal["mean_reward_delta_vs_nbs"], 0.1)
        self.assertAlmostEqual(
            temporal["inference_latency_reduction_vs_nbs"], 1.0 / 7.0
        )

    def test_read_rows_rejects_incomplete_seed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.csv"
            path.write_text("experiment,mean_reward\nnbs_compact_only,0.8\n")
            with self.assertRaises(ValueError):
                multiseed.read_rows(path, 1)

    def test_reuse_rejects_a_different_checkpoint(self):
        rows = [{
            "experiment": item["name"], "data_seed": 1,
            "checkpoint_dir": str(Path("old-checkpoint").resolve()),
            "rank_budget": "1536", "physical_rank": "32",
        } for item in multiseed.best5.EXPERIMENTS]
        args = argparse.Namespace(
            checkpoint_dir=Path("c-checkpoint"), rank_budget=1536,
            physical_rank=32,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            multiseed.validate_reusable_rows(rows, args, 1)

    def test_reuse_accepts_matching_checkpoint(self):
        checkpoint = Path("c-checkpoint").resolve()
        rows = [{
            "experiment": item["name"], "data_seed": 1,
            "checkpoint_dir": str(checkpoint),
            "rank_budget": "1536", "physical_rank": "32",
        } for item in multiseed.best5.EXPERIMENTS]
        args = argparse.Namespace(
            checkpoint_dir=checkpoint, rank_budget=1536, physical_rank=32,
        )
        self.assertIs(
            multiseed.validate_reusable_rows(rows, args, 1), rows
        )


if __name__ == "__main__":
    unittest.main()
