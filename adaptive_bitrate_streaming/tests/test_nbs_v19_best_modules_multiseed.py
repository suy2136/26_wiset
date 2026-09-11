import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / "analysis" / "run_nbs_v19_best_modules_multiseed.py"
SPEC = importlib.util.spec_from_file_location("best_modules_multiseed", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class BestModulesMultiseedTest(unittest.TestCase):
    def row(self, spec, seed, reward=0.8, latency=50.0):
        return {
            "experiment": spec["name"], "module": spec["module"],
            "data_seed": str(seed), "rank_budget": "1536",
            "physical_rank": "32", "mean_reward": str(reward),
            "inference_latency_mean_ms": str(latency),
            **pipeline.sweep.configured_fields(spec),
        }

    def test_only_three_isolated_specs_are_scheduled(self):
        self.assertEqual(
            [spec["name"] for spec in pipeline.RUN_SPECS],
            ["temporal_k1", "token_o0_2_3_4_7", "spec_repeat_last_k5"],
        )

    def test_reuse_and_aggregate_exact_five_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent = [self.row(spec, 1) for spec in pipeline.RUN_SPECS]
            confirmation = []
            for spec in (pipeline.TARGET_SPECS[0], pipeline.TARGET_SPECS[4]):
                confirmation.extend(self.row(spec, seed) for seed in (1, 2, 3))
            pipeline.sweep.write_csv(root / "independent.csv", independent)
            pipeline.sweep.write_csv(root / "confirmation.csv", confirmation)
            args = argparse.Namespace(
                independent_seed1=root / "independent.csv",
                confirmation_runs=root / "confirmation.csv",
                rank_budget=1536, physical_rank=32,
            )
            rows = pipeline.reusable_rows(args)
            for spec in pipeline.RUN_SPECS:
                rows.extend(self.row(spec, seed) for seed in (2, 3))
            summary = pipeline.aggregate(rows)
            self.assertEqual(len(summary), 5)
            self.assertTrue(all(row["num_seeds"] == 3 for row in summary))

    def test_wrong_offsets_are_rejected(self):
        spec = pipeline.RUN_SPECS[1]
        row = self.row(spec, 1)
        row["configured_token_offsets"] = "2,6,7"
        with self.assertRaisesRegex(ValueError, "configured_token_offsets"):
            pipeline.validate_row(row, spec, 1, 1536, 32)


if __name__ == "__main__":
    unittest.main()
