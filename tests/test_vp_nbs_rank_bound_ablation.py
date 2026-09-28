import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis import run_vp_nbs_rank_bound_ablation as pipeline


class RankBoundAblationTest(unittest.TestCase):
    def test_specs_are_feasible_and_configs_match(self):
        self.assertEqual(
            [(name, minimum, maximum) for name, minimum, maximum, _ in pipeline.SPECS],
            [
                ("min4_max32", 4, 32),
                ("min6_max32", 6, 32),
                ("min2_max16", 2, 16),
                ("min2_max12", 2, 12),
            ],
        )
        for _, minimum, maximum, relative in pipeline.SPECS:
            config = pipeline.REPO_ROOT / relative
            pipeline.validate_config(config, minimum, maximum)
            values = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(len(values), 2)
            self.assertLessEqual(minimum * pipeline.MODULE_COUNT, pipeline.TARGET_BUDGET)
            self.assertLessEqual(pipeline.TARGET_BUDGET, maximum * pipeline.MODULE_COUNT)

    def test_signature_pins_comparison_conditions(self):
        value = pipeline.signature()
        self.assertEqual(value["target_budget"], 512)
        self.assertEqual(value["training_data_seed"], 1)
        self.assertEqual(value["evaluation_seeds_and_data_seeds"], [1, 2, 3])
        self.assertEqual(value["evaluation_rng_mode"], "continuous")
        self.assertIn("fp16_prescaled_qk", value["attention_score_mode"])
        self.assertEqual(value["inference"], "compact_pure_nbs")
        self.assertEqual(
            [item["physical_rank"] for item in value["specs"]],
            [32, 32, 16, 12],
        )

    def test_shell_supports_data1_and_rank_config_override(self):
        shell = (pipeline.REPO_ROOT / "scripts/run_netllm_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('nbs_v19_data1', shell)
        self.assertIn('VP_NBS_RANK_CONFIG', shell)
        self.assertIn('VP_NBS_TARGET_RANK', shell)
        self.assertIn('NBS rank config does not exist', shell)

    def test_v1_state_migrates_without_losing_completed_experiments(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            previous = pipeline.signature()
            previous["pipeline"] = "vp_nbs_rank_bound_ablation_v1"
            previous["physical_rank"] = 32
            for item in previous["specs"]:
                item.pop("physical_rank")
            state = {
                "signature": previous,
                "experiments": {"min4_max32": {"status": "complete"}},
            }
            (output / "pipeline_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            args = pipeline.parse_args([
                "--output-dir", str(output), "--resume",
            ])
            _, migrated = pipeline.load_state(args)
            self.assertEqual(migrated["signature"], pipeline.signature())
            self.assertEqual(
                migrated["experiments"]["min4_max32"]["status"], "complete"
            )


if __name__ == "__main__":
    unittest.main()
