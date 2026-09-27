import json
from pathlib import Path
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
        self.assertEqual(value["physical_rank"], 32)
        self.assertEqual(value["training_data_seed"], 1)
        self.assertEqual(value["evaluation_seeds_and_data_seeds"], [1, 2, 3])
        self.assertEqual(value["evaluation_rng_mode"], "continuous")
        self.assertIn("fp16_prescaled_qk", value["attention_score_mode"])
        self.assertEqual(value["inference"], "compact_pure_nbs")

    def test_shell_supports_data1_and_rank_config_override(self):
        shell = (pipeline.REPO_ROOT / "scripts/run_netllm_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('nbs_v19_data1', shell)
        self.assertIn('VP_NBS_RANK_CONFIG', shell)
        self.assertIn('NBS rank config does not exist', shell)


if __name__ == "__main__":
    unittest.main()
