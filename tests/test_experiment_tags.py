import argparse
import unittest

from utils.experiment_tags import safe_experiment_tag


class ExperimentTagValidationTest(unittest.TestCase):
    def test_budget_scaling_tags_are_allowed(self):
        for tag in (
            "uniform_r8_data4_budget1024",
            "adalora_b512_data4_budget1024",
            "shapley_b512_data4_budget1536",
            "eva_b512_data4_budget1536",
            "nbs_v19_data4_budget1536",
        ):
            self.assertEqual(safe_experiment_tag(tag), tag)

    def test_path_like_tags_are_rejected(self):
        for tag in ("../escape", "/absolute", "has space", "semi;colon", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                safe_experiment_tag(tag)


if __name__ == "__main__":
    unittest.main()
