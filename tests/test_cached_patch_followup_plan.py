import unittest

from analysis.evaluate_cached_patch_followups_compact import (
    QUALITY_CASES,
    SPEED_CASES,
)


class CachedPatchFollowupPlanTest(unittest.TestCase):
    def test_all_group_contains_eight_unique_followups(self):
        cases = QUALITY_CASES + SPEED_CASES
        names = [name for name, _ in cases]
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(set(names)), 8)
        self.assertTrue(all(name.startswith("q_") for name, _ in QUALITY_CASES))
        self.assertTrue(all(name.startswith("s_") for name, _ in SPEED_CASES))

    def test_speed_group_contains_real_token_or_compute_reduction(self):
        configs = dict(SPEED_CASES)
        self.assertTrue(any(
            config["policy"].startswith("gated-")
            for config in configs.values()
        ))
        self.assertTrue(any(
            config.get("projector_cache") for config in configs.values()
        ))
        self.assertTrue(any(
            config.get("refresh", 1) > 1 for config in configs.values()
        ))


if __name__ == "__main__":
    unittest.main()
