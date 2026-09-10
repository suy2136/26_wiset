import unittest
from collections import Counter

from analysis.search_kinematic_patch_selector import (
    PARAMETER_KEYS,
    PARAMETER_SPACE,
    base_candidate,
    candidate_key,
    candidates,
    interaction_candidates,
    one_factor_candidates,
)


class KinematicPatchSelectorSearchTest(unittest.TestCase):
    def test_default_search_has_48_unique_candidates(self):
        rows = candidates()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({candidate_key(row) for row in rows}), 48)

    def test_original_one_factor_sweep_is_preserved(self):
        base = base_candidate()
        rows = one_factor_candidates()
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0], base)
        for row in rows[1:]:
            changed = sum(row[key] != base[key] for key in PARAMETER_KEYS)
            self.assertEqual(changed, 1)

    def test_interactions_change_at_least_two_parameters(self):
        base = base_candidate()
        rows = interaction_candidates(32)
        self.assertEqual(len(rows), 32)
        for row in rows:
            changed = sum(row[key] != base[key] for key in PARAMETER_KEYS)
            self.assertGreaterEqual(changed, 2)

    def test_candidate_generation_is_deterministic(self):
        self.assertEqual(candidates(), candidates())

    def test_interaction_levels_are_not_extreme_value_dominated(self):
        rows = interaction_candidates(32)
        for key in PARAMETER_KEYS:
            counts = Counter(row[key] for row in rows)
            self.assertEqual(set(counts), set(PARAMETER_SPACE[key]))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)


if __name__ == '__main__':
    unittest.main()
