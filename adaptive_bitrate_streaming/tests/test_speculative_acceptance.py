import os
import sys
import unittest

import numpy as np


ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ABR_ROOT not in sys.path:
    sys.path.insert(0, ABR_ROOT)

from plm_special.speculative.acceptance import (
    buffer_deviation_seconds,
    build_acceptance_plan,
)


class AcceptancePlanTest(unittest.TestCase):
    def test_accepts_entire_matching_draft(self):
        plan = build_acceptance_plan([1, 2, 3], [1, 2, 3])
        self.assertEqual(plan.actions, (1, 2, 3))
        self.assertEqual(plan.accepted_count, 3)
        self.assertTrue(plan.fully_accepted)

    def test_stops_at_first_mismatch_and_uses_target_correction(self):
        plan = build_acceptance_plan([1, 2, 3, 4], [1, 2, 0, 4])
        self.assertEqual(plan.actions, (1, 2, 0))
        self.assertEqual(plan.accepted_count, 2)
        self.assertEqual(plan.mismatch_index, 2)

    def test_first_mismatch_produces_one_step_target_fallback(self):
        plan = build_acceptance_plan([1, 2], [4, 2])
        self.assertEqual(plan.actions, (4,))
        self.assertEqual(plan.accepted_count, 0)

    def test_buffer_deviation_uses_seconds(self):
        observed = np.zeros((1, 1, 6, 6), dtype=np.float32)
        predicted = np.zeros((6, 6), dtype=np.float32)
        observed[..., 1, -1] = 1.2
        predicted[1, -1] = 1.0
        self.assertAlmostEqual(
            buffer_deviation_seconds(observed, predicted), 2.0, places=5
        )


if __name__ == '__main__':
    unittest.main()
