import os
import sys
import unittest


ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ABR_ROOT not in sys.path:
    sys.path.insert(0, ABR_ROOT)

from plm_special.models.selection_layout import recent_timestep_window


class RecentTimestepWindowTest(unittest.TestCase):
    def test_steady_state_history_reduction(self):
        # 20 completed history steps plus the protected current 7-token block.
        start, available, selected = recent_timestep_window(167, 10)
        self.assertEqual((start, available, selected), (80, 20, 10))
        self.assertEqual(167 - start, 87)

    def test_current_block_is_kept_with_no_history(self):
        start, available, selected = recent_timestep_window(7, 10)
        self.assertEqual((start, available, selected), (0, 0, 0))

    def test_early_episode_keeps_all_available_history(self):
        start, available, selected = recent_timestep_window(31, 10)
        self.assertEqual((start, available, selected), (0, 3, 3))

    def test_five_steps_produce_expected_length(self):
        start, available, selected = recent_timestep_window(167, 5)
        self.assertEqual((available, selected), (20, 5))
        self.assertEqual(167 - start, 47)

    def test_twenty_steps_is_layout_equivalent_to_no_selector(self):
        start, available, selected = recent_timestep_window(167, 20)
        self.assertEqual((start, available, selected), (0, 20, 20))

    def test_rejects_partial_history_block(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            recent_timestep_window(166, 10)

    def test_rejects_sequence_shorter_than_current_block(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            recent_timestep_window(6, 10)

    def test_speculative_suffix_is_protected(self):
        # 20 real-history blocks + current block + two 8-token MPC drafts.
        start, available, selected = recent_timestep_window(
            183, 10, current_step_tokens=23
        )
        self.assertEqual((start, available, selected), (80, 20, 10))
        self.assertEqual(183 - start, 103)


if __name__ == "__main__":
    unittest.main()
