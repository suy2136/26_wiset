import unittest

import torch

from models.kinematic_patch_selector import KinematicPatchSelector


class KinematicPatchSelectorTest(unittest.TestCase):
    def test_is_parameter_free_and_selects_exact_budget(self):
        selector = KinematicPatchSelector(grid_rows=4, grid_cols=4)
        self.assertEqual(sum(parameter.numel() for parameter in selector.parameters()), 0)
        history = torch.zeros(2, 10, 3)
        logits = selector(history)
        mask = selector.select_patches(logits, top_k=6)
        self.assertEqual(tuple(logits.shape), (2, 16))
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.tensor([6, 6])))

    def test_wrapped_yaw_velocity_crosses_seam_without_jump(self):
        selector = KinematicPatchSelector(
            velocity_window=2, horizon_scale=0.5,
            acceleration_weight=0.0, uncertainty_deg=0.0,
        )
        # Degrees 160 -> 175 -> -170 is a smooth positive yaw motion.
        history = torch.zeros(1, 3, 3)
        history[0, :, 2] = torch.tensor([160.0, 175.0, -170.0]) / 180.0
        differences = selector._wrapped_difference_degrees(
            history[..., 2] * 180.0
        )
        self.assertTrue(torch.allclose(differences, torch.tensor([[15.0, 15.0]])))
        self.assertTrue(torch.isfinite(selector(history)).all())

    def test_stationary_center_prefers_center_adjacent_cells(self):
        selector = KinematicPatchSelector(
            grid_rows=4, grid_cols=4, uncertainty_deg=0.0,
            acceleration_weight=0.0,
        )
        logits = selector(torch.zeros(1, 10, 3))
        selected = selector.select_patches(logits, top_k=4)[0].nonzero().flatten()
        center_adjacent = {5, 6, 9, 10}
        self.assertEqual(set(selected.tolist()), center_adjacent)


if __name__ == '__main__':
    unittest.main()
