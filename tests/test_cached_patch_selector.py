import tempfile
import unittest
import argparse
from pathlib import Path

import torch
from torch import nn

from models.cached_patch_selector import (
    CachedPatchFeatureStore,
    CachedViewportPatchSelector,
)
from models.pipeline import Pipeline
from analysis.evaluate_cached_patch_selectors_compact import build_command


class CachedPatchSelectorTest(unittest.TestCase):
    def test_k1_maps_latest_viewport_to_one_cell(self):
        selector = CachedViewportPatchSelector(policy="k1")
        history = torch.zeros(1, 3, 3)
        # pitch=+45 degrees, yaw=-135 degrees -> row 1, col 0.
        history[0, -1, 1] = 0.5
        history[0, -1, 2] = -0.75
        self.assertEqual(selector.select_indices(history).tolist(), [4])

    def test_k2_adds_motion_direction_neighbour_across_yaw_seam(self):
        selector = CachedViewportPatchSelector(policy="k2")
        history = torch.zeros(1, 2, 3)
        history[0, 0, 2] = 170.0 / 180.0
        history[0, 1, 2] = -170.0 / 180.0
        selected = selector.select_indices(history).tolist()
        self.assertEqual(selected, [8, 9])

    def test_adaptive_uses_one_or_two_from_motion_threshold(self):
        selector = CachedViewportPatchSelector(
            policy="adaptive", motion_threshold_deg=12.0
        )
        stationary = torch.zeros(1, 2, 3)
        moving = stationary.clone()
        moving[0, 1, 2] = 20.0 / 180.0
        self.assertEqual(selector.select_indices(stationary).numel(), 1)
        self.assertEqual(selector.select_indices(moving).numel(), 2)

    def test_store_and_pipeline_pool_to_exactly_one_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = torch.arange(
                2 * 16 * 768, dtype=torch.float32
            ).reshape(2, 16, 768).to(torch.float16)
            torch.save(
                {"features": features}, root / "video4_patch_features.pt"
            )
            store = CachedPatchFeatureStore(root, device="cpu")
            chosen = store.get(4, 1, torch.tensor([5, 6]))
            self.assertTrue(torch.equal(chosen, features[0, [5, 6]]))

            pipeline = object.__new__(Pipeline)
            nn.Module.__init__(pipeline)
            pipeline.device = "cpu"
            pipeline.patch_selection_module = CachedViewportPatchSelector(
                policy="k2"
            )
            pipeline.cached_patch_feature_store = store
            pipeline.patch_selection_history = []
            pipeline.embed_multimodal = nn.Linear(768, 8)
            pipeline._resolve_frame_index = lambda _: (4, 1)
            output = pipeline._get_multimodal_information_cached_patch_selection(
                None,
                torch.zeros(1, 2, 3),
            )
            self.assertEqual(tuple(output.shape), (1, 1, 8))
            self.assertEqual(pipeline.patch_selection_history, [2])

    def test_evaluation_command_uses_cache_preload_and_one_policy(self):
        args = argparse.Namespace(
            device="cuda:0", physical_rank=32, rank_budget=512,
            latency_warmup_steps=5, motion_threshold_deg=12.0,
            cache_dir=Path("cache"),
            projector_checkpoint=Path("projector"),
        )
        command = build_command(
            args, Path("compact"), Path("results"), policy="adaptive"
        )
        self.assertIn("--cached-patch-preload", command)
        self.assertEqual(
            command[command.index("--cached-patch-policy") + 1], "adaptive"
        )
        self.assertEqual(
            command[command.index("--multimodal-mode") + 1],
            "cached-patch-selection",
        )


if __name__ == "__main__":
    unittest.main()
