import os
import tempfile
import unittest

import torch
from torch import nn
from types import SimpleNamespace

import run_plm
from models.patch_selection import PatchSelectionModule
from models.pipeline import Pipeline
from utils.losses import CircularViewportMSELoss


class _TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_vp = nn.Linear(2, 3)
        self.embed_multimodal = nn.Linear(4, 3)
        self.embed_ln = nn.LayerNorm(3)
        self.patch_selection_module = PatchSelectionModule(
            grid_rows=2,
            grid_cols=2,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=16,
            max_history_len=8,
        )


class _LogitSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.linspace(-1.0, 1.0, 4))

    def forward(self, history):
        return self.logits.unsqueeze(0).expand(history.shape[0], -1)


class _TinyPLM(nn.Module):
    def forward(self, inputs_embeds, attention_mask=None, teacher_forcing=False):
        pooled = inputs_embeds.mean(dim=1, keepdim=True)[..., :3]
        return SimpleNamespace(logits=pooled.expand(-1, 2, -1))


class _TaskLossPipeline(Pipeline):
    def __init__(self):
        nn.Module.__init__(self)
        self.multimodal_mode = 'patch-selection'
        self.patch_top_k = 2
        self.patch_selection_module = _LogitSelector()
        self.embed_multimodal = nn.Linear(768, 4)
        self.conv1d = nn.Sequential(nn.Conv1d(1, 256, 3), nn.Flatten())
        self.embed_vp = nn.Linear(256, 4)
        self.embed_ln = nn.LayerNorm(4)
        self.plm = _TinyPLM()
        self.loss_fct = CircularViewportMSELoss()
        self.device = 'cpu'

    def _resolve_frame_index(self, video_user_position):
        return 1, 1

    def _load_frame_patches(self, video_index, image_index, indices=None):
        return torch.ones(len(indices), 3, 4, 4)

    def _vit_feature_fn(self, image_batch):
        return torch.ones(image_batch.shape[0], 768)

    def _plm_compute_dtype(self):
        return torch.float32


class FrozenPatchSelectorTrainingTest(unittest.TestCase):
    def test_projector_loader_is_selective(self):
        pipeline = _TinyPipeline()
        original_embed_vp = {
            key: value.detach().clone()
            for key, value in pipeline.embed_vp.state_dict().items()
        }
        source_weight = torch.randn_like(pipeline.embed_multimodal.weight)
        source_bias = torch.randn_like(pipeline.embed_multimodal.bias)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'modules_except_plm.bin')
            torch.save({
                '0.weight': torch.randn_like(pipeline.embed_vp.weight),
                '0.bias': torch.randn_like(pipeline.embed_vp.bias),
                '1.weight': source_weight,
                '1.bias': source_bias,
            }, path)
            report = run_plm.load_multimodal_projector(pipeline, path)

        self.assertTrue(torch.equal(pipeline.embed_multimodal.weight, source_weight))
        self.assertTrue(torch.equal(pipeline.embed_multimodal.bias, source_bias))
        for key, value in pipeline.embed_vp.state_dict().items():
            self.assertTrue(torch.equal(value, original_embed_vp[key]))
        self.assertEqual(report['source_keys']['weight'], '1.weight')
        self.assertEqual(len(report['source_sha256']), 64)

    def test_freeze_leaves_only_selector_trainable(self):
        pipeline = _TinyPipeline()
        names = run_plm.freeze_pipeline_for_selector_training(pipeline)
        self.assertTrue(names)
        self.assertTrue(
            all(name.startswith('patch_selection_module.') for name in names)
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in pipeline.embed_vp.parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad
                for parameter in pipeline.embed_multimodal.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad
                for parameter in pipeline.patch_selection_module.parameters())
        )

    def test_projector_shape_mismatch_fails_closed(self):
        pipeline = _TinyPipeline()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'modules_except_plm.bin')
            torch.save({
                '1.weight': torch.randn(9, 9),
                '1.bias': torch.randn(9),
            }, path)
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                run_plm.load_multimodal_projector(pipeline, path)

    def test_nbs_prediction_loss_reaches_selector_only(self):
        pipeline = _TaskLossPipeline()
        run_plm.freeze_pipeline_for_selector_training(pipeline)
        history = torch.zeros(1, 2, 3)
        future = torch.zeros(1, 2, 3)
        loss, logits, indices = pipeline.differentiable_patch_selection_loss(
            history, future, (torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
        )
        loss.backward()
        self.assertEqual(len(indices), 2)
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertIsNotNone(pipeline.patch_selection_module.logits.grad)
        self.assertGreater(
            float(pipeline.patch_selection_module.logits.grad.abs().sum()), 0.0
        )
        self.assertFalse(pipeline.embed_multimodal.weight.requires_grad)

    def test_rotation_aware_mae_wraps_yaw(self):
        prediction = torch.tensor([[[0.0, 0.0, 179.0 / 180.0]]])
        target = torch.tensor([[[0.0, 0.0, -179.0 / 180.0]]])
        mae = run_plm._rotation_aware_mae_degrees(prediction, target)
        self.assertAlmostEqual(float(mae), 2.0 / 3.0, places=4)


if __name__ == '__main__':
    unittest.main()
