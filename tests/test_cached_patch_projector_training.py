import ast
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:  # local lightweight environments may not bundle PyTorch
    torch = None


ROOT = Path(__file__).resolve().parents[1]


if torch is not None:
    class DummyPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_multimodal = torch.nn.Linear(3, 2).half()
            self.other = torch.nn.Linear(2, 1)


class CachedPatchProjectorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            return
        source = (ROOT / 'run_plm.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == 'freeze_pipeline_for_projector_training')
        module = ast.Module(body=[node], type_ignores=[])
        namespace = {'torch': torch}
        exec(compile(module, 'run_plm.py', 'exec'), namespace)
        cls.freeze = staticmethod(
            namespace['freeze_pipeline_for_projector_training']
        )

    def test_only_projector_is_trainable_and_fp32(self):
        if torch is None:
            self.skipTest('PyTorch is not installed in the local test runtime')
        model = DummyPipeline()
        names = self.freeze(model)
        self.assertEqual(set(names), {
            'embed_multimodal.weight', 'embed_multimodal.bias'
        })
        self.assertTrue(all(parameter.dtype == torch.float32
                            for parameter in model.embed_multimodal.parameters()))
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in model.other.parameters()))

    def test_runner_accepts_explicit_epochs_and_disables_projector_cache(self):
        source = (ROOT / 'analysis' /
                  'train_validate_cached_patch_projector.py').read_text(
                      encoding='utf-8')
        self.assertIn("'--epochs', str(args.epochs)", source)
        self.assertIn("result.add_argument('--epochs'", source)
        self.assertIn("'--train-multimodal-projector-only'", source)
        self.assertNotIn("'--cached-patch-projector-cache'", source)
        self.assertIn("'--cached-patch-policy', args.policy", source)
        self.assertIn("Use k1 to update the projector on every sample", source)
        self.assertIn("'--multimodal-projector-validation-policy'", source)
        self.assertIn("default='cpu'", source)
        self.assertIn('TRAIN_VIDEOS + VALID_VIDEOS', source)

    def test_training_tracks_best_validation_epoch(self):
        source = (ROOT / 'run_plm.py').read_text(encoding='utf-8')
        self.assertIn("for epoch in range(1, args.epochs + 1)", source)
        self.assertIn("best_valid_mae = float('inf')", source)
        self.assertIn("'is_best_valid_mae': int(is_best)", source)
        self.assertIn("torch.save(projector_state, best_path)", source)

    def test_saved_projector_format_matches_loader_aliases(self):
        if torch is None:
            self.skipTest('PyTorch is not installed in the local test runtime')
        model = DummyPipeline().float()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'projector.pth'
            torch.save({key: value.detach().cpu()
                        for key, value in model.embed_multimodal.state_dict().items()},
                       path)
            state = torch.load(path, map_location='cpu')
        self.assertEqual(set(state), {'weight', 'bias'})


if __name__ == '__main__':
    unittest.main()
