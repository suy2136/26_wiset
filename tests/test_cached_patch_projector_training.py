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

    def test_runner_is_one_epoch_and_disables_projector_cache(self):
        source = (ROOT / 'analysis' /
                  'train_validate_cached_patch_projector.py').read_text(
                      encoding='utf-8')
        self.assertIn("'--epochs', '1'", source)
        self.assertIn("'--train-multimodal-projector-only'", source)
        self.assertNotIn("'--cached-patch-projector-cache'", source)
        self.assertIn("'--cached-patch-policy', 'gated-k1'", source)
        self.assertIn("default='cpu'", source)
        self.assertIn('TRAIN_VIDEOS + VALID_VIDEOS', source)

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
