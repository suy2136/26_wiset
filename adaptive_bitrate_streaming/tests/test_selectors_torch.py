import os
import sys
import unittest


ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ABR_ROOT not in sys.path:
    sys.path.insert(0, ABR_ROOT)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class SelectorTensorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plm_special.models.selectors import IdentitySelector, RecentTimestepSelector
        cls.IdentitySelector = IdentitySelector
        cls.RecentTimestepSelector = RecentTimestepSelector

    def test_identity_and_h20_are_exact_for_full_abr_window(self):
        embeddings = torch.arange(167 * 3, dtype=torch.float32).reshape(1, 167, 3)
        mask = torch.ones((1, 167), dtype=torch.long)
        identity = self.IdentitySelector()(embeddings, mask)
        h20 = self.RecentTimestepSelector(20)(embeddings, mask)
        self.assertTrue(torch.equal(identity.embeddings, h20.embeddings))
        self.assertTrue(torch.equal(identity.attention_mask, h20.attention_mask))
        self.assertTrue(torch.equal(identity.selected_indices, h20.selected_indices))

    def test_draft_suffix_override_is_kept_exactly(self):
        embeddings = torch.arange(183, dtype=torch.float32).reshape(1, 183, 1)
        output = self.RecentTimestepSelector(10)(
            embeddings, context={'protected_suffix_tokens': 23}
        )
        self.assertEqual(output.selected_length, 103)
        self.assertTrue(torch.equal(output.embeddings[:, -23:], embeddings[:, -23:]))
        self.assertEqual(output.metadata['protected_suffix_tokens'], 23)


if __name__ == '__main__':
    unittest.main()
