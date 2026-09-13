import unittest

try:
    import torch
    from models.vp_numeric_safety import (
        _prescaled_qk_scores,
        enable_vp_fp16_prescaled_qk,
        vp_numeric_safety_report,
    )
except ModuleNotFoundError:  # lightweight local test environments
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class VPFP16PrescaledQKTests(unittest.TestCase):
    def test_prescaling_matches_standard_formula_in_float32(self):
        query = torch.randn(1, 2, 3, 8)
        key = torch.randn(1, 2, 4, 8)
        expected = torch.matmul(query, key.transpose(2, 3)) / (8 ** 0.5)
        actual = _prescaled_qk_scores(query, key, 8)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_enable_marks_networking_model_and_report(self):
        class LlamaAttention:
            def forward(self):
                return None

        class Decoder:
            def __init__(self):
                self.attention = LlamaAttention()

            def modules(self):
                return [self.attention]

        class LlamaNetworkingHeadModel:
            def __init__(self):
                self.model = Decoder()

        class Root:
            def __init__(self):
                self.target = LlamaNetworkingHeadModel()

            def modules(self):
                return [self.target]

        root = Root()

        self.assertEqual(enable_vp_fp16_prescaled_qk(root), 1)
        report = vp_numeric_safety_report(root)
        self.assertEqual(report["enabled_models"], 1)
        self.assertEqual(report["fp16_prescaled_qk_enabled_models"], 1)


if __name__ == "__main__":
    unittest.main()
