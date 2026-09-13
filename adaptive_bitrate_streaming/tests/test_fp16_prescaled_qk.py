import argparse
import math
from pathlib import Path
import unittest

from adaptive_bitrate_streaming.analysis import (
    run_nbs_v19_best5_inference as best5,
)

try:
    import torch
except ModuleNotFoundError:  # Lightweight local validation environment.
    torch = None

if torch is not None:
    from adaptive_bitrate_streaming.plm_special.models.low_rank import (
        _fp16_prescaled_qk_scores,
    )
    from adaptive_bitrate_streaming.plm_special.models.rl_policy import RLPolicy


if torch is not None:
    class _Attention:
        _abr_attention_score_mode = "fp16_prescaled"
        _abr_force_fp32_attention_scores = False


    class _PLM:
        def __init__(self):
            self.attention = _Attention()

        def modules(self):
            return [self.attention]

        def __call__(self, **kwargs):
            finite = self.attention._abr_force_fp32_attention_scores
            value = 1.0 if finite else float("nan")
            return {"last_hidden_state": torch.full((1, 2, 3), value)}


    class _Policy:
        _require_finite = RLPolicy._require_finite
        _run_plm = RLPolicy._run_plm
        residual = False
        which_layer = -1

        def __init__(self):
            self.plm = _PLM()

        def _plm_compute_dtype(self):
            return torch.float32

        def _adalora_overflow_candidates(self):
            return []


class FP16PrescaledQKTest(unittest.TestCase):
    def test_prescaled_formula_reduces_the_fp16_intermediate_range(self):
        head_dim = 128
        value = 30.0
        unscaled = head_dim * value * value
        operand_scale = math.sqrt(math.sqrt(head_dim))
        prescaled = head_dim * (value / operand_scale) ** 2
        expected = unscaled / math.sqrt(head_dim)
        self.assertGreater(unscaled, 65504.0)
        self.assertLess(prescaled, 65504.0)
        self.assertAlmostEqual(prescaled, expected)

    @unittest.skipIf(torch is None, "PyTorch is unavailable locally")
    def test_prescaling_avoids_unscaled_fp16_overflow(self):
        query = torch.full((1, 1, 1, 128), 30.0, dtype=torch.float16)
        key = torch.full((1, 1, 1, 128), 30.0, dtype=torch.float16)
        unscaled = torch.matmul(query, key.transpose(2, 3))
        self.assertFalse(torch.isfinite(unscaled).all())
        actual = _fp16_prescaled_qk_scores(query, key, 128)
        expected = torch.matmul(
            query.float(), key.transpose(2, 3).float()
        ) / (128 ** 0.5)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2.0)

    @unittest.skipIf(torch is None, "PyTorch is unavailable locally")
    def test_nonfinite_plm_call_retries_once_with_fp32_scores(self):
        policy = _Policy()
        result = policy._run_plm(torch.zeros((1, 2, 3)), None)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(policy.fp16_prescaled_qk_fallback_calls, 1)
        self.assertFalse(policy.plm.attention._abr_force_fp32_attention_scores)

    def test_command_selects_prescaled_qk_without_full_fp32_mode(self):
        args = argparse.Namespace(
            data_seed=3, lora_seed=1, evaluation_rng_mode="per-episode",
            run_tag="per_episode_reseed_fp16_prescaled_qk",
            base_model_dir=Path("base"), checkpoint_dir=Path("checkpoint"),
            exp_pool_path=Path("pool"), physical_rank=32, rank_budget=1536,
            rank_config=Path("rank.json"), trace="fcc-test", trace_num=100,
            video="video1", device="cuda:0", fp16_numeric_safeguards=True,
            fp16_selective_clamp=True, fp16_selective_clamp_threshold=60000.0,
            fp16_attention_fp32_scores=False,
            fp16_attention_prescaled_qk=True,
            nbs_compaction_rtol=5e-3, nbs_compaction_atol=5e-3,
        )
        command = best5.build_command(args, {"temporal": False})
        self.assertIn("--fp16-attention-prescaled-qk", command)
        self.assertNotIn("--fp16-attention-fp32-scores", command)


if __name__ == "__main__":
    unittest.main()
