import unittest

from analysis.evaluate_vp_tuned_projector_full_stack_multiseed import (
    PATCH_CONFIG,
    summarize,
)


class TunedProjectorFullStackMultiseedTests(unittest.TestCase):
    def test_fixed_full_stack_configuration(self):
        self.assertEqual(PATCH_CONFIG, {
            "policy": "gated-k1",
            "threshold": 6.0,
            "max_skip": 0,
            "projector_cache": True,
        })

    def test_summary_includes_effective_forward_metrics(self):
        rows = []
        for seed in (1, 2, 3):
            rows.append({
                "status": "complete",
                "evaluation_seed": seed,
                "mae": 17.2,
                "rmse": 31.0,
                "latency_mean_ms": 200.0,
                "latency_p50_ms": 190.0,
                "latency_p95_ms": 240.0,
                "mean_initial_token_count": 10.2,
                "mean_selected_token_count": 8.2,
                "mean_target_forward_count": 5.0,
                "mean_token_reduction_percent": 19.6,
                "draft_acceptance_rate": 0.91,
                "selected_patches_mean": 0.2,
                "visual_tokens_per_call": 0.2,
            })
        result = summarize(rows)
        self.assertAlmostEqual(
            result["target_forward_reduction_percent_vs_ar20"], 75.0
        )
        self.assertAlmostEqual(
            result["effective_ms_per_target_forward"], 40.0
        )

    def test_incomplete_seed_set_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "seeds 1, 2, 3"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
