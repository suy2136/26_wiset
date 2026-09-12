import unittest

from analysis.evaluate_cached_patch_projector_comparison import (
    PATCH_CONFIG,
    summarize,
)


class CachedPatchProjectorComparisonTests(unittest.TestCase):
    def test_quality_patch_configuration_is_fixed(self):
        self.assertEqual(PATCH_CONFIG, {
            "policy": "gated-k1",
            "threshold": 6.0,
            "max_skip": 0,
            "projector_cache": True,
        })

    def test_three_seed_summary_and_deltas(self):
        rows = []
        for case, mae, latency in (
            ("original_projector", 20.0, 200.0),
            ("tuned_projector", 19.0, 180.0),
        ):
            for seed in (1, 2, 3):
                rows.append({
                    "case": case,
                    "status": "complete",
                    "evaluation_seed": seed,
                    "mae": mae,
                    "rmse": mae + 1,
                    "latency_mean_ms": latency,
                    "latency_p50_ms": latency - 5,
                    "latency_p95_ms": latency + 5,
                    "selected_patches_mean": 1,
                    "visual_tokens_per_call": 1,
                })
        summary = summarize(rows)
        self.assertEqual(len(summary), 2)
        self.assertAlmostEqual(
            summary[1]["mae_change_percent_vs_original"], -5.0
        )
        self.assertAlmostEqual(
            summary[1]["latency_reduction_percent_vs_original"], 10.0
        )

    def test_missing_seed_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "seeds 1, 2, 3"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
