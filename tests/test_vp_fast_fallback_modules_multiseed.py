import argparse
from pathlib import Path
import unittest

from analysis import evaluate_vp_fast_fallback_modules_multiseed as pipeline


def args():
    return argparse.Namespace(
        compact_checkpoint=Path("compact"),
        projector_checkpoint=Path("projector_2ep.pth"),
        cache_dir=Path("cache"),
        output_dir=Path("output"),
        device="cuda:0",
        rank_budget=512,
        physical_rank=32,
        latency_warmup_steps=5,
        projector_cache_max_entries=512,
    )


class VPFastFallbackModulesTest(unittest.TestCase):
    def test_has_four_requested_cases_without_full_stack(self):
        self.assertEqual([item[2] for item in pipeline.CASES], [
            "baseline", "patch", "token", "speculative",
        ])

    def test_each_command_uses_continuous_rng_and_requested_seed(self):
        for _, _, kind in pipeline.CASES:
            command = pipeline.command_for(args(), kind, 3, Path("result") / kind)
            self.assertEqual(
                command[command.index("--evaluation-rng-mode") + 1],
                "continuous",
            )
            self.assertEqual(command[command.index("--data-seed") + 1], "3")
            self.assertEqual(
                command[command.index("--adalora-rank-budget") + 1], "512"
            )
            self.assertEqual(command[command.index("--rank") + 1], "32")

    def test_module_specific_options(self):
        commands = {
            kind: pipeline.command_for(args(), kind, 1, Path("result") / kind)
            for _, _, kind in pipeline.CASES
        }
        patch = commands["patch"]
        self.assertEqual(
            patch[patch.index("--cached-patch-policy") + 1], "gated-k1"
        )
        self.assertEqual(
            patch[patch.index("--cached-patch-motion-threshold-deg") + 1],
            "6.0",
        )
        self.assertEqual(
            patch[patch.index("--cached-patch-max-skip-calls") + 1], "0"
        )
        self.assertIn("--cached-patch-projector-cache", patch)
        token = commands["token"]
        self.assertEqual(token[token.index("--selector-recent-k") + 1], "8")
        spec = commands["speculative"]
        self.assertEqual(spec[spec.index("--speculative-gamma") + 1], "6")
        self.assertEqual(
            spec[spec.index("--speculative-threshold") + 1], "0.4"
        )

    def test_summary_uses_three_seeds_and_baseline_deltas(self):
        rows = []
        for index, (case, label, _) in enumerate(pipeline.CASES):
            for seed in pipeline.SEEDS:
                rows.append({
                    "case": case,
                    "label": label,
                    "evaluation_seed": seed,
                    "status": "complete",
                    "mae": 17.0 + index / 10,
                    "rmse": 31.0,
                    "latency_mean_ms": 700.0 - index * 100,
                    "fp16_prescaled_qk_fallback_calls": 0,
                    "fp32_attention_fallback_calls": 0,
                })
        summaries = pipeline.summarize(rows)
        self.assertEqual(len(summaries), 4)
        self.assertEqual(summaries[0]["mae_change_percent_vs_nbs"], 0.0)
        self.assertGreater(summaries[-1]["latency_reduction_percent_vs_nbs"], 0)
        self.assertTrue(all(row["seed_count"] == 3 for row in summaries))


if __name__ == "__main__":
    unittest.main()
