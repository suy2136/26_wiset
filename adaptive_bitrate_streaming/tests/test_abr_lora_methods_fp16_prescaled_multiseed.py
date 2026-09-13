import argparse
from pathlib import Path
import unittest

from adaptive_bitrate_streaming.analysis import (
    run_abr_lora_methods_fp16_prescaled_multiseed as pipeline,
)


def args():
    return argparse.Namespace(
        base_model_dir=Path("base"), exp_pool_path=Path("pool.pkl"),
        rank_budget=1536, trace="fcc-test", trace_num=100,
        video="video1", device="cuda:0",
    )


class ABRLoRAMethodPrescaledTest(unittest.TestCase):
    def test_all_commands_are_inference_only_and_share_protocol(self):
        for method, spec in pipeline.METHODS.items():
            command = pipeline.build_command(
                args(), method, Path(f"checkpoint_{method}"), 2
            )
            self.assertIn("--test", command)
            self.assertNotIn("--adapt", command)
            self.assertIn("--fp16-attention-prescaled-qk", command)
            self.assertNotIn("--fp16-attention-fp32-scores", command)
            self.assertIn("per-episode", command)
            self.assertIn("none", command)
            self.assertNotIn("--nbs-compact-inference", command)
            self.assertNotIn("--nbs-v19", command)
            self.assertEqual(
                command[command.index("--rank") + 1],
                str(spec["physical_rank"]),
            )

    def test_method_specific_loader_arguments(self):
        uniform = pipeline.build_command(args(), "uniform", Path("u"), 1)
        adalora = pipeline.build_command(args(), "adalora", Path("a"), 1)
        shapley = pipeline.build_command(args(), "shapley", Path("s"), 1)
        eva = pipeline.build_command(args(), "eva", Path("e"), 1)
        self.assertEqual(uniform[uniform.index("--lora-method") + 1], "uniform")
        self.assertIn("--adalora-rank-budget", adalora)
        self.assertIn("--shapley-permutations", shapley)
        self.assertEqual(
            eva[eva.index("--eva-state-path") + 1],
            str(Path("e") / "eva_state.pt"),
        )

    def test_summary_aggregates_four_methods_and_three_seeds(self):
        rows = []
        for index, method in enumerate(pipeline.METHODS):
            for seed in pipeline.SEEDS:
                rows.append({
                    "method": method, "evaluation_seed": seed,
                    "mean_reward": 0.7 + index * 0.05 + seed * 0.01,
                    "inference_latency_mean_ms": 80 + index + seed,
                })
        summary = pipeline.summarize(rows)
        self.assertEqual(len(summary), 4)
        self.assertEqual(summary[0]["num_seeds"], 3)
        self.assertAlmostEqual(summary[0]["mean_reward_mean"], 0.72)
        self.assertAlmostEqual(summary[0]["inference_latency_mean_ms_mean"], 82)


if __name__ == "__main__":
    unittest.main()
