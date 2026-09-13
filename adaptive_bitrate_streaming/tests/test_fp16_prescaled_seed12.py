from pathlib import Path
import tempfile
import unittest

from adaptive_bitrate_streaming.analysis import (
    run_nbs_v19_fp16_prescaled_seed12 as pipeline,
)


class FP16PrescaledSeed12PipelineTest(unittest.TestCase):
    def test_pipeline_is_isolated_to_seeds_one_and_two(self):
        self.assertEqual(pipeline.SEEDS, (1, 2))
        self.assertNotEqual(
            pipeline.DEFAULT_OUTPUT,
            pipeline.seed3_pipeline.DEFAULT_OUTPUT,
        )

    def test_dry_run_builds_ten_prescaled_fp16_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = pipeline.parse_args([
                "--checkpoint-dir", "checkpoint",
                "--base-model-dir", "base",
                "--exp-pool-path", "pool.pkl",
                "--output-dir", str(Path(temporary) / "seed12"),
                "--dry-run",
            ])
            commands = []
            original = pipeline.sweep.best5.build_command

            def capture(command_args, spec):
                command = original(command_args, spec)
                commands.append(command)
                return command

            original_run = pipeline.sweep.run_specs

            def fake_run(args, specs, seed, output, rows):
                for spec in specs:
                    capture(argparse_namespace(args, seed), spec)
                return rows

            pipeline.sweep.run_specs = fake_run
            try:
                pipeline.main([
                    "--checkpoint-dir", str(args.checkpoint_dir),
                    "--base-model-dir", str(args.base_model_dir),
                    "--exp-pool-path", str(args.exp_pool_path),
                    "--output-dir", str(args.output_dir),
                    "--dry-run",
                ])
            finally:
                pipeline.sweep.run_specs = original_run

        self.assertEqual(len(commands), 10)
        self.assertEqual(
            {command[command.index("--data-seed") + 1] for command in commands},
            {"1", "2"},
        )
        for command in commands:
            self.assertIn("--fp16-attention-prescaled-qk", command)
            self.assertNotIn("--fp16-attention-fp32-scores", command)
            self.assertIn("--nbs-compact-inference", command)

    def test_summary_requires_and_aggregates_exactly_seed_one_and_two(self):
        rows = []
        for index, spec in enumerate(pipeline.latest.TARGET_SPECS):
            for seed in pipeline.SEEDS:
                rows.append({
                    "experiment": spec["name"],
                    "data_seed": seed,
                    "mean_reward": 1.0 + index * 0.1 + seed * 0.01,
                    "inference_latency_mean_ms": 100.0 - index - seed,
                })
        summary = pipeline.summarize(rows)
        self.assertEqual(len(summary), len(pipeline.latest.TARGET_SPECS))
        self.assertEqual(summary[0]["num_seeds"], 2)
        self.assertEqual(summary[0]["data_seeds"], "1,2")
        self.assertAlmostEqual(summary[0]["mean_reward_mean"], 1.015)
        self.assertAlmostEqual(summary[0]["mean_reward_change_ratio_vs_nbs"], 0.0)


def argparse_namespace(args, seed):
    import argparse

    values = vars(args).copy()
    values.update({
        "data_seed": seed,
        "lora_seed": 1,
        "evaluation_rng_mode": "per-episode",
        "run_tag": "per_episode_reseed_fp16_prescaled_qk",
        "fp16_numeric_safeguards": True,
        "fp16_selective_clamp": True,
        "fp16_selective_clamp_threshold": 60000.0,
        "fp16_attention_fp32_scores": False,
        "fp16_attention_prescaled_qk": True,
    })
    return argparse.Namespace(**values)


if __name__ == "__main__":
    unittest.main()
