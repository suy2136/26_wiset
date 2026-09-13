import argparse
from pathlib import Path
import unittest

from adaptive_bitrate_streaming.analysis import (
    run_nbs_v19_reseed_latest_modules_multiseed as pipeline,
)


class ABRReseedLatestModulesTest(unittest.TestCase):
    def test_exact_latest_module_matrix(self):
        self.assertEqual(len(pipeline.RUN_SPECS), 4)
        self.assertEqual(pipeline.TOKEN["token_offsets"], (0, 2, 3, 4, 7))
        self.assertEqual(pipeline.SPECULATIVE["speculative_steps"], 5)
        self.assertEqual(pipeline.FULL["event_max_events"], 1)
        self.assertEqual(pipeline.FULL["token_offsets"], (0, 2, 3, 4, 7))
        self.assertEqual(pipeline.FULL["speculative_steps"], 5)

    def test_commands_are_reseeded_and_isolated(self):
        args = argparse.Namespace(
            data_seed=2, evaluation_rng_mode="per-episode",
            base_model_dir=Path("base"), checkpoint_dir=Path("checkpoint"),
            exp_pool_path=Path("pool"), physical_rank=32, rank_budget=1536,
            rank_config=Path("config.json"), trace="fcc-test", trace_num=100,
            video="video1", device="cuda:0",
        )
        for spec in pipeline.RUN_SPECS:
            command = pipeline.sweep.best5.build_command(args, spec)
            self.assertEqual(
                command[command.index("--evaluation-rng-mode") + 1],
                "per-episode",
            )
            self.assertIn("--run-tag", command)
        temporal_command = pipeline.sweep.best5.build_command(args, pipeline.TEMPORAL)
        self.assertNotIn("--intra-token-keep-offsets", temporal_command)
        self.assertEqual(
            temporal_command[temporal_command.index("--speculative-draft-steps") + 1],
            "0",
        )

    def test_reseed_pipeline_enables_compact_fp16_safeguards(self):
        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        self.assertIn("args.fp16_numeric_safeguards = True", source)
        self.assertIn("args.fp16_selective_clamp = True", source)
        self.assertIn("args.fp16_selective_clamp_threshold = 60000.0", source)
        self.assertIn("args.fp16_attention_fp32_scores = True", source)

        command_args = argparse.Namespace(
            data_seed=3, base_model_dir=Path("base"),
            checkpoint_dir=Path("checkpoint"), exp_pool_path=Path("pool"),
            physical_rank=32, rank_budget=1536,
            rank_config=Path("rank.json"), trace="fcc-test", trace_num=100,
            video="video1", device="cuda:0", evaluation_rng_mode="per-episode",
            fp16_numeric_safeguards=True, fp16_selective_clamp=True,
            fp16_selective_clamp_threshold=60000.0,
            fp16_attention_fp32_scores=True,
        )
        command = pipeline.sweep.best5.build_command(
            command_args, pipeline.TEMPORAL
        )
        self.assertIn("--fp16-numeric-safeguards", command)
        self.assertIn("--fp16-selective-clamp", command)
        self.assertEqual(
            command[command.index("--fp16-selective-clamp-threshold") + 1],
            "60000.0",
        )
        self.assertIn("--fp16-attention-fp32-scores", command)


if __name__ == "__main__":
    unittest.main()
