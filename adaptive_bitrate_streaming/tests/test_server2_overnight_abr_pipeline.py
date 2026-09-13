import argparse
import tempfile
import unittest
from pathlib import Path

from adaptive_bitrate_streaming.analysis import (
    run_server2_overnight_abr_pipeline as pipeline,
)


class Server2OvernightABRPipelineTests(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(
            output_dir=Path(root) / "output",
            base_model_dir=Path(root) / "base",
            exp_pool_path=Path(root) / "pool.pkl",
            device="cuda:0",
            trace="fcc-test",
            trace_num=100,
            video="video1",
            resume=False,
            dry_run=True,
        )

    def test_stage_order_matches_requested_queue(self):
        self.assertEqual(
            pipeline.STAGE_ORDER,
            (
                "train_eva_data1",
                "evaluate_eva_data1",
                "validate_existing_data3",
                "evaluate_adalora_data3",
                "evaluate_nbs_data3_modules",
                "train_uniform_data3",
                "evaluate_uniform_data3",
            ),
        )

    def test_new_training_is_capacity_and_seed_matched(self):
        self.assertEqual(pipeline.EVA_DATA1["rank_budget"], 1536)
        self.assertEqual(pipeline.EVA_DATA1["data_seed"], 1)
        self.assertEqual(pipeline.EVA_DATA1["physical_rank"], 32)
        self.assertEqual(pipeline.UNIFORM_DATA3["rank_budget"], 1536)
        self.assertEqual(pipeline.UNIFORM_DATA3["data_seed"], 3)
        self.assertEqual(pipeline.UNIFORM_DATA3["physical_rank"], 24)

    def test_training_retains_only_best_and_latest(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            for key, experiment in (
                ("eva", pipeline.EVA_DATA1),
                ("uniform", pipeline.UNIFORM_DATA3),
            ):
                train_args = pipeline.training_args(args, experiment, key)
                command = pipeline.training.build_training_command(
                    train_args, experiment,
                )
                self.assertIn("--checkpoint-retention", command)
                index = command.index("--checkpoint-retention")
                self.assertEqual(command[index + 1], "best-latest")
                self.assertIn("--run-tag", command)

    def test_all_evaluations_use_prescaled_qk_and_episode_reseed(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            command_args = argparse.Namespace(
                base_model_dir=args.base_model_dir,
                exp_pool_path=args.exp_pool_path,
                rank_budget=1536,
                trace=args.trace,
                trace_num=args.trace_num,
                video=args.video,
                device=args.device,
            )
            command = pipeline.lora_eval.build_command(
                command_args, "eva", Path(root) / "checkpoint", 2,
            )
            self.assertIn("--fp16-attention-prescaled-qk", command)
            index = command.index("--evaluation-rng-mode")
            self.assertEqual(command[index + 1], "per-episode")

            nbs_args = pipeline.nbs_module_args(
                args, Path(root) / "nbs-checkpoint",
            )
            nbs_args.data_seed = 2
            nbs_command = pipeline.module_sweep.best5.build_command(
                nbs_args, pipeline.latest_modules.FULL,
            )
            self.assertIn("--fp16-attention-prescaled-qk", nbs_command)
            index = nbs_command.index("--evaluation-rng-mode")
            self.assertEqual(nbs_command[index + 1], "per-episode")
            self.assertIn("--nbs-compact-inference", nbs_command)


if __name__ == "__main__":
    unittest.main()
