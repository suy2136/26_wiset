import argparse
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from analysis import run_server1_overnight_pipeline as pipeline


class Server1OvernightPipelineTests(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(
            output_dir=Path(root) / "output", device="cuda:0",
            latency_warmup_steps=5,
            base_model_dir=Path(root) / "base",
            exp_pool_path=Path(root) / "pool.pkl",
            tuned_projector=Path(root) / "projector.pth",
            patch_cache=Path(root) / "cache", resume=False, dry_run=True,
        )

    def test_stage_order(self):
        self.assertEqual(
            pipeline.STAGE_ORDER[:6],
            (
                "train_vp_adalora_data1", "evaluate_vp_adalora_data1",
                "train_vp_shapley_data1", "evaluate_vp_shapley_data1",
                "validate_abr_data3", "evaluate_abr_data3_eva_shapley",
            ),
        )

    def test_vp_commands_use_prescaled_qk_and_continuous_rng(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            command = pipeline.vp_module_command(
                args, Path(root) / "compact", "full_stack", 2,
                Path(root) / "result",
            )
            self.assertIn("--vp-fp16-prescaled-qk", command)
            self.assertNotIn("--vp-fp16-fallback", command)
            index = command.index("--evaluation-rng-mode")
            self.assertEqual(command[index + 1], "continuous")
            self.assertIn("--nbs-inference-mode", command)
            self.assertEqual(command[command.index("--seed") + 1], "2")

    def test_abr_commands_use_prescaled_qk_and_episode_reseed(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            command_args = argparse.Namespace(
                base_model_dir=args.base_model_dir,
                exp_pool_path=args.exp_pool_path, rank_budget=1536,
                trace="fcc-test", trace_num=100, video="video1",
                device=args.device,
            )
            command = pipeline.abr_lora.build_command(
                command_args, "eva", Path(root) / "checkpoint", 3,
            )
            self.assertIn("--fp16-attention-prescaled-qk", command)
            self.assertEqual(
                command[command.index("--evaluation-rng-mode") + 1],
                "per-episode",
            )

    def test_signature_keeps_existing_assets_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            signature = pipeline.signature(self.args(root))
            self.assertEqual(signature["vp_evaluation_rng_mode"], "continuous")
            self.assertEqual(signature["abr_evaluation_rng_mode"], "per-episode")
            self.assertEqual(signature["vp_module_settings"]["token_k"], 8)

    def test_budget_mismatch_is_recorded_but_not_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root)
            with mock.patch.object(
                pipeline.vp_lora,
                "checkpoint_description",
                return_value={"active_rank_total": 911},
            ):
                result = pipeline.inspect_vp_budget("shapley", checkpoint)
            self.assertEqual(result["active_rank_total"], 911)


if __name__ == "__main__":
    unittest.main()
