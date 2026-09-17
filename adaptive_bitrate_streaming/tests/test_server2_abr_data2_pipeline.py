from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from adaptive_bitrate_streaming.analysis import run_server2_abr_data2_pipeline as pipeline


class Server2AbrData2PipelineTest(unittest.TestCase):
    def args(self, root: Path):
        return argparse.Namespace(
            output_dir=root / "run", base_model_dir=root / "model",
            exp_pool_path=root / "pool.pkl", device="cuda:0",
            trace="fcc-test", trace_num=100, video="video1",
            resume=False, dry_run=True,
        )

    def test_priority_and_method_order(self):
        self.assertEqual(pipeline.STAGE_ORDER[:2], ("train_nbs", "evaluate_nbs_modules"))
        self.assertEqual(pipeline.METHOD_ORDER, ("nbs", "uniform", "adalora", "shapley", "eva"))

    def test_experiment_seed_budget_and_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            for method in pipeline.METHOD_ORDER:
                experiment = pipeline.experiment_for(args, method)
                self.assertEqual(experiment["seed"], 1)
                self.assertEqual(experiment["lora_seed"], 1)
                self.assertEqual(experiment["data_seed"], 2)
                self.assertEqual(experiment["rank_budget"], 1536)
            self.assertEqual(pipeline.experiment_for(args, "uniform")["physical_rank"], 24)
            self.assertEqual(pipeline.experiment_for(args, "adalora")["adalora_schedule_epochs"], 20)
            self.assertEqual(pipeline.experiment_for(args, "shapley")["adalora_schedule_epochs"], 20)

    def test_training_command_uses_prescaled_qk_and_schedule_safe_early_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            experiment = pipeline.experiment_for(args, "adalora")
            command = pipeline.add_prescaled_qk(
                pipeline.training.build_training_command(
                    pipeline.training_args(args, "adalora_data2"), experiment,
                )
            )
        self.assertIn("--fp16-attention-prescaled-qk", command)
        self.assertNotIn("--fp16-attention-fp32-scores", command)
        self.assertEqual(command[command.index("--num-epochs") + 1], "80")
        self.assertEqual(command[command.index("--early-stopping-min-epochs") + 1], "20")
        self.assertEqual(command[command.index("--adalora-schedule-epochs") + 1], "20")

    def test_module_args_use_reseed_and_prescaled_qk(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            module_args = pipeline.nbs_module_args(args, Path("checkpoint"))
            module_args.data_seed = 3
            module_args.lora_seed = 1
            command = pipeline.module_sweep.best5.build_command(
                module_args, pipeline.latest_modules.FULL,
            )
        self.assertEqual(module_args.evaluation_rng_mode, "per-episode")
        self.assertTrue(module_args.fp16_attention_prescaled_qk)
        self.assertFalse(module_args.fp16_attention_fp32_scores)
        self.assertIn("--fp16-attention-prescaled-qk", command)
        self.assertNotIn("--fp16-attention-fp32-scores", command)
        self.assertEqual(command[command.index("--seed") + 1], "3")
        self.assertEqual(command[command.index("--data-seed") + 1], "3")
        self.assertEqual(command[command.index("--evaluation-rng-mode") + 1], "per-episode")

    def test_budget_mismatch_warns_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "adapter_config.json").write_text(
                json.dumps({"r": 23, "target_modules": ["q_proj", "v_proj"]}),
                encoding="utf-8",
            )
            (checkpoint / "modules_except_plm.bin").touch()
            (checkpoint / "adapter_model.bin").touch()
            args = self.args(checkpoint)
            experiment = pipeline.experiment_for(args, "uniform")
            metadata = {
                "variant": "uniform_lora", "role": "best",
                "seed": 1, "lora_seed": 1, "data_seed": 2,
                "physical_rank": 24, "effective_rank_budget": 1536,
                "run_tag": experiment["run_tag"],
            }
            (checkpoint / "checkpoint_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8",
            )
            # The adapter itself is rank 23 x 64 = 1472. The mismatch is
            # recorded without blocking even though metadata claims 1536.
            result = pipeline.inspect_budget(checkpoint, "uniform", experiment)
        self.assertEqual(result["active_rank_total"], 1472)
        self.assertFalse(result["budget_match"])
        self.assertEqual(result["metadata_effective_rank_budget"], 1536)

    def test_seed4_uses_separate_output_and_training_identity(self):
        parsed = pipeline.parse_args(["--training-data-seed", "4", "--dry-run"])
        self.assertEqual(parsed.output_dir.name, "server2_abr_data4_pipeline")
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(pipeline, "TRAINING_DATA_SEED", 4):
            args = self.args(Path(temporary))
            experiment = pipeline.experiment_for(args, "nbs")
            self.assertEqual(experiment["data_seed"], 4)
            self.assertIn("data4", experiment["run_tag"])
            self.assertEqual(pipeline.signature(args)["training_data_seed"], 4)


if __name__ == "__main__":
    unittest.main()
