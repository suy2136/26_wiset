import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from adaptive_bitrate_streaming.analysis.run_abr_c1536_data2_allocators import (
    EXPERIMENTS,
)
from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
    build_eva_precompute_command,
    build_test_command,
    build_training_command,
    experiment_data_seed,
    experiment_lora_seed,
    experiment_seed,
    signature,
)


class AbrSeedSeparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.args = Namespace(
            base_model_dir=root / "base",
            exp_pool_path=root / "pool.pkl",
            output=root / "results.csv",
            device="cuda:0",
            num_epochs=80,
            eval_per_epoch=2,
            early_stopping_patience=10,
            early_stopping_min_epochs=20,
            early_stopping_min_delta=0.003,
            plateau_lr_patience=5,
            plateau_lr_factor=0.5,
            plateau_min_lr=1e-6,
            train_trace="fcc-valid",
            test_trace="fcc-test",
            trace_num=100,
            video="video1",
            grad_accum_steps=32,
            nbs_rollback_backup_device="cpu",
            nbs_max_rollback_backup_mib=2048,
            nbs_update_ratio_warning=0.01,
            nbs_max_update_ratio=0.05,
            nbs_update_ratio_floor=0.01,
            nbs_max_update_rms=0.01,
            nbs_rollback_lr_factor=0.5,
            nbs_rollback_min_lr=1e-5,
            nbs_skip_batch_at_rollback_lr_floor=True,
            nbs_max_consecutive_rollbacks=3,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_methods_share_capacity_and_component_seeds(self):
        self.assertEqual(
            [item["method"] for item in EXPERIMENTS],
            ["nbs", "uniform_lora", "adalora", "eva", "shapley"],
        )
        for experiment in EXPERIMENTS:
            self.assertEqual(experiment["rank_budget"], 1536)
            self.assertEqual(experiment_seed(experiment), 1)
            self.assertEqual(experiment_lora_seed(experiment), 1)
            self.assertEqual(experiment_data_seed(experiment), 2)

    def test_train_and_test_commands_carry_all_three_seeds(self):
        for experiment in EXPERIMENTS:
            commands = (
                build_training_command(self.args, experiment),
                build_test_command(self.args, experiment, Path("/checkpoint")),
            )
            for command in commands:
                self.assertEqual(command[command.index("--seed") + 1], "1")
                self.assertEqual(command[command.index("--lora-seed") + 1], "1")
                self.assertEqual(command[command.index("--data-seed") + 1], "2")

    def test_eva_calibration_uses_data_seed(self):
        command = build_eva_precompute_command(self.args, EXPERIMENTS[3])
        self.assertEqual(command[command.index("--seed") + 1], "2")

    def test_nbs_training_enables_rollback_floor_and_batch_skip(self):
        command = build_training_command(self.args, EXPERIMENTS[0])
        self.assertEqual(
            command[command.index("--nbs-rollback-min-lr") + 1], "1e-05"
        )
        self.assertIn("--nbs-skip-batch-at-rollback-lr-floor", command)

    def test_signature_records_component_seeds(self):
        seeds = signature(self.args, EXPERIMENTS)["seeds"]
        self.assertEqual(
            seeds["NBS_C1536_DATA2"],
            {"master": 1, "lora": 1, "data": 2},
        )


if __name__ == "__main__":
    unittest.main()
