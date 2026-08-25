from pathlib import Path
import tempfile
import unittest

from adaptive_bitrate_streaming.analysis import (
    run_nbs_v19_budget8192_experiment as runner,
)


class Args:
    base_model_dir = Path('/models/llama')
    exp_pool_path = Path('/data/exp_pool.pkl')
    video = 'video1'
    device = 'cuda:0'
    train_trace = 'fcc-valid'
    trace_num = 100
    grad_accum_steps = 32
    lr = 0.0001
    warmup_steps = 2000
    num_epochs = 80
    eval_per_epoch = 2
    early_stopping_patience = 10
    early_stopping_min_epochs = 20
    early_stopping_min_delta = 0.003
    plateau_lr_patience = 5
    plateau_lr_factor = 0.5
    plateau_min_lr = 1e-6


class NBSBudget8192RunnerTest(unittest.TestCase):
    def test_warm_start_command_begins_after_latest_epoch(self):
        command = runner.training_command(
            Args(), warm_start=Path('/checkpoints/latest'), start_epoch=7,
            best_return=4.5,
        )
        joined = ' '.join(command)
        warm_index = command.index('--warm-start-model-dir')
        self.assertEqual(Path(command[warm_index + 1]).name, 'latest')
        self.assertIn('--start-epoch 7', joined)
        self.assertIn('--initial-best-return 4.5', joined)
        self.assertIn('--save-checkpoint-per-epoch 10', joined)
        self.assertIn('--checkpoint-retention best-latest', joined)

    def test_previous_best_return_is_parsed_before_log_append(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / 'early_stop_-1_checkpoint'
            checkpoint_root.mkdir()
            log = checkpoint_root.parent / 'early_stop_-1_console.log'
            log.write_text(
                "{'best_return': 4.1}\n{'best_return': 4.5}\n",
                encoding='utf-8',
            )
            self.assertEqual(runner.previous_best_return(checkpoint_root), 4.5)


if __name__ == '__main__':
    unittest.main()
