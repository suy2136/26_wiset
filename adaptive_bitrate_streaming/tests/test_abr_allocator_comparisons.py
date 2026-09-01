import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from adaptive_bitrate_streaming.analysis.run_abr_c_matched_allocators import (
    EXPERIMENTS,
)
from adaptive_bitrate_streaming.analysis.run_abr_c_matched_server1 import (
    EXPERIMENTS as SERVER1_SOURCE_EXPERIMENTS,
)
from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
    build_eva_precompute_command,
    build_test_command,
    build_training_command,
    expected_checkpoint_role,
    expected_variant,
    load_state,
    signature,
)


class AllocatorComparisonCommandsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.args = Namespace(
            base_model_dir=root / 'base',
            exp_pool_path=root / 'pool.pkl',
            output=root / 'results.csv',
            device='cuda:0',
            num_epochs=80,
            eval_per_epoch=2,
            early_stopping_patience=10,
            early_stopping_min_epochs=20,
            early_stopping_min_delta=0.003,
            plateau_lr_patience=5,
            plateau_lr_factor=0.5,
            plateau_min_lr=1e-6,
            train_trace='fcc-valid',
            test_trace='fcc-test',
            trace_num=100,
            video='video1',
            grad_accum_steps=32,
            nbs_rollback_backup_device='cpu',
            nbs_max_rollback_backup_mib=2048,
            nbs_update_ratio_warning=0.01,
            nbs_max_update_ratio=0.05,
            nbs_update_ratio_floor=0.01,
            nbs_max_update_rms=0.01,
            nbs_rollback_lr_factor=0.5,
            nbs_max_consecutive_rollbacks=3,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_c_matched_methods_and_budget(self):
        self.assertEqual(
            [item['method'] for item in EXPERIMENTS],
            ['adalora', 'eva', 'shapley'],
        )
        self.assertTrue(all(item['rank_budget'] == 1536 for item in EXPERIMENTS))
        self.assertTrue(all(item['physical_rank'] == 32 for item in EXPERIMENTS))
        self.assertTrue(all(item['lr'] == 2e-4 for item in EXPERIMENTS))

    def test_server_split_keeps_one_shared_experiment_definition(self):
        self.assertIs(SERVER1_SOURCE_EXPERIMENTS, EXPERIMENTS)
        self.assertEqual(
            [item['name'] for item in EXPERIMENTS[:2]],
            ['ADALORA_C1536', 'EVA_C1536'],
        )
        self.assertEqual(EXPERIMENTS[2]['name'], 'SHAPLEY_C1536')

    def test_dynamic_commands_reach_exact_target_before_comparison(self):
        for experiment in (EXPERIMENTS[0], EXPERIMENTS[2]):
            command = build_training_command(self.args, experiment)
            self.assertIn('--adalora-rank-budget', command)
            self.assertIn('1536', command)
            self.assertIn('--adalora-schedule-epochs', command)
            self.assertEqual(expected_checkpoint_role(experiment), 'final')
            self.assertEqual(expected_variant(experiment), experiment['method'])

    def test_eva_precompute_train_and_test_are_state_bound(self):
        experiment = EXPERIMENTS[1]
        precompute = build_eva_precompute_command(self.args, experiment)
        train = build_training_command(self.args, experiment)
        checkpoint = Path(self.temporary.name) / 'checkpoint'
        test = build_test_command(self.args, experiment, checkpoint)
        self.assertIn('analysis/precompute_abr_eva.py', precompute)
        self.assertIn('--rank-budget', precompute)
        self.assertEqual(
            precompute[precompute.index('--max-batches') + 1], '512'
        )
        self.assertIn('--allow-unconverged', precompute)
        self.assertIn('--eva-state-path', train)
        self.assertIn(str((checkpoint / 'eva_state.pt').resolve()), test)
        self.assertEqual(expected_checkpoint_role(experiment), 'best')
        self.assertEqual(expected_variant(experiment), 'eva')

    def test_server1_shell_supports_checkpoint_resume(self):
        script = (
            Path(__file__).resolve().parents[1]
            / 'scripts' / 'run_abr_c_matched_server1.sh'
        ).read_text(encoding='utf-8')
        self.assertIn('"resume"', script)
        self.assertIn('ARGS+=(--resume)', script)

    def test_resume_migrates_only_unfinished_eva_calibration_cap(self):
        previous = copy.deepcopy(EXPERIMENTS)
        previous[1].pop('eva_allow_unconverged', None)
        previous[1]['eva_max_batches'] = 128
        previous_signature = signature(self.args, previous)
        state_path = Path(self.temporary.name) / 'state.json'
        state_path.write_text(
            json.dumps({
                'signature': previous_signature,
                'runs': {
                    'ADALORA_C1536': {
                        'status': 'complete',
                        'checkpoint_dir': '/saved/adalora',
                    }
                },
            }),
            encoding='utf-8',
        )

        requested_signature = signature(self.args, EXPERIMENTS)
        restored = load_state(state_path, True, requested_signature)

        self.assertEqual(restored['signature'], requested_signature)
        self.assertEqual(
            restored['runs']['ADALORA_C1536']['checkpoint_dir'],
            '/saved/adalora',
        )


if __name__ == '__main__':
    unittest.main()
