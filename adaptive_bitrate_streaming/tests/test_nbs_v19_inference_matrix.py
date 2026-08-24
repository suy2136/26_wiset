import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'run_nbs_v19_inference_matrix.py'
SPEC = importlib.util.spec_from_file_location('nbs_v19_matrix', SCRIPT)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class NBSV19InferenceMatrixTest(unittest.TestCase):
    def test_fixed_matrix_has_requested_ablation_counts(self):
        experiments = matrix.EXPERIMENTS
        self.assertEqual(len(experiments), 9)
        self.assertEqual(
            [item['history'] for item in experiments if item['name'].startswith('selector_')],
            [5, 10, 15],
        )
        self.assertEqual(
            [item['draft_steps'] for item in experiments if item['name'].startswith('speculative_')],
            [2, 3, 4],
        )
        self.assertEqual(
            len([item for item in experiments if item['name'].startswith('combined_')]),
            2,
        )

    def test_every_command_uses_same_nbs_seed_and_checkpoint(self):
        args = argparse.Namespace(
            base_model_dir=Path('base'), checkpoint_dir=Path('checkpoint'),
            exp_pool_path=Path('pool.pkl'), device='cuda:0', trace='fcc-test',
            trace_num=100, video='video1', buffer_tolerance=1.0,
            state_tolerance=0.25, return_tolerance=0.01, rank_budget=512,
        )
        for experiment in matrix.EXPERIMENTS:
            command = matrix.build_command(args, experiment)
            self.assertIn('--nbs-v19', command)
            self.assertIn('--fp16', command)
            self.assertEqual(command[command.index('--seed') + 1], '1')
            self.assertEqual(command[command.index('--rank') + 1], '32')
            self.assertEqual(
                command[command.index('--model-dir') + 1],
                str(args.checkpoint_dir.resolve()),
            )

    def test_checkpoint_validation_checks_nbs_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            for filename in (
                'adapter_config.json', 'adapter_model.bin',
                'modules_except_plm.bin', 'nash_rank_allocator.pt',
            ):
                (checkpoint / filename).touch()
            with (checkpoint / 'checkpoint_metadata.json').open(
                'w', encoding='utf-8'
            ) as stream:
                json.dump({
                    'variant': 'nbs_v19', 'seed': 1,
                    'effective_rank_budget': 512,
                }, stream)
            matrix.validate_checkpoint(checkpoint)

    def test_baseline_comparisons_report_reward_and_speedup(self):
        rows = [
            {
                'experiment': 'nbs_only', 'mean_reward': 10.0,
                'inference_latency_mean_ms': 20.0, 'time': 100.0,
            },
            {
                'experiment': 'selector_h5', 'mean_reward': 9.5,
                'inference_latency_mean_ms': 10.0, 'time': 80.0,
            },
        ]
        matrix.add_baseline_comparisons(rows)
        self.assertEqual(rows[1]['mean_reward_delta_vs_nbs'], -0.5)
        self.assertEqual(rows[1]['inference_speedup_vs_nbs'], 2.0)
        self.assertEqual(rows[1]['inference_latency_reduction_vs_nbs'], 0.5)
        self.assertEqual(rows[1]['test_time_speedup_vs_nbs'], 1.25)

    def test_resume_rejects_a_different_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'matrix.csv'
            rows = [{'experiment': 'nbs_only'}]
            matrix.write_results(rows, output, signature={'trace': 'fcc-test'})
            with self.assertRaisesRegex(ValueError, 'does not match'):
                matrix.load_resume_rows(output, {'trace': 'fcc-valid'})


if __name__ == '__main__':
    unittest.main()
