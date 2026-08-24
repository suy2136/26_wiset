import importlib.util
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'run_nbs_v19_budget1536_speculative.py'
SPEC = importlib.util.spec_from_file_location('budget1536_sweep', SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


class NBSV19Budget1536SpeculativeTest(unittest.TestCase):
    def make_args(self):
        return sweep.parse_args([
            '--base-model-dir', 'base', '--exp-pool-path', 'pool.pkl',
            '--official-lora-dir', 'official', '--num-epochs', '4',
            '--eval-per-epoch', '1',
        ])

    def test_grid_has_exactly_40_unique_configurations(self):
        configurations = sweep.speculative_configurations()
        self.assertEqual(len(configurations), 40)
        self.assertEqual(
            len({item['experiment'] for item in configurations}), 40
        )
        self.assertEqual(
            {item['draft_steps'] for item in configurations}, {2, 3}
        )
        self.assertEqual(
            {item['buffer_tolerance'] for item in configurations},
            {0.5, 1.0, 1.5, 2.0, 3.0},
        )
        self.assertEqual(
            {item['state_tolerance'] for item in configurations},
            {0.10, 0.25, 0.40, 0.50},
        )
        self.assertEqual(
            {item['return_tolerance'] for item in configurations}, {0.01}
        )

    def test_training_uses_budget1536_and_disables_inference_features(self):
        command = sweep.build_training_command(self.make_args())
        self.assertEqual(
            command[command.index('--nbs-rank-budget') + 1], '1536'
        )
        self.assertEqual(command[command.index('--seed') + 1], '1')
        self.assertEqual(
            command[command.index('--token-selector') + 1], 'none'
        )
        self.assertEqual(
            command[command.index('--speculative-draft-steps') + 1], '0'
        )

    def test_nbs_and_official_commands_share_test_conditions(self):
        args = self.make_args()
        nbs = sweep.build_nbs_test_command(args, Path('checkpoint'))
        official = sweep.build_official_test_command(args)
        for option in ('--seed', '--trace', '--trace-num', '--video', '--device'):
            self.assertEqual(
                nbs[nbs.index(option) + 1], official[official.index(option) + 1]
            )
        self.assertEqual(nbs[nbs.index('--rank') + 1], '32')
        self.assertEqual(official[official.index('--rank') + 1], '128')
        self.assertIn('--nbs-v19', nbs)
        self.assertNotIn('--nbs-v19', official)

    def test_results_are_durable_and_resume_signature_is_checked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'sweep.csv'
            rows = [{
                'experiment': 'nbs_only', 'mean_reward': 1.0,
                'inference_latency_mean_ms': 10.0,
            }]
            signature = {'rank_budget': 1536}
            sweep.write_results(rows, output, signature)
            self.assertTrue(output.is_file())
            self.assertEqual(sweep.load_resume(output, signature)[0][
                'experiment'
            ], 'nbs_only')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                sweep.load_resume(output, {'rank_budget': 512})


if __name__ == '__main__':
    unittest.main()
