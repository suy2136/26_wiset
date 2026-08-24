import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'run_nbs_v19_full_experiment.py'
SPEC = importlib.util.spec_from_file_location('nbs_v19_full', SCRIPT)
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


class NBSV19FullExperimentTest(unittest.TestCase):
    def make_args(self):
        return full.parse_args([
            '--base-model-dir', 'base', '--exp-pool-path', 'pool.pkl',
            '--num-epochs', '4', '--eval-per-epoch', '1',
        ])

    def test_training_command_fixes_nbs_and_disables_inference_features(self):
        command = full.build_training_command(self.make_args())
        self.assertIn('--nbs-v19', command)
        self.assertIn('--fp16', command)
        self.assertEqual(command[command.index('--seed') + 1], '1')
        self.assertEqual(command[command.index('--rank') + 1], '32')
        self.assertEqual(
            command[command.index('--token-selector') + 1], 'none'
        )
        self.assertEqual(
            command[command.index('--speculative-draft-steps') + 1], '0'
        )

    def test_inference_command_runs_matrix_and_can_resume(self):
        args = self.make_args()
        args.resume_inference = True
        command = full.build_inference_command(args, Path('checkpoint'))
        self.assertIn('analysis/run_nbs_v19_inference_matrix.py', command)
        self.assertIn('--resume', command)

    def test_checkpoint_discovery_prefers_best_over_newer_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            best = root / 'best'
            final = root / 'final'
            best.mkdir()
            final.mkdir()
            for directory, role in ((best, 'best'), (final, 'final')):
                with (directory / 'checkpoint_metadata.json').open(
                    'w', encoding='utf-8'
                ) as stream:
                    json.dump({
                        'variant': 'nbs_v19', 'seed': 1, 'role': role,
                        'effective_rank_budget': 512,
                    }, stream)
                time.sleep(0.01)
            selected = full.discover_new_checkpoint(root, time.time() - 10)
            self.assertEqual(selected, best)


if __name__ == '__main__':
    unittest.main()
