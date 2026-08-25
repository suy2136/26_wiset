import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'run_event_temporal_selector_sweep.py'
SPEC = importlib.util.spec_from_file_location('event_temporal_sweep', SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def args():
    return argparse.Namespace(
        checkpoint_dir=Path('checkpoint'), base_model_dir=Path('base'),
        exp_pool_path=Path('pool.pkl'), rank_budget=1536, physical_rank=32,
        rank_config=Path('configs/nbs_v19_rank_config.json'),
        event_min_spacing=2, throughput_threshold=0.60,
        buffer_threshold=6.0, bitrate_jump_threshold=1,
        trace='fcc-test', trace_num=100, video='video1', device='cuda:0',
    )


class EventTemporalSelectorSweepTest(unittest.TestCase):
    def test_default_matrix_has_baseline_and_k1_through_k4(self):
        experiments = list(sweep.configurations([1, 2, 3, 4]))
        self.assertEqual(
            [item['experiment'] for item in experiments],
            ['nbs_only', 'event_temporal_k1', 'event_temporal_k2',
             'event_temporal_k3', 'event_temporal_k4'],
        )

    def test_commands_share_checkpoint_and_disable_speculative(self):
        values = args()
        for experiment in sweep.configurations([1, 2, 3, 4]):
            command = sweep.build_command(values, experiment)
            self.assertEqual(command[command.index('--seed') + 1], '1')
            self.assertEqual(command[command.index('--trace') + 1], 'fcc-test')
            self.assertEqual(command[command.index('--trace-num') + 1], '100')
            self.assertEqual(
                command[command.index('--model-dir') + 1],
                str(values.checkpoint_dir.resolve()),
            )
            self.assertEqual(
                command[command.index('--speculative-draft-steps') + 1], '0'
            )
        event_command = sweep.build_command(
            values, list(sweep.configurations([3]))[1]
        )
        self.assertEqual(
            event_command[event_command.index('--token-selector') + 1],
            'event-aware',
        )
        self.assertEqual(
            event_command[event_command.index('--event-max-events') + 1], '3'
        )
        self.assertNotIn('--selector-history-steps', event_command)

    def test_checkpoint_must_match_seed_and_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            for filename in (
                'adapter_config.json', 'adapter_model.bin',
                'modules_except_plm.bin', 'nash_rank_allocator.pt',
            ):
                (checkpoint / filename).touch()
            (checkpoint / 'checkpoint_metadata.json').write_text(
                json.dumps({
                    'variant': 'nbs_v19', 'seed': 1,
                    'effective_rank_budget': 1536,
                }),
                encoding='utf-8',
            )
            sweep.validate_checkpoint(checkpoint, 1536)
            with self.assertRaisesRegex(ValueError, 'rank budget'):
                sweep.validate_checkpoint(checkpoint, 512)

    def test_results_include_nbs_comparisons(self):
        rows = [
            {'experiment': 'nbs_only', 'mean_reward': 1.0,
             'inference_latency_mean_ms': 20.0},
            {'experiment': 'event_temporal_k2', 'mean_reward': 0.99,
             'inference_latency_mean_ms': 10.0},
        ]
        sweep.add_baseline_comparisons(rows)
        self.assertAlmostEqual(rows[1]['mean_reward_delta_vs_nbs'], -0.01)
        self.assertEqual(rows[1]['inference_speedup_vs_nbs'], 2.0)
        self.assertEqual(rows[1]['inference_latency_reduction_vs_nbs'], 0.5)


if __name__ == '__main__':
    unittest.main()
