import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'run_nbs_v19_three_module_ablation.py'
SPEC = importlib.util.spec_from_file_location('three_module_ablation', SCRIPT)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def args():
    return argparse.Namespace(
        checkpoint_dir=Path('checkpoint'), base_model_dir=Path('base'),
        exp_pool_path=Path('pool.pkl'), rank_budget=1536, physical_rank=32,
        rank_config=Path('configs/nbs_v19_rank_config.json'),
        event_min_spacing=2, throughput_threshold=0.60,
        buffer_threshold=6.0, bitrate_jump_threshold=1,
        trace='fcc-test', trace_num=100, video='video1', device='cuda:0',
    )


class ThreeModuleAblationTest(unittest.TestCase):
    def test_matrix_has_only_three_variants_per_selector(self):
        counts = {}
        for experiment in ablation.EXPERIMENTS:
            counts[experiment['family']] = counts.get(experiment['family'], 0) + 1
        self.assertEqual(
            counts,
            {'temporal': 3, 'token': 3},
        )
        self.assertNotIn('nbs_only', {item['name'] for item in ablation.EXPERIMENTS})

    def test_each_family_isolated_and_uses_same_checkpoint(self):
        values = args()
        for experiment in ablation.EXPERIMENTS:
            command = ablation.build_command(values, experiment)
            family = experiment['family']
            self.assertEqual(command[command.index('--seed') + 1], '1')
            self.assertEqual(command[command.index('--trace-num') + 1], '100')
            self.assertEqual(
                command[command.index('--model-dir') + 1],
                str(values.checkpoint_dir.resolve()),
            )
            self.assertEqual(
                command[command.index('--temporal-selector') + 1],
                'event-aware' if family == 'temporal' else 'none',
            )
            self.assertEqual(
                command[command.index('--token-selector') + 1],
                'recent-timestep' if family == 'token' else 'none',
            )
            draft_steps = int(
                command[command.index('--speculative-draft-steps') + 1]
            )
            self.assertEqual(draft_steps, 0)

    def test_checkpoint_validation_enforces_budget(self):
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
            ablation.validate_checkpoint(checkpoint, 1536)
            with self.assertRaisesRegex(ValueError, 'rank budget'):
                ablation.validate_checkpoint(checkpoint, 512)

    def test_baseline_comparisons_include_quality_and_latency(self):
        rows = [
            {'experiment': 'nbs_only', 'mean_reward': 0.8,
             'inference_latency_mean_ms': 80.0},
            {'experiment': 'temporal_k3', 'mean_reward': 0.84,
             'inference_latency_mean_ms': 60.0},
        ]
        ablation.add_baseline_comparisons(rows)
        self.assertAlmostEqual(rows[1]['mean_reward_change_ratio_vs_nbs'], 0.05)
        self.assertAlmostEqual(
            rows[1]['inference_latency_reduction_vs_nbs'], 0.25
        )

    def test_discovers_matching_existing_nbs_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = (
                Path(temp_dir) / 'fcc-test_video1' /
                'trace_num_100_fixed_True' /
                'rank_32_nbs_v19_budget1536_seed_1' /
                'selector_none' / 'speculative_none' /
                'selector_metrics.json'
            )
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps({
                'selector': 'none', 'speculative_draft_steps': 0,
                'mean_reward': 0.8, 'inference_latency_mean_ms': 80.0,
            }), encoding='utf-8')
            found, metrics = ablation.discover_baseline_metrics(
                args(), results_root=Path(temp_dir)
            )
            self.assertEqual(found, metrics_path.resolve())
            self.assertEqual(metrics['mean_reward'], 0.8)

    def test_explicit_baseline_rejects_selector_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / 'selector_metrics.json'
            metrics_path.write_text(json.dumps({
                'selector': 'recent-timestep',
                'speculative_draft_steps': 0,
                'mean_reward': 0.8,
                'inference_latency_mean_ms': 80.0,
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'selector=none'):
                ablation.discover_baseline_metrics(
                    args(), explicit_path=metrics_path
                )

    def test_resume_reuses_only_an_identical_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'ablation.csv'
            rows = [{'experiment': 'nbs_only', 'family': 'baseline'}]
            ablation.write_results(rows, output, {'trace': 'fcc-test'})
            self.assertEqual(
                ablation.load_resume_rows(output, {'trace': 'fcc-test'}), rows
            )
            with self.assertRaisesRegex(ValueError, 'does not match'):
                ablation.load_resume_rows(output, {'trace': 'fcc-valid'})


if __name__ == '__main__':
    unittest.main()
