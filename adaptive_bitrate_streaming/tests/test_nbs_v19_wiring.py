import json
from pathlib import Path
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]


class NBSV19WiringTest(unittest.TestCase):
    def test_rank_config_encodes_v19_bounds(self):
        path = ABR_ROOT / 'configs' / 'nbs_v19_rank_config.json'
        with path.open(encoding='utf-8') as stream:
            config = json.load(stream)
        self.assertEqual(set(config), {
            '*.layers.*.self_attn.q_proj',
            '*.layers.*.self_attn.v_proj',
        })
        self.assertTrue(all(
            bounds == {'min_rank': 2, 'max_rank': 32}
            for bounds in config.values()
        ))

    def test_equal_capacity_rank_config_and_runner(self):
        config_path = (
            ABR_ROOT / 'configs' / 'nbs_v19_rank_config_max256.json'
        )
        with config_path.open(encoding='utf-8') as stream:
            config = json.load(stream)
        self.assertTrue(all(
            bounds == {'min_rank': 2, 'max_rank': 256}
            for bounds in config.values()
        ))
        source = (
            ABR_ROOT / 'analysis' / 'run_nbs_v19_budget8192_experiment.py'
        ).read_text(encoding='utf-8')
        for setting in (
            'RANK_BUDGET = 8192', 'PHYSICAL_RANK = 256',
            'nbs_v19_rank_config_max256.json',
        ):
            self.assertIn(setting, source)

    def test_v19_accepts_configurable_physical_rank_with_legacy_shadow(self):
        low_rank_source = (
            ABR_ROOT / 'plm_special' / 'models' / 'low_rank.py'
        ).read_text(encoding='utf-8')
        run_source = (ABR_ROOT / 'run_plm.py').read_text(encoding='utf-8')
        self.assertNotIn("if rank != 32:", low_rank_source)
        self.assertNotIn("if args.rank != 32:", run_source)
        self.assertIn("if args.rank <= 0:", run_source)
        self.assertIn("init_r=rank", low_rank_source)
        self.assertNotIn("init_r=32", low_rank_source)
        self.assertIn("shadow_update_policy='legacy'", low_rank_source)

    def test_training_order_matches_v19_gradient_definition(self):
        source = (ABR_ROOT / 'plm_special' / 'trainer.py').read_text(
            encoding='utf-8'
        )
        start = source.index('if should_update:')
        end = source.index('if self.lr_scheduler is not None:', start)
        update_block = source[start:end]
        self.assertLess(
            update_block.index('update_sensitivity()'),
            update_block.index('clip_grad_norm_'),
        )
        self.assertLess(
            update_block.index('clip_grad_norm_'),
            update_block.index('self.optimizer.step()'),
        )
        self.assertLess(
            update_block.index('self.optimizer.step()'),
            update_block.index('self.nbs_allocator.allocate'),
        )
        self.assertLess(
            update_block.index('self.nbs_allocator.allocate'),
            update_block.index('self.optimizer.zero_grad'),
        )

    def test_checkpoint_saves_and_restores_allocator_state(self):
        source = (ABR_ROOT / 'run_plm.py').read_text(encoding='utf-8')
        self.assertGreaterEqual(source.count('nash_rank_allocator.pt'), 2)
        self.assertIn('allocator.state_dict()', source)
        self.assertIn('allocator.load_state_dict(allocator_state)', source)

    def test_training_launcher_fixes_single_training_conditions(self):
        source = (
            ABR_ROOT / 'scripts' / 'run_nbs_v19_training.sh'
        ).read_text(encoding='utf-8')
        for argument in (
            '--adapt --nbs-v19 --fp16', '--seed 1', '--rank 32',
            '--nbs-rank-budget 512', '--token-selector none',
            '--speculative-draft-steps 0', '--device cuda:0',
        ):
            self.assertIn(argument, source)

    def test_numeric_guards_cover_forward_gradient_and_allocator(self):
        trainer_source = (
            ABR_ROOT / 'plm_special' / 'trainer.py'
        ).read_text(encoding='utf-8')
        low_rank_source = (
            ABR_ROOT / 'plm_special' / 'models' / 'low_rank.py'
        ).read_text(encoding='utf-8')
        for guard in (
            '_adalora_delta_issues', '_gradient_issues',
            '_parameter_issues', '_allocator_numeric_issues',
            'error_if_nonfinite=True',
            'nbs_numeric_events.jsonl',
        ):
            source = trainer_source if guard != 'nbs_numeric_events.jsonl' else (
                ABR_ROOT / 'run_plm.py'
            ).read_text(encoding='utf-8')
            self.assertIn(guard, source)
        self.assertIn('result_fp32 = base_result.float()', low_rank_source)
        self.assertIn('_nbs_last_precast_absmax', low_rank_source)

    def test_rank_reallocation_resets_adam_moments(self):
        source = (ABR_ROOT / 'plm_special' / 'trainer.py').read_text(
            encoding='utf-8'
        )
        self.assertIn('_reset_reallocated_optimizer_moments', source)
        self.assertIn("('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')", source)
        allocate = source.index('self.nbs_allocator.allocate')
        reset = source.index('_reset_reallocated_optimizer_moments', allocate)
        self.assertLess(allocate, reset)

    def test_rank_diagnostics_export_nash_analysis_fields(self):
        allocator_source = (
            ABR_ROOT.parent / 'models' / 'rank_allocator.py'
        ).read_text(encoding='utf-8')
        trainer_source = (ABR_ROOT / 'plm_special' / 'trainer.py').read_text(
            encoding='utf-8'
        )
        for field in (
            '"rank"', '"utility"', '"next_marginal_gain"',
            '"spectral_energy_total"', '"sensitivity"',
        ):
            self.assertIn(field, allocator_source)
        self.assertIn("'bargaining_weight': row.get('alpha')", trainer_source)

    def test_abr_metrics_export_qoe_components_and_latency_percentiles(self):
        source = (ABR_ROOT / 'plm_special' / 'test.py').read_text(
            encoding='utf-8'
        )
        for field in (
            'qoe_raw_mean', 'mean_bitrate_mbps', 'total_rebuffer_s',
            'mean_rebuffer_s_per_chunk', 'mean_smoothness_mbps',
            'inference_latency_p50_ms', 'inference_latency_p95_ms',
        ):
            self.assertIn(field, source)

    def test_inference_nonfinite_is_guarded_before_probability_sampling(self):
        policy_source = (
            ABR_ROOT / 'plm_special' / 'models' / 'rl_policy.py'
        ).read_text(encoding='utf-8')
        sample_start = policy_source.index('    def _sample(self, logits):')
        sample_block = policy_source[sample_start:]
        self.assertLess(
            sample_block.index("_require_finite(logits.float()"),
            sample_block.index('F.softmax'),
        )
        self.assertLess(
            sample_block.index("'sampling_probabilities'"),
            sample_block.index('random.choices'),
        )
        self.assertIn("outputs['last_hidden_state'].float()", policy_source)
        self.assertIn('hidden + plm_inputs.float()', policy_source)

    def test_invalid_validation_cannot_replace_best_checkpoint(self):
        run_source = (ABR_ROOT / 'run_plm.py').read_text(encoding='utf-8')
        start = run_source.index("if not eval_logs.get('evaluation_valid', True)")
        end = run_source.index("episodes_return = eval_logs['episodes_return']", start)
        invalid_block = run_source[start:end]
        self.assertIn('record_invalid_validation', invalid_block)
        self.assertIn('continue', invalid_block)
        self.assertNotIn('save_model(', invalid_block)
        evaluate_source = (
            ABR_ROOT / 'plm_special' / 'evaluate.py'
        ).read_text(encoding='utf-8')
        self.assertIn("'evaluation_valid': False", evaluate_source)
        self.assertIn("'nonfinite_timestep': int(timestep)", evaluate_source)


if __name__ == '__main__':
    unittest.main()
