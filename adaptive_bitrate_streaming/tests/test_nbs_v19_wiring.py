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


if __name__ == '__main__':
    unittest.main()
