import unittest

import torch

from plm_special.models.low_rank import _mixed_precision_adalora_forward
from plm_special.trainer import Trainer, ensure_trainable_parameters_fp32


class FakeAllocator:
    def __init__(self):
        self.value = torch.tensor([1.0, 2.0])
        self.last_diagnostics = [{'rank': 2}]

    def state_dict(self):
        return {'value': self.value.clone()}

    def load_state_dict(self, state):
        self.value.copy_(state['value'])


class MixedDtypeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3).half()
        self.adapter = torch.nn.Linear(3, 2).half()
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


class NBSOptimizerTransactionTest(unittest.TestCase):
    def test_fp16_projection_overflow_is_contained_and_flagged(self):
        class Projection:
            weight = torch.zeros((1, 1), dtype=torch.float16)
            disable_adapters = False
            merged = False
            active_adapters = ['default']
            lora_A = {'default': torch.ones((1, 1), dtype=torch.float32)}
            lora_B = {'default': torch.tensor([[70000.0]])}
            lora_E = {'default': torch.ones((1, 1), dtype=torch.float32)}
            lora_dropout = {'default': torch.nn.Identity()}
            ranknum = {'default': torch.tensor(1.0 - 1e-5)}
            scaling = {'default': 1.0}

            @staticmethod
            def _linear(value):
                return torch.zeros_like(value, dtype=torch.float16)

        projection = Projection()
        output = _mixed_precision_adalora_forward(
            projection, torch.ones((1, 1), dtype=torch.float16)
        )

        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(float(output.abs().max()), 65504.0)
        self.assertGreater(
            float(projection._nbs_last_precast_absmax), 65504.0
        )
        self.assertTrue(bool(projection._nbs_last_precast_finite))

    def test_only_trainable_parameters_are_promoted_to_fp32(self):
        model = MixedDtypeModel()

        summary = ensure_trainable_parameters_fp32(model)

        self.assertEqual(summary['promoted_tensor_count'], 2)
        self.assertTrue(all(
            parameter.dtype == torch.float16
            for parameter in model.frozen.parameters()
        ))
        self.assertTrue(all(
            parameter.dtype == torch.float32
            for parameter in model.adapter.parameters()
        ))

    @staticmethod
    def _make_transaction_trainer():
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.nbs_allocator = FakeAllocator()
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )
        trainer.grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        trainer.optimizer_step = 7
        trainer.optimizer_state_dtype_verified = False
        trainer.rollback_backup_device = 'cpu'
        trainer.max_rollback_backup_mib = 2048.0
        trainer.update_ratio_warning = 0.01
        trainer.max_update_ratio = 0.05
        trainer.update_ratio_floor = 0.01
        trainer.max_update_rms = 0.01
        return trainer, model, optimizer

    def test_transaction_memory_limit_fails_before_backup(self):
        trainer, model, _ = self._make_transaction_trainer()
        trainer.max_rollback_backup_mib = 1e-9
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with self.assertRaises(MemoryError):
            trainer._snapshot_optimizer_transaction()

    def test_optimizer_transaction_restores_parameters_and_adam_state(self):
        trainer, model, optimizer = self._make_transaction_trainer()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 2.0)
        snapshot = trainer._snapshot_optimizer_transaction()
        self.assertTrue(all(
            previous.device.type == 'cpu'
            for previous in snapshot['parameters'].values()
        ))
        optimizer.step()
        first_parameter = next(model.parameters())
        first_parameter.data.fill_(float('nan'))
        optimizer.state[first_parameter]['exp_avg'].fill_(float('inf'))
        trainer.nbs_allocator.value.fill_(99.0)
        trainer.optimizer_step = 8

        self.assertTrue(trainer._parameter_issues())
        self.assertTrue(trainer._optimizer_state_issues())
        trainer._restore_optimizer_transaction(snapshot)

        self.assertFalse(trainer._parameter_issues())
        self.assertFalse(trainer._optimizer_state_issues(check_dtype=True))
        self.assertTrue(torch.equal(
            trainer.nbs_allocator.value,
            snapshot['allocator_state']['value'],
        ))
        self.assertEqual(trainer.optimizer_step, 7)
        for parameter, previous in snapshot['parameters'].items():
            self.assertTrue(torch.equal(parameter, previous))
        for parameter, previous_state in snapshot['optimizer_states'].items():
            for name, previous in previous_state.items():
                current = optimizer.state[parameter][name]
                if isinstance(previous, dict) and 'tensor' in previous:
                    self.assertTrue(torch.equal(
                        current,
                        previous['tensor'].to(
                            device=current.device, dtype=current.dtype
                        ),
                    ))
                else:
                    self.assertEqual(current, previous)

    def test_eva_transaction_restores_without_nbs_allocator(self):
        trainer, model, optimizer = self._make_transaction_trainer()
        trainer.nbs_allocator = None
        trainer.lora_method = 'eva'
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 2.0)
        snapshot = trainer._snapshot_optimizer_transaction()
        optimizer.step()
        first_parameter = next(model.parameters())
        first_parameter.data.fill_(float('nan'))
        optimizer.state[first_parameter]['exp_avg'].fill_(float('inf'))

        trainer._restore_optimizer_transaction(snapshot)

        self.assertIsNone(snapshot['allocator_state'])
        self.assertFalse(trainer._parameter_issues())
        self.assertFalse(trainer._optimizer_state_issues(check_dtype=True))
        for parameter, previous in snapshot['parameters'].items():
            self.assertTrue(torch.equal(parameter, previous))

    def test_update_ratio_rejects_large_finite_update(self):
        trainer, model, optimizer = self._make_transaction_trainer()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        snapshot = trainer._snapshot_optimizer_transaction()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.001)

        diagnostics = trainer._update_ratio_diagnostics(snapshot)

        self.assertTrue(diagnostics['issues'])
        self.assertGreater(diagnostics['max_update_ratio'], 0.05)

    def test_optimizer_state_dtype_guard_detects_non_fp32_state(self):
        trainer, model, optimizer = self._make_transaction_trainer()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        self.assertFalse(trainer._optimizer_state_issues(check_dtype=True))
        parameter = next(model.parameters())
        optimizer.state[parameter]['exp_avg'] = (
            optimizer.state[parameter]['exp_avg'].half()
        )
        issues = trainer._optimizer_state_issues(check_dtype=True)
        self.assertTrue(any(
            issue['reason'] == 'non_fp32' for issue in issues
        ))


if __name__ == '__main__':
    unittest.main()
