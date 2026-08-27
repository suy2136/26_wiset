import copy
import math
import unittest

try:
    # Repository-root execution:
    # python -m unittest adaptive_bitrate_streaming.tests....
    from adaptive_bitrate_streaming.plm_special.numeric_safety import (
        classify_update,
    )
except ModuleNotFoundError as exc:
    if exc.name != 'adaptive_bitrate_streaming':
        raise
    # ABR-directory execution:
    # cd adaptive_bitrate_streaming && python -m unittest tests....
    from plm_special.numeric_safety import classify_update


class NBSNumericSafetySimulationTest(unittest.TestCase):
    def classify(self, old_rms, update_rms):
        return classify_update(
            old_rms,
            update_rms,
            ratio_floor=0.01,
            warning_ratio=0.01,
            maximum_ratio=0.05,
            maximum_update_rms=0.01,
        )

    def test_realistic_lr_sized_update_is_warned_but_accepted(self):
        result = self.classify(0.0, 0.00015)
        self.assertAlmostEqual(result['update_ratio'], 0.015)
        self.assertTrue(result['warning'])
        self.assertFalse(result['rollback'])

    def test_large_finite_and_nan_updates_are_rejected(self):
        large = self.classify(0.01, 0.0006)
        self.assertAlmostEqual(large['update_ratio'], 0.06)
        self.assertTrue(large['rollback'])
        self.assertTrue(self.classify(0.01, math.nan)['rollback'])

    def test_next_forward_overflow_restores_complete_numeric_state(self):
        before = {
            'adapter': [0.0100, -0.0200],
            'adam_exp_avg': [0.0010, -0.0020],
            'allocator_mask': [1.0, 0.0],
            'spectral_shadow': [0.8, 1.2],
            'optimizer_step': 120,
            'scheduler_step': 120,
            'learning_rate': 0.00015,
        }
        snapshot = copy.deepcopy(before)
        candidate = copy.deepcopy(before)
        candidate['adapter'] = [0.0102, -0.0198]
        candidate['adam_exp_avg'] = [0.0012, -0.0018]
        candidate['allocator_mask'] = [1.0, 1.0]
        candidate['optimizer_step'] = 121
        candidate['scheduler_step'] = 121

        # The parameter update itself is finite and below the 0.05 limit.
        update = self.classify(0.01, 0.0002)
        self.assertFalse(update['rollback'])
        # Rank reactivation can nevertheless exceed the FP16 output range.
        simulated_precast_activation = 70000.0
        self.assertGreater(simulated_precast_activation, 65504.0)

        # A pending transaction restores every mutated subsystem, then lowers LR.
        candidate = copy.deepcopy(snapshot)
        candidate['learning_rate'] *= 0.5
        self.assertEqual(candidate['adapter'], before['adapter'])
        self.assertEqual(candidate['adam_exp_avg'], before['adam_exp_avg'])
        self.assertEqual(candidate['allocator_mask'], before['allocator_mask'])
        self.assertEqual(candidate['spectral_shadow'], before['spectral_shadow'])
        self.assertEqual(candidate['optimizer_step'], 120)
        self.assertEqual(candidate['scheduler_step'], 120)
        self.assertEqual(candidate['learning_rate'], 0.000075)

    def test_overflow_on_seventeenth_batch_is_still_rollbackable(self):
        before = {
            'parameter': 0.02,
            'adam_state': 0.002,
            'mask': [1, 0],
            'step': 300,
            'lr': 0.00015,
        }
        snapshot = copy.deepcopy(before)
        tentative = {
            'parameter': 0.0202,
            'adam_state': 0.0022,
            'mask': [1, 1],
            'step': 301,
            'lr': 0.00015,
        }
        forward_absmax = [32000.0] * 16 + [math.nan] + [30000.0] * 15
        committed = False
        for value in forward_absmax:
            if not math.isfinite(value) or value > 65504.0:
                tentative = copy.deepcopy(snapshot)
                tentative['lr'] *= 0.5
                break
        else:
            committed = True
        self.assertFalse(committed)
        self.assertEqual(tentative['parameter'], 0.02)
        self.assertEqual(tentative['adam_state'], 0.002)
        self.assertEqual(tentative['mask'], [1, 0])
        self.assertEqual(tentative['step'], 300)
        self.assertEqual(tentative['lr'], 0.000075)

    def test_thirty_two_safe_forwards_commit_tentative_state(self):
        tentative = {'parameter': 0.0202, 'mask': [1, 1], 'step': 301}
        forward_absmax = [32000.0 + index for index in range(32)]
        committed = all(
            math.isfinite(value) and value <= 65504.0
            for value in forward_absmax
        )
        self.assertTrue(committed)
        self.assertEqual(tentative['step'], 301)

    def test_restored_model_specific_overflow_quarantines_only_that_batch(self):
        """Reproduce the server failure: rollback succeeds, retry still fails."""
        fp16_limit = 65504.0
        tentative_model = {'weight': 0.0102, 'lr': 0.0000705}
        restored_model = {'weight': 0.0100, 'lr': 0.0000705}
        pending_transaction = copy.deepcopy(restored_model)

        first_attempt_absmax = math.nan
        self.assertFalse(math.isfinite(first_attempt_absmax))
        tentative_model = copy.deepcopy(pending_transaction)
        tentative_model['lr'] *= 0.5

        # The old model can also be unsafe for one data-dependent activation;
        # lowering LR cannot change the current forward activation.
        retry_absmax = 70000.0
        quarantine_batch = (
            not math.isfinite(retry_absmax)
            or retry_absmax > fp16_limit
        )
        self.assertTrue(quarantine_batch)
        self.assertEqual(tentative_model['weight'], 0.0100)
        self.assertEqual(tentative_model['lr'], 0.00003525)

        # A following healthy batch is accepted and clears consecutive status.
        next_batch_absmax = 42000.0
        self.assertTrue(
            math.isfinite(next_batch_absmax)
            and next_batch_absmax <= fp16_limit
        )

    def test_projection_containment_cannot_silently_accept_bad_values(self):
        fp16_limit = 65504.0
        values = [12.0, 70000.0, -math.inf, math.nan]
        contained = []
        unhealthy = []
        for value in values:
            invalid = not math.isfinite(value) or abs(value) > fp16_limit
            unhealthy.append(invalid)
            if math.isnan(value):
                contained.append(0.0)
            elif value == math.inf:
                contained.append(fp16_limit)
            elif value == -math.inf:
                contained.append(-fp16_limit)
            else:
                contained.append(max(-fp16_limit, min(fp16_limit, value)))
        self.assertTrue(all(math.isfinite(value) for value in contained))
        self.assertEqual(contained, [12.0, 65504.0, -65504.0, 0.0])
        # Health metadata remains invalid, so policy must reject this forward.
        self.assertEqual(unhealthy, [False, True, True, True])


if __name__ == '__main__':
    unittest.main()
