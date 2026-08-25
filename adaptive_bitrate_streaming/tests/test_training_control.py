import math
from pathlib import Path
import sys
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABR_ROOT))

from plm_special.training_control import ValidationPlateauController


class ValidationPlateauControllerTest(unittest.TestCase):
    def test_significant_improvement_resets_both_counters(self):
        controller = ValidationPlateauController(initial_metric=1.0)
        controller.update(1.001, 1)
        decision = controller.update(1.004, 2)
        self.assertTrue(decision.significant_improvement)
        self.assertEqual(decision.validations_without_improvement, 0)
        self.assertEqual(decision.validations_since_lr_reduction, 0)

    def test_lr_reduction_precedes_early_stopping(self):
        controller = ValidationPlateauController(
            initial_metric=1.0, lr_patience=2,
            early_stopping_patience=4, min_epochs=1,
        )
        decisions = [controller.update(1.0, epoch) for epoch in range(1, 5)]
        self.assertTrue(decisions[1].reduce_learning_rate)
        self.assertFalse(decisions[1].should_stop)
        self.assertTrue(decisions[3].reduce_learning_rate)
        self.assertTrue(decisions[3].should_stop)

    def test_minimum_epoch_prevents_early_stop(self):
        controller = ValidationPlateauController(
            initial_metric=1.0, lr_patience=0,
            early_stopping_patience=2, min_epochs=5,
        )
        for epoch in range(1, 5):
            self.assertFalse(controller.update(1.0, epoch).should_stop)
        self.assertTrue(controller.update(1.0, 5).should_stop)

    def test_invalid_metric_is_rejected(self):
        controller = ValidationPlateauController()
        with self.assertRaisesRegex(ValueError, "finite"):
            controller.update(math.nan, 1)


if __name__ == "__main__":
    unittest.main()
