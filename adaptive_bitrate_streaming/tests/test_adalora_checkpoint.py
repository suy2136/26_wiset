import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ABR_ROOT = Path(__file__).resolve().parents[1]
if str(ABR_ROOT) not in sys.path:
    sys.path.insert(0, str(ABR_ROOT))

from plm_special.utils.adalora_checkpoint import (
    _physical_adapter_budget,
    load_resized_adalora_adapter,
)


class TinyProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        return self.q_proj(inputs)


class AdaLoraCheckpointTest(unittest.TestCase):
    def test_saved_rank_pattern_resizes_fresh_adapter_before_load(self):
        from peft import AdaLoraConfig, TaskType, get_peft_model

        def wrapped_model():
            return get_peft_model(
                TinyProjectionModel(),
                AdaLoraConfig(
                    init_r=4,
                    target_r=2,
                    target_modules=['q_proj'],
                    task_type=TaskType.FEATURE_EXTRACTION,
                    total_step=10,
                    tinit=1,
                    tfinal=2,
                    deltaT=1,
                ),
            )

        source = wrapped_model()
        rank_pattern = {
            'q_proj.lora_E.default': [True, True, False, False]
        }
        source.peft_config['default'].rank_pattern = rank_pattern

        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            restored = wrapped_model()
            report = load_resized_adalora_adapter(restored, directory)

        self.assertEqual(report['rank_budget'], 2)
        self.assertEqual(report['module_count'], 1)
        self.assertEqual(_physical_adapter_budget(restored), (2, 1))


if __name__ == '__main__':
    unittest.main()
