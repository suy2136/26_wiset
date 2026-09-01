import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ABR_ROOT = Path(__file__).resolve().parents[1]
if str(ABR_ROOT) not in sys.path:
    sys.path.insert(0, str(ABR_ROOT))

from plm_special.utils.utils import process_batch


class ProcessBatchTest(unittest.TestCase):
    @staticmethod
    def _batch(states):
        return (
            states,
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            np.asarray([0, 1, 2], dtype=np.int32),
        )

    def test_numpy_and_tensor_states_produce_identical_batches(self):
        numpy_states = [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0]], dtype=np.float32),
        ]
        tensor_states = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ]

        numpy_batch = process_batch(self._batch(numpy_states))
        tensor_batch = process_batch(self._batch(tensor_states))

        for numpy_value, tensor_value in zip(numpy_batch, tensor_batch):
            self.assertTrue(torch.equal(numpy_value, tensor_value))
        self.assertEqual(tuple(numpy_batch[0].shape), (1, 3, 2))
        self.assertEqual(tuple(numpy_batch[4].shape), (1, 3))


if __name__ == '__main__':
    unittest.main()
