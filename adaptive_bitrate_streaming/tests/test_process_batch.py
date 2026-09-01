import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ABR_ROOT = Path(__file__).resolve().parents[1]
if str(ABR_ROOT) not in sys.path:
    sys.path.insert(0, str(ABR_ROOT))

from plm_special.utils.utils import process_batch
from plm_special.models.state_encoder import EncoderNetwork


class ProcessBatchTest(unittest.TestCase):
    @staticmethod
    def _batch(states):
        return (
            states,
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0.1, 0.2], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int32),
        )

    def test_direct_numpy_and_dataloader_tensor_states_are_identical(self):
        numpy_states = [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        ]
        # This is the shape produced by DataLoader(batch_size=1): the
        # sequence remains a list and each timestep gains a batch axis.
        tensor_states = [
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        ]

        numpy_batch = process_batch(self._batch(numpy_states))
        tensor_batch = process_batch(self._batch(tensor_states))

        for numpy_value, tensor_value in zip(numpy_batch, tensor_batch):
            self.assertTrue(torch.equal(numpy_value, tensor_value))
        self.assertEqual(tuple(numpy_batch[0].shape), (1, 2, 2, 2))
        self.assertEqual(tuple(numpy_batch[4].shape), (1, 2))

    def test_direct_numpy_array_keeps_sequence_and_state_axes(self):
        states = np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )
        processed = process_batch(self._batch(states))
        self.assertEqual(tuple(processed[0].shape), (1, 2, 2, 2))

    def test_direct_shapley_sample_runs_through_state_encoder(self):
        states = [
            np.full((6, 6), timestep, dtype=np.float32)
            for timestep in range(20)
        ]
        batch = (
            states,
            np.zeros(20, dtype=np.int64),
            np.zeros(20, dtype=np.float32),
            np.arange(20, dtype=np.int32),
        )
        processed_states = process_batch(batch)[0]

        self.assertEqual(tuple(processed_states.shape), (1, 20, 6, 6))
        features = EncoderNetwork()(processed_states)
        self.assertEqual(len(features), 6)
        self.assertTrue(all(feature.shape[:2] == (1, 20) for feature in features))


if __name__ == '__main__':
    unittest.main()
