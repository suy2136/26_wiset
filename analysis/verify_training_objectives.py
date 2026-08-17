"""Regression checks for viewport loss and rotation-aware evaluation metrics."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.losses import CircularViewportMSELoss
from utils.metrics import compute_each_rmse, compute_rmse


def verify_circular_viewport_loss():
    prediction = torch.tensor([[[179.0 / 180.0, 0.25, -179.0 / 180.0]]], requires_grad=True)
    target = torch.tensor([[[-179.0 / 180.0, 0.25, 179.0 / 180.0]]])
    loss = CircularViewportMSELoss()(prediction, target)
    expected = torch.tensor(2.0 * (2.0 / 180.0) ** 2 / 3.0)
    torch.testing.assert_close(loss.detach(), expected, rtol=1e-5, atol=1e-8)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    print('[PASS] circular roll/yaw loss uses the shortest angular distance')


def verify_rotation_rmse():
    prediction = np.array([[[179.0]]])
    target = np.array([[[-179.0]]])
    assert np.isclose(compute_rmse(prediction, target, rotation=True), 2.0)
    np.testing.assert_allclose(
        compute_each_rmse(prediction, target, rotation=True), np.array([2.0])
    )
    assert np.isclose(compute_rmse(prediction, target, rotation=False), 358.0)
    print('[PASS] aggregate and per-sample RMSE honor rotation wraparound')


if __name__ == '__main__':
    verify_circular_viewport_loss()
    verify_rotation_rmse()
    print('All training-objective checks completed.')
