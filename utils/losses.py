import torch
import torch.nn as nn


def wrap_normalized_angle_error(error):
    """Wrap an angle error normalized by 180 degrees into [-1, 1)."""
    return torch.remainder(error + 1.0, 2.0) - 1.0


class CircularViewportMSELoss(nn.Module):
    """MSE for normalized (roll, pitch, yaw), wrapping circular axes.

    Roll and yaw are normalized by 180 degrees and represent circular angles,
    while pitch is normalized by 90 degrees and remains a linear coordinate.
    """

    def forward(self, prediction, target):
        error = prediction - target
        circular_error = torch.stack(
            (
                wrap_normalized_angle_error(error[..., 0]),
                error[..., 1],
                wrap_normalized_angle_error(error[..., 2]),
            ),
            dim=-1,
        )
        return circular_error.square().mean()
