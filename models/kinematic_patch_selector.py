"""Training-free viewport-motion patch selector."""
import torch
from torch import nn


class KinematicPatchSelector(nn.Module):
    """Score equirectangular patches from extrapolated yaw/pitch motion."""

    def __init__(
            self, grid_rows=4, grid_cols=4, velocity_window=3,
            horizon_scale=1.0, acceleration_weight=0.5,
            uncertainty_deg=20.0, uncertainty_growth=1.0,
            horizontal_fov_deg=110.0, vertical_fov_deg=90.0,
            prediction_points=5, future_steps=20):
        super().__init__()
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError('patch grid dimensions must be positive')
        if (velocity_window <= 0 or horizon_scale <= 0
                or prediction_points <= 0 or future_steps <= 0):
            raise ValueError('window, horizon, and prediction points must be positive')
        if uncertainty_deg < 0 or uncertainty_growth < 0:
            raise ValueError('uncertainty values must be non-negative')
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.num_patches = self.grid_rows * self.grid_cols
        self.velocity_window = int(velocity_window)
        self.horizon_scale = float(horizon_scale)
        self.acceleration_weight = float(acceleration_weight)
        self.uncertainty_deg = float(uncertainty_deg)
        self.uncertainty_growth = float(uncertainty_growth)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.prediction_points = int(prediction_points)
        self.future_steps = int(future_steps)

        rows = torch.arange(self.grid_rows, dtype=torch.float32)
        cols = torch.arange(self.grid_cols, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
        self.register_buffer(
            'patch_pitch_deg',
            90.0 - (row_grid.reshape(-1) + 0.5) * (180.0 / self.grid_rows),
            persistent=False,
        )
        self.register_buffer(
            'patch_yaw_deg',
            -180.0 + (col_grid.reshape(-1) + 0.5) * (360.0 / self.grid_cols),
            persistent=False,
        )

    @staticmethod
    def _wrapped_difference_degrees(values):
        difference = values[:, 1:] - values[:, :-1]
        return torch.remainder(difference + 180.0, 360.0) - 180.0

    def _motion(self, values, circular):
        differences = (
            self._wrapped_difference_degrees(values)
            if circular else values[:, 1:] - values[:, :-1]
        )
        if differences.shape[1] == 0:
            zeros = torch.zeros_like(values[:, -1])
            return zeros, zeros
        recent = differences[:, -min(self.velocity_window, differences.shape[1]):]
        velocity = recent.median(dim=1).values
        acceleration = (
            (recent[:, 1:] - recent[:, :-1]).median(dim=1).values
            if recent.shape[1] >= 2 else torch.zeros_like(velocity)
        )
        return velocity, acceleration

    def forward(self, history_viewports):
        if history_viewports.ndim != 3 or history_viewports.shape[-1] != 3:
            raise ValueError('history_viewports must have shape (B, T, 3)')
        pitch = history_viewports[..., 1].float() * 90.0
        yaw = history_viewports[..., 2].float() * 180.0
        yaw_velocity, yaw_acceleration = self._motion(yaw, circular=True)
        pitch_velocity, pitch_acceleration = self._motion(pitch, circular=False)

        fractions = torch.linspace(
            0.0, 1.0, self.prediction_points + 1,
            device=history_viewports.device, dtype=torch.float32,
        )
        steps = fractions * self.horizon_scale * float(self.future_steps)
        yaw_future = (
            yaw[:, -1:] + yaw_velocity[:, None] * steps[None, :]
            + 0.5 * self.acceleration_weight * yaw_acceleration[:, None]
            * steps[None, :].square()
        )
        yaw_future = torch.remainder(yaw_future + 180.0, 360.0) - 180.0
        pitch_future = (
            pitch[:, -1:] + pitch_velocity[:, None] * steps[None, :]
            + 0.5 * self.acceleration_weight * pitch_acceleration[:, None]
            * steps[None, :].square()
        ).clamp(-90.0, 90.0)

        yaw_distance = torch.remainder(
            self.patch_yaw_deg[None, None, :] - yaw_future[:, :, None] + 180.0,
            360.0,
        ) - 180.0
        pitch_distance = (
            self.patch_pitch_deg[None, None, :] - pitch_future[:, :, None]
        )
        uncertainty = self.uncertainty_deg * (
            1.0 + self.uncertainty_growth * fractions
        )
        yaw_scale = self.horizontal_fov_deg / 2.0 + uncertainty
        pitch_scale = self.vertical_fov_deg / 2.0 + uncertainty
        distance = (
            yaw_distance / yaw_scale[None, :, None]
        ).square() + (
            pitch_distance / pitch_scale[None, :, None]
        ).square()
        return -distance.min(dim=1).values

    def select_patches(self, logits, top_k=None, threshold=None):
        if top_k is None:
            raise ValueError('kinematic selector requires fixed top_k selection')
        top_k = min(int(top_k), logits.shape[1])
        indices = torch.topk(logits, top_k, dim=1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask

    def configuration(self):
        return {
            'type': 'kinematic',
            'velocity_window': self.velocity_window,
            'horizon_scale': self.horizon_scale,
            'acceleration_weight': self.acceleration_weight,
            'uncertainty_deg': self.uncertainty_deg,
            'uncertainty_growth': self.uncertainty_growth,
            'horizontal_fov_deg': self.horizontal_fov_deg,
            'vertical_fov_deg': self.vertical_fov_deg,
            'prediction_points': self.prediction_points,
            'future_steps': self.future_steps,
        }
