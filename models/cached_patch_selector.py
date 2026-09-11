"""Low-overhead viewport-guided selection over offline ViT patch features."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


POLICIES = ("k1", "adaptive", "k2")


class CachedViewportPatchSelector(nn.Module):
    """Select the current viewport cell and, optionally, one motion neighbour.

    The selector deliberately avoids learned layers, trajectory extrapolation and
    top-k.  It maps the latest yaw/pitch directly to a grid cell and uses only the
    last motion delta to choose a neighbouring cell.
    """

    def __init__(self, policy="adaptive", grid_rows=4, grid_cols=4,
                 motion_threshold_deg=12.0):
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy}")
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("patch grid dimensions must be positive")
        if motion_threshold_deg < 0:
            raise ValueError("motion threshold must be non-negative")
        self.policy = policy
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.motion_threshold_deg = float(motion_threshold_deg)

    @staticmethod
    def _wrapped_yaw_delta(current, previous):
        return torch.remainder(current - previous + 180.0, 360.0) - 180.0

    def select_indices(self, history_viewports):
        """Return selected row-major patch indices for a batch-size-one history."""
        if history_viewports.ndim != 3 or history_viewports.shape[-1] != 3:
            raise ValueError("history_viewports must have shape (B, T, 3)")
        if history_viewports.shape[0] != 1:
            raise ValueError("cached patch selection currently requires batch size 1")
        if history_viewports.shape[1] == 0:
            raise ValueError("history_viewports must contain at least one timestep")

        history = history_viewports.float()
        pitch = (history[0, -1, 1] * 90.0).clamp(-90.0, 90.0)
        yaw = torch.remainder(history[0, -1, 2] * 180.0 + 180.0, 360.0) - 180.0
        row = torch.floor((90.0 - pitch) / (180.0 / self.grid_rows)).long()
        col = torch.floor((yaw + 180.0) / (360.0 / self.grid_cols)).long()
        row = row.clamp(0, self.grid_rows - 1)
        col = torch.remainder(col, self.grid_cols)
        current = row * self.grid_cols + col

        if self.policy == "k1":
            return current.reshape(1)

        if history.shape[1] >= 2:
            previous_pitch = history[0, -2, 1] * 90.0
            previous_yaw = history[0, -2, 2] * 180.0
            delta_pitch = pitch - previous_pitch
            delta_yaw = self._wrapped_yaw_delta(yaw, previous_yaw)
        else:
            delta_pitch = torch.zeros_like(pitch)
            delta_yaw = torch.zeros_like(yaw)

        speed = torch.sqrt(delta_pitch.square() + delta_yaw.square())
        if (self.policy == "adaptive"
                and bool((speed < self.motion_threshold_deg).item())):
            return current.reshape(1)

        # Compare displacement in grid-cell units so horizontal and vertical
        # directions are treated consistently on non-square equirectangular grids.
        horizontal = delta_yaw.abs() / (360.0 / self.grid_cols)
        vertical = delta_pitch.abs() / (180.0 / self.grid_rows)
        horizontal_motion = horizontal >= vertical
        horizontal_direction = torch.where(
            delta_yaw >= 0, torch.ones_like(col), -torch.ones_like(col)
        )
        # Positive pitch moves toward the top (smaller row index).
        vertical_direction = torch.where(
            delta_pitch >= 0, -torch.ones_like(row), torch.ones_like(row)
        )
        neighbour_row = torch.where(
            horizontal_motion,
            row,
            (row + vertical_direction).clamp(0, self.grid_rows - 1),
        )
        neighbour_col = torch.where(
            horizontal_motion,
            torch.remainder(col + horizontal_direction, self.grid_cols),
            col,
        )
        neighbour = neighbour_row * self.grid_cols + neighbour_col
        fallback = row * self.grid_cols + torch.remainder(col + 1, self.grid_cols)
        neighbour = torch.where(neighbour == current, fallback, neighbour)
        return torch.stack((current, neighbour))

    def configuration(self):
        return {
            "type": "cached-viewport",
            "policy": self.policy,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "motion_threshold_deg": self.motion_threshold_deg,
        }


class CachedPatchFeatureStore:
    """Lazy/preloadable per-video store for ``[frames, patches, 768]`` tensors."""

    def __init__(self, root, device="cpu", expected_patches=16,
                 expected_feature_dim=768):
        self.root = Path(root)
        self.device = torch.device(device)
        self.expected_patches = int(expected_patches)
        self.expected_feature_dim = int(expected_feature_dim)
        self._videos = {}

    def _path(self, video_index):
        return self.root / f"video{int(video_index)}_patch_features.pt"

    def load_video(self, video_index):
        video_index = int(video_index)
        if video_index in self._videos:
            return self._videos[video_index]
        path = self._path(video_index)
        if not path.is_file():
            raise FileNotFoundError(f"cached patch features not found: {path}")
        payload = torch.load(path, map_location="cpu")
        features = payload.get("features") if isinstance(payload, dict) else payload
        if not torch.is_tensor(features) or features.ndim != 3:
            raise ValueError(f"{path} must contain a [frames, patches, dim] tensor")
        expected_tail = (self.expected_patches, self.expected_feature_dim)
        if tuple(features.shape[1:]) != expected_tail:
            raise ValueError(
                f"{path} feature shape {tuple(features.shape)} does not end in "
                f"{expected_tail}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"non-finite cached patch features in {path}")
        features = features.to(device=self.device, dtype=torch.float16)
        self._videos[video_index] = features
        return features

    def preload(self, video_indices):
        for video_index in video_indices:
            self.load_video(video_index)

    def get(self, video_index, image_index, patch_indices):
        features = self.load_video(video_index)
        # Extracted Jin2022 frames are one-based.  Treat a boundary index of
        # zero as the first frame, matching the intended current-frame lookup.
        frame_offset = max(1, int(image_index)) - 1
        if frame_offset < 0 or frame_offset >= features.shape[0]:
            raise IndexError(
                f"frame {image_index} outside cached video{video_index} range "
                f"1..{features.shape[0]}"
            )
        indices = torch.as_tensor(
            patch_indices, dtype=torch.long, device=features.device
        )
        return features[frame_offset].index_select(0, indices)
