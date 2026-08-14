"""
Utilities for looking up how many extracted frames a video actually has, and for safely
mapping a computed frame index onto the range that is actually available on disk.

A few Jin2022 videos (see config.dataset_short_frame_videos) have fewer extracted frames
than the nominal count assumed elsewhere in the codebase (1500 for videos 10-18, 1800 for
the rest). dataset/extract_features_cache.py generates a frame_counts.json manifest with
the true per-video frame count; this module loads that manifest once and clamps any
out-of-range frame index to the last available frame, logging the first time this happens
per video so silent frame-repetition at the tail of short videos doesn't go unnoticed.
"""
import json
import os
from config import cfg

_NOMINAL_JIN2022_SHORT_VIDEO_GROUP = list(range(10, 19))  # nominal 1500
_NOMINAL_JIN2022_COUNT = 1500
_NOMINAL_JIN2022_COUNT_LONG = 1800


def nominal_frame_count(dataset, video):
    if dataset == 'Jin2022':
        return _NOMINAL_JIN2022_COUNT if video in _NOMINAL_JIN2022_SHORT_VIDEO_GROUP else _NOMINAL_JIN2022_COUNT_LONG
    raise ValueError(f'nominal_frame_count is only defined for Jin2022, got dataset={dataset}')


def load_frame_counts(dataset):
    """
    Load the {video_index: actual_frame_count} manifest built by extract_features_cache.py.
    Falls back to nominal counts (with a warning) if the manifest hasn't been generated yet.

    :return: dict mapping int video index -> int actual frame count
    """
    manifest_path = cfg.dataset_frame_count_manifest[dataset]
    if not os.path.exists(manifest_path):
        print(f'\033[33mWarning:\033[0m frame count manifest not found at {manifest_path}. '
              f'Falling back to nominal frame counts; run dataset/extract_features_cache.py first '
              f'to get the real per-video counts (needed for videos '
              f'{cfg.dataset_short_frame_videos.get(dataset, [])}).')
        video_num = 27 if dataset == 'Jin2022' else 18
        return {v: nominal_frame_count(dataset, v) for v in range(1, video_num + 1)}
    with open(manifest_path, 'r') as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


class FrameIndexClamper:
    """
    Clamps a 1-based frame index to the actual available range for a video, logging the
    first clamp event per video (subsequent clamps for the same video are silent to avoid
    log spam over a training run).
    """
    def __init__(self, dataset, frame_counts=None):
        self.dataset = dataset
        self.frame_counts = frame_counts if frame_counts is not None else load_frame_counts(dataset)
        self._warned_videos = set()

    def clamp(self, video, frame_index_1based):
        max_frame = self.frame_counts.get(video)
        if max_frame is None or frame_index_1based <= max_frame:
            return frame_index_1based
        if video not in self._warned_videos:
            print(f'\033[33mWarning:\033[0m {self.dataset} video{video} requested frame '
                  f'{frame_index_1based} but only {max_frame} frames are available; '
                  f'clamping to the last available frame for the remainder of this run.')
            self._warned_videos.add(video)
        return max_frame
