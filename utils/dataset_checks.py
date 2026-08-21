"""Preflight checks for processed viewport trajectory datasets."""

from __future__ import annotations

import os
from collections.abc import Iterable

from config import cfg


def expected_viewport_files(dataset: str, splits: Iterable[str], frequency: int) -> list[str]:
    """Return every processed viewport CSV required by the requested splits."""
    if dataset not in cfg.dataset_list:
        raise ValueError(f"Unknown dataset: {dataset!r}; expected one of {cfg.dataset_list}")
    split_names = tuple(splits)
    unknown = [name for name in split_names if name not in {"train", "valid", "test"}]
    if unknown:
        raise ValueError(f"Unknown dataset splits: {unknown}")

    paths = []
    for split in split_names:
        for video in cfg.dataset_video_split[dataset][split]:
            for user in cfg.dataset_user_split[dataset][split]:
                paths.append(os.path.join(
                    cfg.dataset[dataset],
                    f"video{video}",
                    f"{frequency}Hz",
                    f"simple_{frequency}Hz_user{user}.csv",
                ))
    return paths


def validate_viewport_dataset(
    dataset: str,
    splits: Iterable[str] = ("test",),
    frequency: int = 5,
) -> dict[str, object]:
    """Fail early with useful diagnostics when processed trajectory files are absent."""
    split_names = tuple(splits)
    expected = expected_viewport_files(dataset, split_names, frequency)
    missing = [path for path in expected if not os.path.isfile(path)]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        remainder = len(missing) - min(len(missing), 10)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            f"Processed {dataset} viewport dataset is incomplete: "
            f"{len(missing)}/{len(expected)} required CSV files are missing for "
            f"splits={split_names}, frequency={frequency}Hz.\n{examples}{suffix}"
        )
    return {
        "dataset": dataset,
        "splits": list(split_names),
        "frequency_hz": int(frequency),
        "root": os.path.abspath(cfg.dataset[dataset]),
        "required_csv_files": len(expected),
        "status": "ok",
    }
