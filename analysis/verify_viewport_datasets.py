"""Verify that processed Jin2022/Wu2017 viewport CSVs are ready for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_checks import validate_viewport_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Jin2022", "Wu2017"])
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "valid", "test"), default=["test"]
    )
    parser.add_argument("--frequency", type=int, default=5)
    args = parser.parse_args()

    reports = [
        validate_viewport_dataset(dataset, args.splits, args.frequency)
        for dataset in args.datasets
    ]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
