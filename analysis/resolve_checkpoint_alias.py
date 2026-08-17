"""Resolve a logical NBS checkpoint role to its physical checkpoint path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_checkpoint(path: Path) -> Path:
    current = path.resolve()
    visited: set[Path] = set()
    while True:
        if current in visited:
            raise ValueError(f"checkpoint alias cycle detected at {current}")
        visited.add(current)
        alias_path = current / "checkpoint_alias.json"
        if not alias_path.exists():
            return current
        alias = json.loads(alias_path.read_text(encoding="utf-8"))
        target = alias.get("alias_of")
        if not target:
            raise ValueError(f"checkpoint alias has no alias_of target: {alias_path}")
        target_path = Path(target)
        current = (
            target_path.resolve()
            if target_path.is_absolute()
            else (current / target_path).resolve()
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(resolve_checkpoint(args.checkpoint))


if __name__ == "__main__":
    main()
