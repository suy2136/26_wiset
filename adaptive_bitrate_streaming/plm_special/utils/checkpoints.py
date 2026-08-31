"""Safe checkpoint rotation helpers for large NBS adapter checkpoints."""

from pathlib import Path
import shutil
import uuid


ADAPTER_REQUIRED_FILES = (
    "adapter_config.json",
    "modules_except_plm.bin",
    "checkpoint_metadata.json",
)
NBS_REQUIRED_FILES = (*ADAPTER_REQUIRED_FILES, "nash_rank_allocator.pt")
NBS_ADAPTER_FILES = ("adapter_model.bin", "adapter_model.safetensors")


def is_complete_nbs_checkpoint(path):
    return is_complete_adapter_checkpoint(path, require_nbs_allocator=True)


def is_complete_adapter_checkpoint(path, require_nbs_allocator=False):
    """Return whether a PEFT checkpoint is complete for its allocator mode."""
    path = Path(path)
    required = (
        NBS_REQUIRED_FILES if require_nbs_allocator else ADAPTER_REQUIRED_FILES
    )
    return (
        path.is_dir()
        and all((path / name).is_file() for name in required)
        and any((path / name).is_file() for name in NBS_ADAPTER_FILES)
    )


def _remove_directory(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def prepare_best_latest_retention(
    checkpoint_root, require_nbs_allocator=True
):
    """Remove corrupt epoch saves and retain only the newest complete save.

    ``require_nbs_allocator=True`` preserves the historical NBS validation
    contract.  Plain LoRA callers opt out because their checkpoints do not
    contain ``nash_rank_allocator.pt``.
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        if path.is_dir() and (
            path.name.startswith(".checkpoint-tmp-")
            or path.name.startswith(".checkpoint-previous-")
        ):
            _remove_directory(path)

    numeric = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    complete = []
    removed_incomplete = []
    for path in numeric:
        if is_complete_adapter_checkpoint(path, require_nbs_allocator):
            complete.append(path)
        else:
            removed_incomplete.append(path.name)
            _remove_directory(path)

    latest = root / "latest"
    if latest.exists() and not is_complete_adapter_checkpoint(
        latest, require_nbs_allocator
    ):
        removed_incomplete.append("latest")
        _remove_directory(latest)
    if not latest.exists() and complete:
        newest = complete.pop()
        newest.replace(latest)
    for path in complete:
        _remove_directory(path)
    return removed_incomplete


def atomic_save_nbs_checkpoint(target, writer):
    """Write a complete NBS checkpoint before replacing the prior target."""
    return atomic_save_adapter_checkpoint(
        target, writer, require_nbs_allocator=True
    )


def atomic_save_adapter_checkpoint(
    target, writer, require_nbs_allocator=False
):
    """Atomically replace a complete LoRA or NBS adapter checkpoint."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = target.parent / f".checkpoint-tmp-{token}"
    previous = target.parent / f".checkpoint-previous-{token}"
    try:
        writer(str(temporary))
        if not is_complete_adapter_checkpoint(
            temporary, require_nbs_allocator
        ):
            checkpoint_type = "NBS" if require_nbs_allocator else "LoRA"
            raise RuntimeError(
                f"incomplete {checkpoint_type} checkpoint written to {temporary}"
            )
        if target.exists():
            target.replace(previous)
        temporary.replace(target)
        _remove_directory(previous)
    except Exception:
        _remove_directory(temporary)
        if previous.exists() and not target.exists():
            previous.replace(target)
        raise
