"""Pure numeric policies shared by NBS optimizer safety checks."""

import math


def classify_update(
    old_rms,
    update_rms,
    *,
    ratio_floor,
    warning_ratio,
    maximum_ratio,
    maximum_update_rms,
):
    """Classify a finite optimizer update without depending on PyTorch."""
    old_rms = float(old_rms)
    update_rms = float(update_rms)
    ratio = update_rms / max(old_rms, float(ratio_floor))
    finite = math.isfinite(old_rms) and math.isfinite(update_rms)
    finite = finite and math.isfinite(ratio)
    rollback = (
        not finite
        or ratio > float(maximum_ratio)
        or update_rms > float(maximum_update_rms)
    )
    return {
        'old_rms': old_rms,
        'update_rms': update_rms,
        'update_ratio': ratio,
        'finite': finite,
        'warning': finite and ratio > float(warning_ratio),
        'rollback': rollback,
    }
