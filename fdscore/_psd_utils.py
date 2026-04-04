from __future__ import annotations

import numpy as np

from .validate import ValidationError


def clip_tiny_negative_psd_or_raise(
    values: np.ndarray,
    *,
    label: str,
    rtol: float = 1e-12,
    atol: float = 1e-30,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(arr < 0.0):
        return arr

    ref = max(float(np.max(np.abs(arr))), 1.0)
    tol = max(float(atol), float(rtol) * ref)
    min_val = float(np.min(arr))
    if min_val < -tol:
        raise ValidationError(f"{label} contains negative values below numerical tolerance.")
    return np.maximum(arr, 0.0)
