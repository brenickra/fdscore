from __future__ import annotations

import numpy as np

from .validate import ValidationError


def preprocess_shock_signal(x: np.ndarray, *, detrend: str) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()

    if detrend == "linear":
        n = y.size
        if n > 1:
            t = np.arange(n, dtype=float)
            p = np.polyfit(t, y, 1)
            y -= p[0] * t + p[1]
    elif detrend == "mean":
        y -= float(np.mean(y))
    elif detrend == "median":
        y -= float(np.median(y))
    elif detrend != "none":
        raise ValidationError("detrend must be one of: 'linear', 'mean', 'median', 'none'.")

    return y
