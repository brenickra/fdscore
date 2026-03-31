from __future__ import annotations

import numpy as np


def preprocess_signal(x: np.ndarray, mode: str) -> np.ndarray:
    """Optional centering/detrending for a time signal.

    Modes:
      - 'linear': remove best-fit line (polyfit degree 1) in sample-index domain
      - 'mean'  : remove mean
      - 'none'  : no preprocessing
    """
    y = np.asarray(x, dtype=float).copy()
    if mode == "linear":
        n = y.size
        if n > 1:
            t = np.arange(n, dtype=float)
            p = np.polyfit(t, y, 1)
            y -= p[0] * t + p[1]
    elif mode == "mean":
        y -= float(np.mean(y))
    elif mode != "none":
        raise ValueError(f"Invalid preprocess mode: {mode}")
    return y
