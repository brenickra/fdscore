"""Generic preprocessing helpers for time-domain input signals.

The routines in this module provide lightweight centering and detrending
operations that are reused by time-domain workflows before FFT-based
response reconstruction or cycle counting.
"""

from __future__ import annotations

import numpy as np


def preprocess_signal(x: np.ndarray, mode: str) -> np.ndarray:
    """Apply the requested preprocessing mode to a time history.

    Parameters
    ----------
    x : numpy.ndarray
        Input one-dimensional signal.
    mode : {"linear", "mean", "none"}
        Preprocessing mode. ``"linear"`` removes the least-squares
        affine trend in sample-index space, ``"mean"`` removes the
        arithmetic mean, and ``"none"`` leaves the signal unchanged.

    Returns
    -------
    numpy.ndarray
        Preprocessed copy of the input signal.

    Notes
    -----
    The function always operates on a copy of the input array so that the
    caller's original signal is preserved. The detrending is intentionally
    minimal and is designed to remove low-order bias terms rather than to
    reshape the underlying vibration content.
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
