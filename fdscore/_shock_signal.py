"""Preprocessing helpers for transient shock time histories.

The functions in this module prepare raw one-dimensional signals before
shock-event detection or shock-spectrum analysis. The supported
operations are deliberately minimal and focus on simple detrending modes
that remove offsets or low-order drift without altering the underlying
transient structure more than necessary.
"""

from __future__ import annotations

import numpy as np

from .validate import ValidationError


def preprocess_shock_signal(x: np.ndarray, *, detrend: str) -> np.ndarray:
    """Apply the configured detrending policy to a shock time history.

    Parameters
    ----------
    x : numpy.ndarray
        Input one-dimensional signal.
    detrend : {"linear", "mean", "median", "none"}
        Detrending mode to apply before shock processing.

    Returns
    -------
    numpy.ndarray
        Detrended copy of the input signal.

    Notes
    -----
    ``"linear"`` removes the least-squares affine trend, ``"mean"``
    removes the arithmetic mean, ``"median"`` removes the median, and
    ``"none"`` leaves the signal unchanged. The function always returns a
    copy so that downstream routines can operate without mutating the
    caller's original array.
    """
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
