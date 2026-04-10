"""Utility helpers for PSD sanitation and numerical robustness.

This module contains low-level support functions used to enforce
positivity constraints on PSD-like arrays before they enter downstream
operations such as inversion, metrics, or synthesis.
"""

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
    """Clip numerically tiny negative PSD values or raise on material negativity.

    Parameters
    ----------
    values : numpy.ndarray
        Input PSD-like array to validate.
    label : str
        Human-readable name used in error messages.
    rtol : float, optional
        Relative tolerance used to distinguish floating-point noise from
        physically invalid negative values.
    atol : float, optional
        Absolute tolerance floor combined with ``rtol``.

    Returns
    -------
    numpy.ndarray
        Validated array where numerically tiny negative entries have been
        clipped to zero.

    Notes
    -----
    Small negative values can appear after interpolation or spectral
    manipulations due to roundoff. This helper treats those values as
    numerical noise while still rejecting genuinely negative PSD content
    that would invalidate downstream logarithms or synthesis steps.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(arr < 0.0):
        return arr

    ref = max(float(np.max(np.abs(arr))), 1.0)
    tol = max(float(atol), float(rtol) * ref)
    min_val = float(np.min(arr))
    if min_val < -tol:
        raise ValidationError(f"{label} contains negative values below numerical tolerance.")
    return np.maximum(arr, 0.0)
