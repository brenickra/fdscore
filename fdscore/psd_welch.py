"""Welch PSD estimation for vibration and inversion workflows.

This module provides the library's standard power spectral density
estimator for stationary time histories. The implementation is a thin
validated wrapper around ``scipy.signal.welch``, with explicit
handling of positivity, optional band cropping, and metadata-friendly
error reporting.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .types import PSDParams
from .validate import ValidationError
from ._psd_utils import clip_tiny_negative_psd_or_raise


def compute_psd_welch(
    x: np.ndarray,
    fs: float,
    psd: PSDParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a one-sided PSD using Welch's method.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional input time history.
    fs : float
        Sampling rate in Hz.
    psd : PSDParams
        PSD-estimation settings. The current implementation supports only
        ``method="welch"``.

    Returns
    -------
    f_hz : numpy.ndarray
        One-dimensional frequency grid in Hz.
    psd_values : numpy.ndarray
        One-dimensional PSD values in units squared per hertz.

    Notes
    -----
    This helper is purely numerical and performs no file I/O.

    The output is the one-sided PSD returned by Welch's method under the
    requested detrending, window, overlap, and segment-length settings.
    When ``psd.fmin`` or ``psd.fmax`` is provided, the spectrum is
    cropped after estimation.

    Tiny negative PSD values caused by floating-point noise are clipped to
    zero. Materially negative values are rejected because they would
    invalidate downstream log-domain operations and stochastic synthesis.

    References
    ----------
    Welch, P. D. (1967). The use of the fast Fourier transform for the
    estimation of power spectra: A method based on time averaging over
    short, modified periodograms. *IEEE Transactions on Audio and
    Electroacoustics*, 15(2), 70-73.
    """
    if psd.method != "welch":
        raise ValidationError("PSDParams.method must be 'welch'.")
    if not np.isfinite(fs) or float(fs) <= 0:
        raise ValidationError("fs must be finite and > 0.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValidationError("x must be a 1D array with length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("x must contain only finite values.")

    detrend = False if psd.detrend == "none" else psd.detrend

    f, Pxx = welch(
        x,
        fs=float(fs),
        window=str(psd.window),
        nperseg=psd.nperseg,
        noverlap=psd.noverlap,
        detrend=detrend,
        scaling=str(psd.scaling),
        return_onesided=bool(psd.onesided),
    )

    f = np.asarray(f, dtype=float)
    Pxx = np.asarray(Pxx, dtype=float)

    if f.ndim != 1 or Pxx.ndim != 1 or f.shape != Pxx.shape:
        raise ValidationError("welch returned unexpected shapes.")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(Pxx))):
        raise ValidationError("welch produced non-finite outputs.")
    Pxx = clip_tiny_negative_psd_or_raise(Pxx, label="welch PSD")

    if psd.fmin is not None or psd.fmax is not None:
        fmin = float(psd.fmin) if psd.fmin is not None else float(f[0])
        fmax = float(psd.fmax) if psd.fmax is not None else float(f[-1])
        if fmax <= fmin:
            raise ValidationError("PSDParams requires fmax > fmin when cropping.")
        m = (f >= fmin) & (f <= fmax)
        f = f[m]
        Pxx = Pxx[m]

    if f.size < 2:
        raise ValidationError("PSD frequency grid too small after cropping.")

    return f, Pxx
