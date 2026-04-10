from __future__ import annotations

import numpy as np

from ._psd_utils import clip_tiny_negative_psd_or_raise
from .validate import ValidationError


def synthesize_time_from_psd(
    *,
    f_psd_hz: np.ndarray,
    psd: np.ndarray,
    fs: float,
    duration_s: float,
    seed: int | None = None,
    nfft: int | None = None,
    remove_mean: bool = True,
) -> np.ndarray:
    """Synthesize a stationary Gaussian time history from a one-sided PSD.

    Parameters
    ----------
    f_psd_hz : numpy.ndarray
        One-dimensional one-sided PSD frequency grid in Hz.
    psd : numpy.ndarray
        One-dimensional one-sided PSD values in units squared per hertz.
    fs : float
        Sampling rate in Hz.
    duration_s : float
        Desired duration in seconds. The output length is
        ``N = round(duration_s * fs)``.
    seed : int or None
        Random seed used for reproducible phase generation.
    nfft : int or None
        FFT length used for synthesis. If ``None``, the next power of two
        greater than or equal to ``N`` is used. If provided, it must satisfy
        ``nfft >= N``.
    remove_mean : bool
        If ``True``, subtract the sample mean after synthesis.

    Returns
    -------
    numpy.ndarray
        Synthesized time history of length ``N``.

    Notes
    -----
    This routine is typically used by iterative inversion predictors that map
    a PSD to a synthetic time history.

    The generated realization is stationary and Gaussian by construction and
    preserves the target PSD only in the statistical, ensemble sense. It does
    not reproduce transient structure, non-stationarity, or non-Gaussian
    tails.

    For finite durations, Welch estimates from the synthesized signal will not
    match the input PSD exactly point by point.
    """
    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    P = np.asarray(psd, dtype=float).reshape(-1)

    if f_psd.size < 2 or P.size < 2 or f_psd.shape != P.shape:
        raise ValidationError("f_psd_hz and psd must be 1D arrays with same length >= 2.")
    if not np.all(np.isfinite(f_psd)) or not np.all(np.isfinite(P)):
        raise ValidationError("PSD inputs must be finite.")
    if not np.all(np.diff(f_psd) > 0):
        raise ValidationError("f_psd_hz must be strictly increasing.")
    P = clip_tiny_negative_psd_or_raise(P, label="psd")

    if not np.isfinite(fs) or float(fs) <= 0:
        raise ValidationError("fs must be finite and > 0.")
    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValidationError("duration_s must be finite and > 0.")

    N = int(round(float(duration_s) * float(fs)))
    if N < 8:
        raise ValidationError("duration_s*fs must yield N>=8 samples.")

    if nfft is None:
        nfft = 1 << (N - 1).bit_length()
    nfft = int(nfft)
    if nfft < N:
        raise ValidationError("nfft must be >= N.")

    df = float(fs) / float(nfft)
    f_fft = np.arange(0, nfft // 2 + 1, dtype=float) * df

    # Interpolate PSD onto FFT bins (one-sided)
    P_fft = np.interp(f_fft, f_psd, P, left=0.0, right=0.0)
    P_fft = clip_tiny_negative_psd_or_raise(P_fft, label="interpolated psd")

    rng = np.random.default_rng(seed)

    # Complex spectrum for rfft bins
    X = np.zeros_like(f_fft, dtype=np.complex128)
    has_nyquist = (nfft % 2) == 0
    interior_stop = -1 if has_nyquist else None

    # Random phases for positive freqs with conjugate pairs.
    paired_bins = P_fft[1:interior_stop]
    if paired_bins.size > 0:
        phi = rng.uniform(0.0, 2.0 * np.pi, size=paired_bins.size)
        mag = float(nfft) * np.sqrt(0.5 * paired_bins * df)
        X[1:interior_stop] = mag * (np.cos(phi) + 1j * np.sin(phi))

    # DC is a singleton bin; preserve it unless remove_mean is requested.
    if not remove_mean and P_fft[0] > 0.0:
        X[0] = float(nfft) * np.sqrt(P_fft[0] * df) * (1.0 if rng.uniform() < 0.5 else -1.0)

    # Even-length FFTs also have a singleton Nyquist bin that must not be doubled.
    if has_nyquist and P_fft[-1] > 0.0:
        X[-1] = float(nfft) * np.sqrt(P_fft[-1] * df) * (1.0 if rng.uniform() < 0.5 else -1.0)

    x = np.fft.irfft(X, n=nfft)[:N].astype(float, copy=False)
    if remove_mean:
        x = x - float(np.mean(x))

    return x
