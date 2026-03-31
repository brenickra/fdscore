from __future__ import annotations

import numpy as np

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
    """Synthesize a stationary Gaussian time history from a **one-sided** PSD using random phase IFFT.

    Parameters
    ----------
    f_psd_hz, psd:
        One-sided PSD definition (Hz, units^2/Hz). Must be same shape and strictly increasing in frequency.
    fs:
        Sampling rate [Hz]
    duration_s:
        Desired duration [s]. Output length is `N = round(duration_s*fs)`.
    seed:
        Random seed for reproducibility.
    nfft:
        FFT length. If None, uses next power-of-two >= N.
        If provided, must be >= N.
    remove_mean:
        If True, subtracts the mean after synthesis.

    Returns
    -------
    x : ndarray
        Time history of length N.

    Notes
    -----
    This routine is typically used by iterative inversion predictors that map
    a PSD to a synthetic time history.
    """
    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    P = np.asarray(psd, dtype=float).reshape(-1)

    if f_psd.size < 2 or P.size < 2 or f_psd.shape != P.shape:
        raise ValidationError("f_psd_hz and psd must be 1D arrays with same length >= 2.")
    if not np.all(np.isfinite(f_psd)) or not np.all(np.isfinite(P)):
        raise ValidationError("PSD inputs must be finite.")
    if not np.all(np.diff(f_psd) > 0):
        raise ValidationError("f_psd_hz must be strictly increasing.")
    if np.any(P < 0):
        P = np.maximum(P, 0.0)

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
    P_fft = np.maximum(P_fft, 0.0)

    rng = np.random.default_rng(seed)

    # Complex spectrum for rfft bins
    X = np.zeros_like(f_fft, dtype=np.complex128)

    # Random phases for positive freqs excluding DC and Nyquist
    if f_fft.size > 2:
        phi = rng.uniform(0.0, 2.0 * np.pi, size=f_fft.size - 2)
        mag = float(nfft) * np.sqrt(0.5 * P_fft[1:-1] * df)
        X[1:-1] = mag * (np.cos(phi) + 1j * np.sin(phi))

    # DC and Nyquist set to zero (mean removed later anyway)
    X[0] = 0.0 + 0.0j
    X[-1] = 0.0 + 0.0j

    x = np.fft.irfft(X, n=nfft)[:N].astype(float, copy=False)
    if remove_mean:
        x = x - float(np.mean(x))

    return x
