from __future__ import annotations

import numpy as np


def build_example_psd(*, fs: float, fmax_hz: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth one-sided acceleration PSD in g^2/Hz."""
    nyq = float(fs) / 2.0
    fmax = min(float(fmax_hz) if fmax_hz is not None else nyq, nyq)
    f = np.linspace(0.0, fmax, int(round(fmax)) + 1)

    psd = (
        2.0e-4 * np.exp(-0.5 * ((f - 28.0) / 10.0) ** 2)
        + 5.0e-4 * np.exp(-0.5 * ((f - 85.0) / 18.0) ** 2)
        + 3.5e-4 * np.exp(-0.5 * ((f - 180.0) / 35.0) ** 2)
        + 1.0e-6
    )
    psd[0] = 0.0
    return f, psd.astype(float, copy=False)


def build_multitone_signal(*, fs: float, duration_s: float) -> np.ndarray:
    """Return a deterministic base-acceleration time history in g."""
    t = np.arange(0.0, float(duration_s), 1.0 / float(fs))
    x = (
        0.08 * np.sin(2.0 * np.pi * 22.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 67.0 * t + 0.4)
        + 0.03 * np.sin(2.0 * np.pi * 145.0 * t + 1.1)
    )
    x += 0.005 * np.sin(2.0 * np.pi * 2.0 * t)
    return x.astype(float, copy=False)


def median_abs_log10(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (a > 0.0) & (b > 0.0)
    if not np.any(mask):
        return float("nan")
    return float(np.median(np.abs(np.log10(a[mask]) - np.log10(b[mask]))))
