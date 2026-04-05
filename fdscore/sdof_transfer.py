from __future__ import annotations

import numpy as np

from .types import Metric


def h_baseacc_to_x(freq_hz: np.ndarray, fn_hz: np.ndarray, zeta: float) -> np.ndarray:
    """Base-acceleration -> relative displacement receptance for an SDOF.

    Hx(w) = -1 / ((w0^2 - w^2) + 2*zeta*w0*j*w)
    """
    w = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    w0 = 2.0 * np.pi * np.asarray(fn_hz, dtype=float)
    return -1.0 / ((w0 * w0 - w * w) + 2.0 * zeta * w0 * (1j * w))


def build_transfer_matrix(
    *,
    fs: float,
    n: int,
    f0_hz: np.ndarray,
    zeta: float,
    metric: Metric,
) -> np.ndarray:
    """Build FFT-domain transfer matrix for different response metrics.

    Input is base acceleration time history a_base(t). We compute the oscillator response in frequency domain
    and reconstruct time series via irfft.

    Definitions (base excitation):
      x_rel = Hx * A_base
      v_rel = j*w * x_rel
      a_rel = -w^2 * x_rel
      a_abs = a_base + a_rel = (1 - w^2 * Hx) * A_base

    Metric mapping:
      - 'disp': uses x_rel
      - 'vel' : uses v_rel
      - 'acc' : uses a_abs  (absolute acceleration of the mass)
      - 'pv'  : uses pseudo-velocity = (2*pi*f0) * x_rel
    """
    f_fft = np.fft.rfftfreq(int(n), d=1.0 / float(fs))
    f0 = np.asarray(f0_hz, dtype=float)
    Hx = h_baseacc_to_x(f_fft[None, :], f0[:, None], float(zeta))

    w = 2.0 * np.pi * f_fft[None, :]
    if metric == "disp":
        H = Hx
    elif metric == "vel":
        H = (1j * w) * Hx
    elif metric == "acc":
        H = 1.0 + (-w * w) * Hx
    elif metric == "pv":
        H = (2.0 * np.pi * f0[:, None]) * Hx
    else:  # pragma: no cover
        raise ValueError(f"Unknown metric: {metric}")

    return H


def build_transfer_psd(
    *,
    f_psd_hz: np.ndarray,
    f0_hz: np.ndarray,
    zeta: float,
    metric: Metric,
) -> np.ndarray:
    """Build frequency-response transfer matrix for PSD-domain computations.

    Parameters
    ----------
    f_psd_hz:
        Frequency vector of the input PSD (one-sided), shape (Nf,)
    f0_hz:
        Oscillator natural frequencies, shape (No,)
    zeta:
        Damping ratio
    metric:
        'disp' | 'vel' | 'acc' | 'pv'

    Returns
    -------
    H : complex ndarray, shape (No, Nf)
        Transfer such that: PSD_resp = |H|^2 * PSD_base_acc
    """
    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    f0 = np.asarray(f0_hz, dtype=float).reshape(-1)

    Hx = h_baseacc_to_x(f_psd[None, :], f0[:, None], float(zeta))
    w = 2.0 * np.pi * f_psd[None, :]

    if metric == "disp":
        return Hx
    if metric == "vel":
        return (1j * w) * Hx
    if metric == "acc":
        return 1.0 + (-w * w) * Hx
    if metric == "pv":
        return (2.0 * np.pi * f0[:, None]) * Hx
    raise ValueError(f"Unknown metric: {metric}")
