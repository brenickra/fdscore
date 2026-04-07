from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .types import Metric
from .validate import ValidationError

_SHOCK_SUPPORTED_METRICS = ("acc", "pv")
_SHOCK_SUPPORTED_PEAK_MODES = ("abs", "pos", "neg", "both")


def _validate_shock_metric(metric: Metric) -> None:
    if metric not in _SHOCK_SUPPORTED_METRICS:
        raise ValidationError("shock engine metric must be one of: 'acc','pv'.")


def _validate_shock_peak_mode(peak_mode: str) -> None:
    if peak_mode not in _SHOCK_SUPPORTED_PEAK_MODES:
        raise ValidationError("shock peak_mode must be one of: 'abs','pos','neg','both'.")


def _validate_shock_inputs(
    *,
    x: np.ndarray,
    fs: float,
    f0_hz: np.ndarray,
    zeta: float,
    metric: Metric,
    peak_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_shock_metric(metric)
    _validate_shock_peak_mode(peak_mode)

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 4:
        raise ValidationError("shock engine input x must have length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("shock engine input x must contain only finite values.")

    if not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValidationError("shock engine fs must be finite and > 0.")
    if not np.isfinite(zeta) or float(zeta) <= 0.0 or float(zeta) >= 1.0:
        raise ValidationError("shock engine zeta must satisfy 0 < zeta < 1.")

    f0 = np.asarray(f0_hz, dtype=float).reshape(-1)
    if f0.size == 0:
        raise ValidationError("shock engine frequency grid must be non-empty.")
    if not np.all(np.isfinite(f0)) or np.any(f0 <= 0.0):
        raise ValidationError("shock engine frequency grid must contain only finite values > 0.")
    if not np.all(np.diff(f0) > 0.0):
        raise ValidationError("shock engine frequency grid must be strictly increasing.")
    if float(np.max(f0)) >= float(fs) / 2.0:
        raise ValidationError("shock engine frequency grid must stay strictly below Nyquist.")

    return x, f0


def _shock_abs_acc_coefficients(omega: float, q: float, dt: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ISO-18431-4 style IIR coefficients for absolute acceleration response."""
    a1 = omega * dt / (2.0 * q)
    b1 = omega * dt * math.sqrt(1.0 - 1.0 / (4.0 * (q * q)))

    exp_a = math.exp(-a1)
    sin_term = math.sin(b1) / b1
    cos_term = math.cos(b1)
    exp_2a = math.exp(-2.0 * a1)

    b = (
        1.0 - exp_a * sin_term,
        2.0 * exp_a * (sin_term - cos_term),
        exp_2a - exp_a * sin_term,
    )
    a = (
        1.0,
        -2.0 * exp_a * cos_term,
        exp_2a,
    )
    return b, a


def _shock_pv_coefficients(omega: float, q: float, dt: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ISO-18431-4 style IIR coefficients for pseudo-velocity response."""
    a1 = omega * dt / (2.0 * q)
    b1 = omega * dt * math.sqrt(1.0 - 1.0 / (4.0 * (q * q)))
    c1 = dt * (omega * omega)
    q_term = (1.0 / (2.0 * (q * q)) - 1.0) / math.sqrt(1.0 - 1.0 / (4.0 * (q * q)))

    exp_a = math.exp(-a1)
    cos_term = math.cos(b1)
    sin_term = math.sin(b1)
    exp_2a = math.exp(-2.0 * a1)

    b = (
        ((1.0 - exp_a * cos_term) / q - q_term * exp_a * sin_term - omega * dt) / c1,
        (2.0 * exp_a * cos_term * omega * dt - (1.0 - exp_2a) / q + 2.0 * q_term * exp_a * sin_term) / c1,
        (-exp_2a * (omega * dt + 1.0 / q) + exp_a * cos_term / q - q_term * exp_a * sin_term) / c1,
    )
    a = (
        1.0,
        -2.0 * exp_a * cos_term,
        exp_2a,
    )
    return b, a


def _shock_filter_coefficients(
    *,
    f0_hz: np.ndarray,
    zeta: float,
    dt: float,
    metric: Metric,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized coefficient builder for the shock-domain recursive engine.

    Returns
    -------
    b, a : ndarray
        Arrays of shape `(len(f0_hz), 3)` suitable for per-frequency `lfilter` calls.
    """
    _validate_shock_metric(metric)

    if not np.isfinite(zeta) or float(zeta) <= 0.0 or float(zeta) >= 1.0:
        raise ValidationError("shock engine zeta must satisfy 0 < zeta < 1.")
    if not np.isfinite(dt) or float(dt) <= 0.0:
        raise ValidationError("shock engine dt must be finite and > 0.")

    f0 = np.asarray(f0_hz, dtype=float).reshape(-1)
    if f0.size == 0:
        raise ValidationError("shock engine frequency grid must be non-empty.")

    q = 1.0 / (2.0 * float(zeta))
    if q <= 0.5:
        raise ValidationError("shock engine requires underdamped oscillators with q > 0.5.")

    b = np.zeros((f0.size, 3), dtype=float)
    a = np.zeros((f0.size, 3), dtype=float)

    coeff_fn = _shock_abs_acc_coefficients if metric == "acc" else _shock_pv_coefficients
    for i, fn in enumerate(f0):
        bi, ai = coeff_fn(2.0 * math.pi * float(fn), q, float(dt))
        b[i, :] = np.asarray(bi, dtype=float)
        a[i, :] = np.asarray(ai, dtype=float)

    return b, a


def _shock_zero_padding_length(*, fs: float, f0_hz: np.ndarray, zeta: float, peak_mode: str) -> int:
    """Return the number of trailing zero samples used to capture ring-down."""
    _validate_shock_peak_mode(peak_mode)

    if not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValidationError("shock engine fs must be finite and > 0.")
    if not np.isfinite(zeta) or float(zeta) <= 0.0 or float(zeta) >= 1.0:
        raise ValidationError("shock engine zeta must satisfy 0 < zeta < 1.")

    fmin = float(np.min(np.asarray(f0_hz, dtype=float)))
    if fmin <= 0.0:
        raise ValidationError("shock engine minimum natural frequency must be > 0.")

    pad_s = 1.0 / (fmin * math.sqrt(1.0 - float(zeta) ** 2))
    if peak_mode == "abs":
        pad_s *= 0.5
    return max(1, int(math.floor(pad_s * float(fs))) + 1)


def _extract_shock_peak(
    response: np.ndarray,
    response_tail: np.ndarray,
    *,
    peak_mode: str,
) -> np.ndarray:
    """Extract shock peaks from main response and zero-padded ring-down."""
    _validate_shock_peak_mode(peak_mode)

    neg_mag = -min(float(np.min(response)), float(np.min(response_tail)))
    pos_mag = max(float(np.max(response)), float(np.max(response_tail)))

    if peak_mode == "abs":
        return np.asarray(max(neg_mag, pos_mag), dtype=float)
    if peak_mode == "pos":
        return np.asarray(pos_mag, dtype=float)
    if peak_mode == "neg":
        return np.asarray(neg_mag, dtype=float)
    return np.asarray([neg_mag, pos_mag], dtype=float)


def _shock_response_spectrum_iir(
    x: np.ndarray,
    *,
    fs: float,
    f0_hz: np.ndarray,
    zeta: float,
    metric: Metric,
    peak_mode: str = "abs",
) -> np.ndarray:
    """Compute a shock-domain response spectrum using recursive SDOF filters.

    This is a private engine intended for future `SRS` / `PVSS` wrappers.

    Contract
    --------
    - `metric='acc'` computes absolute-acceleration shock response.
    - `metric='pv'` computes pseudo-velocity shock response.
    - `peak_mode='abs'|'pos'|'neg'` returns shape `(n_freq,)`.
    - `peak_mode='both'` returns shape `(2, n_freq)` with row order `('neg', 'pos')`.
    """
    x, f0 = _validate_shock_inputs(
        x=x,
        fs=fs,
        f0_hz=f0_hz,
        zeta=zeta,
        metric=metric,
        peak_mode=peak_mode,
    )

    dt = 1.0 / float(fs)
    b, a = _shock_filter_coefficients(f0_hz=f0, zeta=zeta, dt=dt, metric=metric)
    pad_len = _shock_zero_padding_length(fs=fs, f0_hz=f0, zeta=zeta, peak_mode=peak_mode)
    pad = np.zeros(pad_len, dtype=float)

    if peak_mode == "both":
        out = np.zeros((2, f0.size), dtype=float)
    else:
        out = np.zeros(f0.size, dtype=float)

    zi = np.zeros(2, dtype=float)
    for i in range(f0.size):
        resp, zf = lfilter(b[i], a[i], x, zi=zi)
        resp_tail, _ = lfilter(b[i], a[i], pad, zi=zf)
        peak = _extract_shock_peak(resp, resp_tail, peak_mode=peak_mode)
        if peak_mode == "both":
            out[:, i] = peak
        else:
            out[i] = float(peak)

    return out
