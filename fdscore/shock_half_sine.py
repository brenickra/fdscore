from __future__ import annotations

import numpy as np

from .types import ERSResult, HalfSinePulse
from .validate import ValidationError, parse_ers_compat, validate_frequency_vector


def _half_sine_pvss_amp_factor(zeta: float) -> float:
    if not np.isfinite(zeta) or float(zeta) < 0.0 or float(zeta) >= 1.0:
        raise ValidationError("zeta must be finite and satisfy 0 <= zeta < 1 for half-sine fitting.")

    imag_a = float(np.sqrt(max(0.0, 1.0 - float(zeta) ** 2)))
    if imag_a == 0.0:
        raise ValidationError("half-sine fitting requires zeta < 1.")

    t_max = (2.0 / imag_a) * float(np.arctan2(imag_a, 1.0 + float(zeta)))
    return float(np.exp(-float(zeta) * t_max) * np.sin(imag_a * t_max) / imag_a)


def fit_half_sine_to_pvss(
    pvss: ERSResult,
    *,
    polarity: str = "pos",
) -> HalfSinePulse:
    """Fit a half-sine acceleration pulse whose PVSS envelopes the input PVSS.

    Notes
    -----
    - `pvss` must be a one-sided 1D `ERSResult` produced by `compute_pvss_time(...)`
      or an otherwise compatible PVSS workflow.
    - The returned amplitude is in the same acceleration units used to generate
      the input time history that produced `pvss`.
    """
    if polarity not in ("pos", "neg"):
        raise ValidationError("polarity must be 'pos' or 'neg'.")
    if not isinstance(pvss, ERSResult):
        raise ValidationError("pvss must be an ERSResult.")

    compat = parse_ers_compat((pvss.meta or {}).get("compat", {}))
    if compat.metric != "pv":
        raise ValidationError("pvss compat metric must be 'pv'.")
    if compat.ers_kind != "pseudo_velocity_shock_spectrum":
        raise ValidationError("pvss must have ers_kind 'pseudo_velocity_shock_spectrum'.")

    f = np.asarray(pvss.f, dtype=float)
    r = np.asarray(pvss.response, dtype=float)
    validate_frequency_vector(f)
    if r.ndim != 1 or r.shape != f.shape:
        raise ValidationError("pvss.response must be a 1D array with the same shape as pvss.f.")
    if not np.all(np.isfinite(r)):
        raise ValidationError("pvss.response must contain only finite values.")
    if np.any(r < 0.0):
        raise ValidationError("pvss.response must be non-negative for half-sine fitting.")

    max_idx = int(np.argmax(r))
    max_fp_idx = int(np.argmax(r * f))
    max_pvss = float(r[max_idx])
    max_f_pvss = float((r * f)[max_fp_idx])
    if max_pvss <= 0.0 or max_f_pvss <= 0.0:
        raise ValidationError("pvss must contain at least one strictly positive response value.")

    zeta = float((pvss.meta or {}).get("zeta", 1.0 / (2.0 * float(compat.q))))
    amp_factor = _half_sine_pvss_amp_factor(zeta)

    amplitude = float(2.0 * np.pi * max_f_pvss)
    duration_s = float(max_pvss / (4.0 * amp_factor * max_f_pvss))

    meta = {
        "source": "fit_half_sine_to_pvss",
        "fit_method": "pvss_enveloping_half_sine",
        "fit_from": {
            "peak_mode": compat.peak_mode,
            "q": float(compat.q),
            "zeta": zeta,
            "f_peak_pvss_hz": float(f[max_idx]),
            "f_peak_fpvss_hz": float(f[max_fp_idx]),
            "max_pvss": max_pvss,
            "max_f_pvss": max_f_pvss,
            "amp_factor": amp_factor,
        },
    }
    return HalfSinePulse(
        amplitude=amplitude,
        duration_s=duration_s,
        polarity=polarity,
        meta=meta,
    )


def synthesize_half_sine_pulse(
    pulse: HalfSinePulse,
    fs: float,
    *,
    total_duration_s: float | None = None,
    t_start_s: float = 0.0,
) -> np.ndarray:
    """Synthesize a half-sine acceleration pulse as a 1D time history."""
    if not isinstance(pulse, HalfSinePulse):
        raise ValidationError("pulse must be a HalfSinePulse.")
    if pulse.polarity not in ("pos", "neg"):
        raise ValidationError("pulse.polarity must be 'pos' or 'neg'.")
    if not np.isfinite(pulse.amplitude) or float(pulse.amplitude) <= 0.0:
        raise ValidationError("pulse.amplitude must be finite and > 0.")
    if not np.isfinite(pulse.duration_s) or float(pulse.duration_s) <= 0.0:
        raise ValidationError("pulse.duration_s must be finite and > 0.")
    if not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValidationError("fs must be finite and > 0.")
    if not np.isfinite(t_start_s) or float(t_start_s) < 0.0:
        raise ValidationError("t_start_s must be finite and >= 0.")

    duration_s = float(pulse.duration_s)
    t_start_s = float(t_start_s)
    if total_duration_s is None:
        total_duration_s = t_start_s + 2.0 * duration_s
    if not np.isfinite(total_duration_s) or float(total_duration_s) <= 0.0:
        raise ValidationError("total_duration_s must be finite and > 0.")
    total_duration_s = float(total_duration_s)
    if total_duration_s < t_start_s + duration_s:
        raise ValidationError("total_duration_s must be >= t_start_s + pulse.duration_s.")

    n = int(round(total_duration_s * float(fs)))
    if n < 4:
        raise ValidationError("total_duration_s*fs must yield at least 4 samples.")

    t = np.arange(n, dtype=float) / float(fs)
    x = np.zeros(n, dtype=float)

    mask = (t >= t_start_s) & (t < t_start_s + duration_s)
    if int(np.count_nonzero(mask)) < 2:
        raise ValidationError("pulse.duration_s and fs must yield at least 2 pulse samples.")
    if np.any(mask):
        phase = np.pi * (t[mask] - t_start_s) / duration_s
        x[mask] = float(pulse.signed_amplitude) * np.sin(phase)

    return x
