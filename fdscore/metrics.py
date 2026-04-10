from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import PSDResult, PSDMetricsResult
from .validate import ValidationError
from ._psd_utils import clip_tiny_negative_psd_or_raise

G0 = 9.80665
EULER_GAMMA = 0.5772156649015329
DEFAULT_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (5.0, 20.0),
    (20.0, 80.0),
    (80.0, 200.0),
    (200.0, 400.0),
)


def _integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size == 0 or x.size == 0:
        return 0.0

    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is not None:
        return float(trapz_fn(y, x))

    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    return float(np.sum(avg * dx))


def _rms_from_psd(psd: np.ndarray, freq_hz: np.ndarray) -> float:
    area = _integrate_trapz(np.clip(psd, 0.0, None), freq_hz)
    return float(np.sqrt(max(area, 0.0)))


def _spectral_moment(psd: np.ndarray, freq_hz: np.ndarray, order: int) -> float:
    w = 2.0 * np.pi * freq_hz
    return _integrate_trapz((w**order) * np.clip(psd, 0.0, None), freq_hz)


def _gaussian_peak_statistics(psd: np.ndarray, freq_hz: np.ndarray, duration_s: float) -> tuple[float, float, float, bool, float]:
    m0 = _spectral_moment(psd, freq_hz, order=0)
    m2 = _spectral_moment(psd, freq_hz, order=2)
    if m0 <= 0.0 or m2 <= 0.0 or duration_s <= 0.0:
        return float("nan"), float("nan"), float("nan"), False, float("nan")

    nu0 = (1.0 / (2.0 * np.pi)) * np.sqrt(m2 / m0)
    n_eff_raw = float(nu0) * float(duration_s)
    floor_applied = bool(n_eff_raw < np.e)
    n_eff = max(n_eff_raw, np.e)
    u = np.sqrt(2.0 * np.log(n_eff))
    if u <= 0.0:
        return float("nan"), float(nu0), float(n_eff), floor_applied, float(n_eff_raw)
    peak_factor = float(u + (EULER_GAMMA / u))
    return peak_factor, float(nu0), float(n_eff), floor_applied, float(n_eff_raw)


def _acceleration_to_velocity_displacement_psd(psd_acc_ms2: np.ndarray, freq_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = 2.0 * np.pi * freq_hz
    psd_v = np.zeros_like(psd_acc_ms2, dtype=float)
    psd_d = np.zeros_like(psd_acc_ms2, dtype=float)
    valid = w > 0.0
    psd_v[valid] = psd_acc_ms2[valid] / (w[valid] ** 2)
    psd_d[valid] = psd_acc_ms2[valid] / (w[valid] ** 4)
    return psd_v, psd_d


def _normalize_acc_unit(unit: str) -> str:
    u = str(unit).strip().lower()
    if u in ("g",):
        return "g"
    if u in ("m/s2", "m/s^2"):
        return "m/s2"
    raise ValidationError("Unsupported acceleration unit. Use 'g' or 'm/s2'.")


def _infer_acc_to_m_s2(*, acc_unit: str | None, acc_to_m_s2: float | None) -> tuple[float, str]:
    if acc_to_m_s2 is not None:
        k = float(acc_to_m_s2)
        if not np.isfinite(k) or k <= 0:
            raise ValidationError("acc_to_m_s2 must be finite and > 0.")
        return k, "scale"

    if acc_unit is not None:
        unit = _normalize_acc_unit(acc_unit)
        return (G0 if unit == "g" else 1.0), unit

    raise ValidationError(
        "Acceleration unit is required for PSD metrics. Provide acc_unit='g'|'m/s2' "
        "or acc_to_m_s2."
    )


def _validate_bands_hz(bands_hz: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if len(bands_hz) == 0:
        raise ValidationError("bands_hz must not be empty.")
    out: list[tuple[float, float]] = []
    for lo, hi in bands_hz:
        flo = float(lo)
        fhi = float(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi) and flo >= 0 and fhi > flo):
            raise ValidationError("Each frequency band must satisfy finite 0 <= f_lo < f_hi.")
        out.append((flo, fhi))
    return tuple(out)


def _format_band_edge(freq_hz: float) -> str:
    val = float(freq_hz)
    rounded = round(val)
    if np.isclose(val, float(rounded), rtol=0.0, atol=1e-12):
        return str(int(rounded))
    label = f"{val:.6g}".replace("-", "m").replace(".", "p")
    return label.replace("+", "")


def _make_band_key(f_lo: float, f_hi: float) -> str:
    return f"rms_g_{_format_band_edge(f_lo)}_{_format_band_edge(f_hi)}Hz"


def compute_psd_metrics(
    psd: PSDResult | np.ndarray,
    *,
    f_hz: np.ndarray | None = None,
    duration_s: float | None = None,
    acc_unit: str | None = None,
    acc_to_m_s2: float | None = None,
    bands_hz: Sequence[tuple[float, float]] = DEFAULT_BANDS_HZ,
) -> PSDMetricsResult:
    r"""Compute scalar summary metrics from an acceleration PSD.

    The routine derives RMS, Gaussian peak estimates, velocity and
    displacement metrics, and band-limited RMS values from a one-sided
    acceleration PSD.

    Parameters
    ----------
    psd : PSDResult or numpy.ndarray
        Either a ``PSDResult`` or a one-dimensional PSD array.
    f_hz : numpy.ndarray or None
        Frequency vector in Hz when ``psd`` is passed as a raw array. Must be
        omitted when ``psd`` is a ``PSDResult``.
    duration_s : float or None
        Exposure duration used for Gaussian peak estimation. If ``None``,
        peak-related outputs are returned as ``nan``.
    acc_unit : str or None
        Acceleration unit of the PSD values. Supported values are ``"g"`` and
        ``"m/s2"``.
    acc_to_m_s2 : float or None
        Explicit conversion factor from the PSD acceleration unit to
        :math:`m/s^2`. If provided, overrides ``acc_unit``.
    bands_hz : collections.abc.Sequence
        Sequence of ``(f_lo, f_hi)`` frequency-band pairs used to compute
        band-limited RMS acceleration in ``g``.

    Returns
    -------
    PSDMetricsResult
        Structured set of scalar metrics including acceleration RMS and peak,
        zero-upcrossing rate, effective cycles, velocity metrics, displacement
        metrics, and per-band RMS values.

    Notes
    -----
    RMS values are obtained by integrating the one-sided PSD. Velocity and
    displacement PSDs are derived from acceleration through division by
    :math:`\omega^2` and :math:`\omega^4`, respectively, for nonzero angular
    frequency.

    When ``duration_s`` is provided, the peak factor is estimated with a
    Gaussian extreme-value approximation of Davenport form:

    .. math::

       \nu_0 = \frac{1}{2 \pi} \sqrt{\frac{m_2}{m_0}}

    .. math::

       n_{eff} = \nu_0 T

    .. math::

       u = \sqrt{2 \ln(n_{eff})}

    .. math::

       peak\_factor \approx u + \frac{\gamma_E}{u}

    where :math:`m_0` and :math:`m_2` are spectral moments and
    :math:`\gamma_E` is the Euler-Mascheroni constant. The implementation
    floors :math:`n_{eff}` at :math:`e` to avoid a singular or negative
    argument in the logarithm for very short exposures or very low
    zero-upcrossing rates.

    These peak estimates assume a stationary Gaussian response process. They
    should be interpreted cautiously for strongly non-stationary environments,
    deterministic transients, or markedly non-Gaussian signals.

    References
    ----------
    Davenport, A. G. (1964). "Note on the Distribution of the Largest Value of
        a Random Function with Application to Gust Loading." *Proceedings of
        the Institution of Civil Engineers*, 28(2), 187-196.
    """
    if isinstance(psd, PSDResult):
        if f_hz is not None:
            raise ValidationError("f_hz must be None when psd is a PSDResult.")
        freq = np.asarray(psd.f, dtype=float).reshape(-1)
        p = np.asarray(psd.psd, dtype=float).reshape(-1)
    else:
        if f_hz is None:
            raise ValidationError("f_hz is required when psd is a raw array.")
        freq = np.asarray(f_hz, dtype=float).reshape(-1)
        p = np.asarray(psd, dtype=float).reshape(-1)

    if freq.ndim != 1 or p.ndim != 1 or freq.shape != p.shape or freq.size < 2:
        raise ValidationError("PSD inputs must be 1D arrays of the same length >= 2.")
    if not (np.all(np.isfinite(freq)) and np.all(np.isfinite(p))):
        raise ValidationError("PSD inputs must be finite.")
    if not np.all(np.diff(freq) > 0):
        raise ValidationError("Frequency vector must be strictly increasing.")
    if np.any(freq < 0):
        raise ValidationError("Frequency vector must be >= 0.")

    p = clip_tiny_negative_psd_or_raise(p, label="PSD input")
    bands = _validate_bands_hz(bands_hz)
    band_keys = [_make_band_key(f_lo, f_hi) for f_lo, f_hi in bands]
    if len(set(band_keys)) != len(band_keys):
        raise ValidationError("bands_hz must map to unique output keys in band_rms_g.")

    if duration_s is not None:
        if not np.isfinite(duration_s) or float(duration_s) <= 0.0:
            raise ValidationError("duration_s must be finite and > 0 when provided.")
        dur = float(duration_s)
    else:
        dur = float("nan")

    acc_scale, resolved_unit = _infer_acc_to_m_s2(acc_unit=acc_unit, acc_to_m_s2=acc_to_m_s2)
    psd_acc_ms2 = p * (float(acc_scale) ** 2)
    psd_acc_g2 = psd_acc_ms2 / (G0**2)

    rms_acc_m_s2 = _rms_from_psd(psd_acc_ms2, freq)
    rms_acc_g = rms_acc_m_s2 / G0

    if np.isfinite(dur):
        peak_factor, nu0, n_eff, n_eff_floor_applied, n_eff_raw = _gaussian_peak_statistics(psd_acc_ms2, freq, dur)
    else:
        peak_factor, nu0, n_eff, n_eff_floor_applied, n_eff_raw = float("nan"), float("nan"), float("nan"), False, float("nan")
    peak_acc_m_s2 = float(rms_acc_m_s2 * peak_factor) if np.isfinite(peak_factor) else float("nan")
    peak_acc_g = peak_acc_m_s2 / G0 if np.isfinite(peak_acc_m_s2) else float("nan")

    psd_v, psd_d = _acceleration_to_velocity_displacement_psd(psd_acc_ms2, freq)
    rms_vel_m_s = _rms_from_psd(psd_v, freq)
    rms_disp_m = _rms_from_psd(psd_d, freq)

    if np.isfinite(dur):
        peak_factor_v, nu0_v, n_eff_v, n_eff_floor_applied_v, n_eff_raw_v = _gaussian_peak_statistics(psd_v, freq, dur)
        peak_factor_d, nu0_d, n_eff_d, n_eff_floor_applied_d, n_eff_raw_d = _gaussian_peak_statistics(psd_d, freq, dur)
    else:
        peak_factor_v, nu0_v, n_eff_v, n_eff_floor_applied_v, n_eff_raw_v = float("nan"), float("nan"), float("nan"), False, float("nan")
        peak_factor_d, nu0_d, n_eff_d, n_eff_floor_applied_d, n_eff_raw_d = float("nan"), float("nan"), float("nan"), False, float("nan")

    peak_vel_m_s = float(rms_vel_m_s * peak_factor_v) if np.isfinite(peak_factor_v) else float("nan")
    peak_disp_m = float(rms_disp_m * peak_factor_d) if np.isfinite(peak_factor_d) else float("nan")
    peak_disp_mm = float(peak_disp_m * 1000.0) if np.isfinite(peak_disp_m) else float("nan")
    disp_pk_pk_mm = float(2.0 * peak_disp_mm) if np.isfinite(peak_disp_mm) else float("nan")

    band_rms_g: dict[str, float] = {}
    band_coverage: dict[str, dict[str, float | bool | None]] = {}
    for f_lo, f_hi in bands:
        key = _make_band_key(f_lo, f_hi)
        mask = (freq >= f_lo) & (freq <= f_hi)
        if np.any(mask):
            freq_band = freq[mask]
            band_rms_g[key] = _rms_from_psd(psd_acc_g2[mask], freq_band)
            covered_lo = float(freq_band[0])
            covered_hi = float(freq_band[-1])
            partial = bool((covered_lo > (f_lo + 1e-12)) or (covered_hi < (f_hi - 1e-12)))
            coverage_fraction = float(max(covered_hi - covered_lo, 0.0) / (f_hi - f_lo))
            band_coverage[key] = {
                "requested_hz": (float(f_lo), float(f_hi)),
                "covered_hz": (covered_lo, covered_hi),
                "has_data": True,
                "partial_coverage": partial,
                "coverage_fraction": coverage_fraction,
            }
        else:
            band_rms_g[key] = float("nan")
            band_coverage[key] = {
                "requested_hz": (float(f_lo), float(f_hi)),
                "covered_hz": None,
                "has_data": False,
                "partial_coverage": False,
                "coverage_fraction": 0.0,
            }

    out_meta = {
        "source": "compute_psd_metrics",
        "input_acc_unit": resolved_unit,
        "acc_to_m_s2": float(acc_scale),
        "duration_s": dur,
        "bands_hz": [tuple(map(float, b)) for b in bands],
        "band_coverage": band_coverage,
        "peak_statistics": {
            "acc": {
                "zero_upcrossing_hz": float(nu0),
                "effective_cycles_raw": float(n_eff_raw),
                "effective_cycles_used": float(n_eff),
                "n_eff_floor_applied": bool(n_eff_floor_applied),
            },
            "vel": {
                "zero_upcrossing_hz": float(nu0_v),
                "effective_cycles_raw": float(n_eff_raw_v),
                "effective_cycles_used": float(n_eff_v),
                "n_eff_floor_applied": bool(n_eff_floor_applied_v),
            },
            "disp": {
                "zero_upcrossing_hz": float(nu0_d),
                "effective_cycles_raw": float(n_eff_raw_d),
                "effective_cycles_used": float(n_eff_d),
                "n_eff_floor_applied": bool(n_eff_floor_applied_d),
            },
        },
    }

    return PSDMetricsResult(
        rms_acc_g=float(rms_acc_g),
        rms_acc_m_s2=float(rms_acc_m_s2),
        peak_acc_g=float(peak_acc_g),
        peak_acc_m_s2=float(peak_acc_m_s2),
        peak_factor=float(peak_factor),
        zero_upcrossing_hz=float(nu0),
        effective_cycles=float(n_eff),
        rms_vel_m_s=float(rms_vel_m_s),
        peak_vel_m_s=float(peak_vel_m_s),
        rms_disp_mm=float(rms_disp_m * 1000.0),
        peak_disp_mm=float(peak_disp_mm),
        disp_pk_pk_mm=float(disp_pk_pk_mm),
        band_rms_g=band_rms_g,
        meta=out_meta,
    )
