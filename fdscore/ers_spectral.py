"""Spectral/random extreme-response spectra from acceleration PSD inputs.

This module adds a PSD-domain ERS workflow alongside the existing
time-domain ``compute_ers_time(...)`` route. The spectral workflow
interprets the ERS as an expected extreme response of a stationary
Gaussian process over a specified duration, rather than as the maximum
observed in a realized time record.
"""

from __future__ import annotations

import numpy as np

from .types import PSDParams, SDOFParams, ERSResult
from .grid import build_frequency_grid
from .validate import (
    ValidationError,
    _bool_flag_or_raise,
    _finite_positive_float_or_raise,
    ers_compat_dict,
    validate_sdof,
)
from .sdof_transfer import build_transfer_psd
from .psd_welch import compute_psd_welch
from ._psd_utils import clip_tiny_negative_psd_or_raise


_EULER_GAMMA = 0.5772156649015329
_EDGE_BANDWIDTH_SCALE = 2.0


def _gaussian_peak_factor_davenport(peak_rate_hz: np.ndarray, duration_s: float) -> np.ndarray:
    """Return the expected Gaussian maximum factor via Davenport/Gumbel."""
    n_peaks = np.maximum(np.asarray(peak_rate_hz, dtype=float) * float(duration_s), np.e)
    u = np.sqrt(2.0 * np.log(n_peaks))
    return u + (_EULER_GAMMA / u)


def _expected_max_from_response_psd(
    *,
    f_hz: np.ndarray,
    response_psd: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Convert one-sided response PSDs into expected extreme maxima."""
    freq = np.asarray(f_hz, dtype=float).reshape(-1)
    psd = np.asarray(response_psd, dtype=float)

    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
    if psd.ndim != 2 or psd.shape[1] != freq.size:
        raise ValidationError("response_psd must be a 1D or 2D array aligned with f_hz.")

    m0 = np.trapezoid(psd, freq, axis=1)
    m2 = np.trapezoid(((2.0 * np.pi * freq) ** 2)[None, :] * psd, freq, axis=1)

    out = np.zeros(psd.shape[0], dtype=float)
    positive = m0 > 0.0
    if not np.any(positive):
        return out

    peak_rate_hz = np.zeros_like(m0)
    peak_rate_hz[positive] = (1.0 / (2.0 * np.pi)) * np.sqrt(np.maximum(m2[positive] / m0[positive], 0.0))
    g = _gaussian_peak_factor_davenport(peak_rate_hz[positive], float(duration_s))
    out[positive] = g * np.sqrt(m0[positive])
    return out


def _extend_psd_high_edge_auto(
    *,
    f_psd_hz: np.ndarray,
    psd_baseacc: np.ndarray,
    fn_hz: float,
    zeta: float,
    nyquist_hz: float,
    bandwidth_scale: float = _EDGE_BANDWIDTH_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a cropped PSD tail with a raised-cosine taper.

    The completion span is proportional to the oscillator bandwidth:

    ``delta_f = bandwidth_scale * zeta * f_n``.

    The added tail starts at the last available PSD point and decays
    smoothly to zero over that span, clipped at ``nyquist_hz``.
    """
    f = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    p = np.asarray(psd_baseacc, dtype=float).reshape(-1)

    if f.size < 2 or f.shape != p.shape:
        raise ValidationError("f_psd_hz and psd_baseacc must be aligned 1D arrays.")
    if float(nyquist_hz) <= float(f[-1]):
        return f, p

    delta_f = float(bandwidth_scale) * float(zeta) * float(fn_hz)
    if delta_f <= 0.0:
        return f, p

    f_stop = min(float(nyquist_hz), float(f[-1]) + delta_f)
    if f_stop <= float(f[-1]):
        return f, p

    df = float(f[1] - f[0])
    extra_f = np.arange(float(f[-1]) + df, f_stop + 0.5 * df, df)
    if extra_f.size == 0:
        return f, p

    x = (extra_f - float(f[-1])) / (f_stop - float(f[-1]))
    taper = 0.5 * (1.0 + np.cos(np.pi * x))
    extra_p = float(p[-1]) * taper
    return np.concatenate([f, extra_f]), np.concatenate([p, extra_p])


def compute_ers_spectral_psd(
    f_psd_hz: np.ndarray,
    psd_baseacc: np.ndarray,
    *,
    duration_s: float,
    sdof: SDOFParams,
    nyquist_hz: float | None = None,
    edge_correction: bool = True,
) -> ERSResult:
    r"""Compute a spectral/random ERS from a one-sided acceleration PSD.

    This workflow interprets the ERS as the expected extreme response of
    a stationary Gaussian process over ``duration_s``. It is therefore a
    different quantity from :func:`fdscore.compute_ers_time`, which
    extracts the maximum observed in a realized time history.

    For each oscillator frequency :math:`f_n`, the response PSD is built
    as

    .. math::

       P_{resp}(f; f_n) = \left| H(f; f_n) \right|^2 P_{base}(f)

    and the response moments

    .. math::

       m_0 = \int P_{resp}(f) \, df

    .. math::

       m_2 = \int (2\pi f)^2 P_{resp}(f) \, df

    are used to estimate the peak rate and the expected Gaussian maximum.

    Parameters
    ----------
    f_psd_hz : numpy.ndarray
        One-sided PSD frequency grid in Hz.
    psd_baseacc : numpy.ndarray
        One-sided base-acceleration PSD defined on ``f_psd_hz``.
    duration_s : float
        Exposure duration associated with the expected extreme response.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    nyquist_hz : float or None, optional
        Original Nyquist limit of the underlying time-history sampling.
        This is used only when ``edge_correction=True`` and the PSD has
        been cropped below that limit.
    edge_correction : bool, optional
        Whether to apply an automatic high-frequency edge correction for
        cropped PSDs. The correction is a per-oscillator raised-cosine
        taper over a bandwidth proportional to ``zeta * f_n`` and is a
        no-op when the PSD already reaches ``nyquist_hz``.

    Returns
    -------
    ERSResult
        Spectral/random ERS evaluated on the oscillator grid defined by
        ``sdof``.

    Notes
    -----
    This route assumes that the PSD is an adequate descriptor of the
    environment and that Gaussian extreme-value statistics are an
    acceptable approximation for the response process.

    The high-frequency edge correction exists because PSD exports are
    often cropped below the original Nyquist frequency for plotting or
    exchange. Near the top of the oscillator grid, that cropping can
    artificially suppress the response moments unless the missing tail is
    reconstructed.

    References
    ----------
    Davenport, A. G. (1964). Note on the Distribution of the Largest
    Value of a Random Function with Application to Gust Loading.
    Lalanne, C. Mechanical Vibration and Shock Analysis, Volume 3
    (Random Vibration).
    """
    validate_sdof(sdof)
    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")

    duration_s = _finite_positive_float_or_raise(duration_s, field="duration_s")
    edge_correction = _bool_flag_or_raise(edge_correction, field="edge_correction")
    nyquist_val = None if nyquist_hz is None else _finite_positive_float_or_raise(nyquist_hz, field="nyquist_hz")

    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    Pyy = np.asarray(psd_baseacc, dtype=float).reshape(-1)
    if f_psd.size < 2 or Pyy.size < 2 or f_psd.shape != Pyy.shape:
        raise ValidationError("f_psd_hz and psd_baseacc must be 1D arrays of the same length >= 2.")
    if not (np.all(np.isfinite(f_psd)) and np.all(np.isfinite(Pyy))):
        raise ValidationError("PSD inputs must be finite.")
    if not np.all(np.diff(f_psd) > 0):
        raise ValidationError("f_psd_hz must be strictly increasing.")
    if float(f_psd[0]) < 0.0:
        raise ValidationError("f_psd_hz must be >= 0.")
    Pyy = clip_tiny_negative_psd_or_raise(Pyy, label="psd_baseacc")

    f0 = build_frequency_grid(sdof)
    zeta = 1.0 / (2.0 * float(sdof.q))
    H = build_transfer_psd(f_psd_hz=f_psd, f0_hz=f0, zeta=zeta, metric=sdof.metric)
    P_resp = (np.abs(H) ** 2) * Pyy[None, :]
    response = _expected_max_from_response_psd(f_hz=f_psd, response_psd=P_resp, duration_s=duration_s)

    edge_applied = False
    corrected_count = 0
    if edge_correction and nyquist_val is not None and float(f_psd[-1]) < nyquist_val:
        corrected = response.copy()
        for i, fn in enumerate(f0):
            f_ext, P_ext = _extend_psd_high_edge_auto(
                f_psd_hz=f_psd,
                psd_baseacc=Pyy,
                fn_hz=float(fn),
                zeta=float(zeta),
                nyquist_hz=float(nyquist_val),
            )
            if f_ext.size == f_psd.size:
                continue
            H_ext = build_transfer_psd(f_psd_hz=f_ext, f0_hz=np.asarray([fn], dtype=float), zeta=zeta, metric=sdof.metric)
            P_resp_ext = (np.abs(H_ext) ** 2) * P_ext[None, :]
            corrected[i] = _expected_max_from_response_psd(
                f_hz=f_ext,
                response_psd=P_resp_ext,
                duration_s=duration_s,
            )[0]
            corrected_count += 1
        response = corrected
        edge_applied = corrected_count > 0

    meta = {
        "compat": ers_compat_dict(
            metric=sdof.metric,
            q=sdof.q,
            peak_mode="expected_gaussian_max",
            engine="spectral_random_psd",
            ers_kind="random_extreme_response_spectrum",
        ),
        "metric": sdof.metric,
        "q": float(sdof.q),
        "zeta": float(zeta),
        "peak_mode": "expected_gaussian_max",
        "provenance": {
            "source": "compute_ers_spectral_psd",
            "duration_s": float(duration_s),
            "peak_model": "gaussian_davenport",
            "edge_correction_enabled": bool(edge_correction),
            "edge_correction_applied": bool(edge_applied),
            "edge_correction_mode": "auto_bandwidth_taper",
            "edge_bandwidth_scale": float(_EDGE_BANDWIDTH_SCALE),
            "edge_corrected_oscillator_count": int(corrected_count),
            "input_psd_fmax_hz": float(f_psd[-1]),
            "nyquist_hz": None if nyquist_val is None else float(nyquist_val),
        },
    }
    return ERSResult(f=f0, response=np.asarray(response, dtype=float), meta=meta)


def compute_ers_spectral_time(
    x: np.ndarray,
    fs: float,
    *,
    sdof: SDOFParams,
    psd: PSDParams,
    duration_s: float | None = None,
    edge_correction: bool = True,
) -> ERSResult:
    r"""Compute a spectral/random ERS from a time history through Welch.

    This convenience route first estimates a one-sided acceleration PSD
    with :func:`fdscore.compute_psd_welch` and then delegates to
    :func:`fdscore.compute_ers_spectral_psd`.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional base-acceleration time history.
    fs : float
        Sampling rate in Hz.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    psd : PSDParams
        PSD-estimation settings passed to ``compute_psd_welch(...)``.
    duration_s : float or None, optional
        Exposure duration associated with the expected extreme response.
        If ``None``, uses ``len(x) / fs``.
    edge_correction : bool, optional
        Whether to enable the automatic high-frequency edge correction
        when the internally estimated PSD is cropped below Nyquist.

    Returns
    -------
    ERSResult
        Spectral/random ERS evaluated from the internally estimated PSD.

    Notes
    -----
    This function is a convenience wrapper. It is not equivalent to
    :func:`fdscore.compute_ers_time`, which operates on the realized time
    history directly and extracts the maximum observed response.
    """
    if not bool(psd.onesided):
        raise ValidationError("compute_ers_spectral_time requires PSDParams.onesided=True.")

    fs = _finite_positive_float_or_raise(fs, field="fs")
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValidationError("x must be a 1D array.")
    if duration_s is None:
        duration_s = float(x_arr.size) / float(fs)
    else:
        duration_s = _finite_positive_float_or_raise(duration_s, field="duration_s")

    f_psd, Pyy = compute_psd_welch(x_arr, fs=fs, psd=psd)
    out = compute_ers_spectral_psd(
        f_psd_hz=f_psd,
        psd_baseacc=Pyy,
        duration_s=float(duration_s),
        sdof=sdof,
        nyquist_hz=float(fs) / 2.0,
        edge_correction=edge_correction,
    )

    meta = dict(out.meta)
    provenance = dict(meta.get("provenance", {}))
    provenance["source"] = "compute_ers_spectral_time"
    provenance["fs_hz"] = float(fs)
    provenance["psd_method"] = str(psd.method)
    provenance["psd_window"] = str(psd.window)
    provenance["psd_nperseg"] = None if psd.nperseg is None else int(psd.nperseg)
    provenance["psd_noverlap"] = None if psd.noverlap is None else int(psd.noverlap)
    provenance["psd_detrend"] = str(psd.detrend)
    provenance["psd_fmin_hz"] = None if psd.fmin is None else float(psd.fmin)
    provenance["psd_fmax_hz"] = None if psd.fmax is None else float(psd.fmax)
    meta["provenance"] = provenance
    return ERSResult(f=np.asarray(out.f, dtype=float), response=np.asarray(out.response, dtype=float), meta=meta)
