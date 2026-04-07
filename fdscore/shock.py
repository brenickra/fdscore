from __future__ import annotations

import numpy as np

from ._shock_iir import _shock_response_spectrum_iir
from .grid import build_frequency_grid
from .types import ERSResult, SDOFParams
from .validate import ValidationError, ers_compat_dict, validate_nyquist, validate_sdof


def _preprocess_shock_signal(x: np.ndarray, *, detrend: str) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()

    if detrend == "linear":
        n = y.size
        if n > 1:
            t = np.arange(n, dtype=float)
            p = np.polyfit(t, y, 1)
            y -= p[0] * t + p[1]
    elif detrend == "mean":
        y -= float(np.mean(y))
    elif detrend == "median":
        y -= float(np.median(y))
    elif detrend != "none":
        raise ValidationError("detrend must be one of: 'linear', 'mean', 'median', 'none'.")

    return y


def _validate_shock_wrapper_inputs(
    *,
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    expected_metric: str,
    detrend: str,
    peak_mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    validate_sdof(sdof)

    if sdof.metric != expected_metric:
        raise ValidationError(f"sdof.metric must be '{expected_metric}' for this shock wrapper.")

    if peak_mode not in ("abs", "pos", "neg"):
        raise ValidationError("peak_mode must be one of: 'abs', 'pos', 'neg'.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValidationError("x must be a 1D array with length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("x must contain only finite values.")

    if not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValidationError("fs must be finite and > 0.")

    f0 = build_frequency_grid(sdof)
    zeta = 1.0 / (2.0 * float(sdof.q))

    _preprocess_shock_signal(np.zeros(4, dtype=float), detrend=detrend)
    return x, f0, zeta


def compute_srs_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    *,
    detrend: str = "mean",
    strict_nyquist: bool = True,
    peak_mode: str = "abs",
) -> ERSResult:
    """Compute a shock response spectrum (SRS) from a base-acceleration time history.

    Notes
    -----
    - This wrapper uses the dedicated recursive shock engine.
    - `sdof.metric` must be ``"acc"``.
    - Public sidedness is currently limited to ``abs``, ``pos``, and ``neg``.
    """
    x, f0, zeta = _validate_shock_wrapper_inputs(
        x=x,
        fs=fs,
        sdof=sdof,
        expected_metric="acc",
        detrend=detrend,
        peak_mode=peak_mode,
    )
    f0 = validate_nyquist(f0, fs=float(fs), strict=strict_nyquist)
    y = _preprocess_shock_signal(x, detrend=detrend)

    response = _shock_response_spectrum_iir(
        y,
        fs=float(fs),
        f0_hz=f0,
        zeta=float(zeta),
        metric="acc",
        peak_mode=peak_mode,
    )

    meta = {
        "compat": ers_compat_dict(
            metric="acc",
            q=sdof.q,
            peak_mode=peak_mode,
            engine="shock_iir",
            ers_kind="shock_response_spectrum",
        ),
        "metric": "acc",
        "q": float(sdof.q),
        "zeta": float(zeta),
        "peak_mode": peak_mode,
        "provenance": {
            "source": "compute_srs_time",
            "detrend": detrend,
            "engine": "recursive_iir",
        },
    }
    return ERSResult(f=f0, response=np.asarray(response, dtype=float), meta=meta)


def compute_pvss_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    *,
    detrend: str = "mean",
    strict_nyquist: bool = True,
    peak_mode: str = "abs",
) -> ERSResult:
    """Compute a pseudo-velocity shock spectrum (PVSS) from a base-acceleration time history.

    Notes
    -----
    - This wrapper uses the dedicated recursive shock engine.
    - `sdof.metric` must be ``"pv"``.
    - Public sidedness is currently limited to ``abs``, ``pos``, and ``neg``.
    """
    x, f0, zeta = _validate_shock_wrapper_inputs(
        x=x,
        fs=fs,
        sdof=sdof,
        expected_metric="pv",
        detrend=detrend,
        peak_mode=peak_mode,
    )
    f0 = validate_nyquist(f0, fs=float(fs), strict=strict_nyquist)
    y = _preprocess_shock_signal(x, detrend=detrend)

    response = _shock_response_spectrum_iir(
        y,
        fs=float(fs),
        f0_hz=f0,
        zeta=float(zeta),
        metric="pv",
        peak_mode=peak_mode,
    )

    meta = {
        "compat": ers_compat_dict(
            metric="pv",
            q=sdof.q,
            peak_mode=peak_mode,
            engine="shock_iir",
            ers_kind="pseudo_velocity_shock_spectrum",
        ),
        "metric": "pv",
        "q": float(sdof.q),
        "zeta": float(zeta),
        "peak_mode": peak_mode,
        "provenance": {
            "source": "compute_pvss_time",
            "detrend": detrend,
            "engine": "recursive_iir",
        },
    }
    return ERSResult(f=f0, response=np.asarray(response, dtype=float), meta=meta)
