from __future__ import annotations

import numpy as np

from ._shock_iir import _shock_response_spectrum_iir
from .grid import build_frequency_grid
from .types import ERSResult, SDOFParams, ShockSpectrumPair
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

    if peak_mode not in ("abs", "pos", "neg", "both"):
        raise ValidationError("peak_mode must be one of: 'abs', 'pos', 'neg', 'both'.")

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


def _build_shock_result(
    *,
    f0: np.ndarray,
    response: np.ndarray,
    metric: str,
    q: float,
    zeta: float,
    peak_mode: str,
    ers_kind: str,
    source: str,
    detrend: str,
) -> ERSResult:
    meta = {
        "compat": ers_compat_dict(
            metric=metric,
            q=q,
            peak_mode=peak_mode,
            engine="shock_iir",
            ers_kind=ers_kind,
        ),
        "metric": metric,
        "q": float(q),
        "zeta": float(zeta),
        "peak_mode": peak_mode,
        "provenance": {
            "source": source,
            "detrend": detrend,
            "engine": "recursive_iir",
        },
    }
    return ERSResult(f=np.asarray(f0, dtype=float), response=np.asarray(response, dtype=float), meta=meta)


def compute_srs_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    *,
    detrend: str = "mean",
    strict_nyquist: bool = True,
    peak_mode: str = "abs",
) -> ERSResult | ShockSpectrumPair:
    """Compute a shock response spectrum (SRS) from a base-acceleration time history.

    Notes
    -----
    - This wrapper uses the dedicated recursive shock engine.
    - `sdof.metric` must be ``"acc"``.
    - Public sidedness supports ``abs``, ``pos``, ``neg``, and ``both``.
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

    if peak_mode == "both":
        neg = _build_shock_result(
            f0=f0,
            response=response[0],
            metric="acc",
            q=sdof.q,
            zeta=zeta,
            peak_mode="neg",
            ers_kind="shock_response_spectrum",
            source="compute_srs_time",
            detrend=detrend,
        )
        pos = _build_shock_result(
            f0=f0,
            response=response[1],
            metric="acc",
            q=sdof.q,
            zeta=zeta,
            peak_mode="pos",
            ers_kind="shock_response_spectrum",
            source="compute_srs_time",
            detrend=detrend,
        )
        return ShockSpectrumPair(
            neg=neg,
            pos=pos,
            meta={
                "metric": "acc",
                "q": float(sdof.q),
                "zeta": float(zeta),
                "peak_mode": "both",
                "ers_kind": "shock_response_spectrum",
                "provenance": {
                    "source": "compute_srs_time",
                    "detrend": detrend,
                    "engine": "recursive_iir",
                },
            },
        )

    return _build_shock_result(
        f0=f0,
        response=response,
        metric="acc",
        q=sdof.q,
        zeta=zeta,
        peak_mode=peak_mode,
        ers_kind="shock_response_spectrum",
        source="compute_srs_time",
        detrend=detrend,
    )


def compute_pvss_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    *,
    detrend: str = "mean",
    strict_nyquist: bool = True,
    peak_mode: str = "abs",
) -> ERSResult | ShockSpectrumPair:
    """Compute a pseudo-velocity shock spectrum (PVSS) from a base-acceleration time history.

    Notes
    -----
    - This wrapper uses the dedicated recursive shock engine.
    - `sdof.metric` must be ``"pv"``.
    - Public sidedness supports ``abs``, ``pos``, ``neg``, and ``both``.
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

    if peak_mode == "both":
        neg = _build_shock_result(
            f0=f0,
            response=response[0],
            metric="pv",
            q=sdof.q,
            zeta=zeta,
            peak_mode="neg",
            ers_kind="pseudo_velocity_shock_spectrum",
            source="compute_pvss_time",
            detrend=detrend,
        )
        pos = _build_shock_result(
            f0=f0,
            response=response[1],
            metric="pv",
            q=sdof.q,
            zeta=zeta,
            peak_mode="pos",
            ers_kind="pseudo_velocity_shock_spectrum",
            source="compute_pvss_time",
            detrend=detrend,
        )
        return ShockSpectrumPair(
            neg=neg,
            pos=pos,
            meta={
                "metric": "pv",
                "q": float(sdof.q),
                "zeta": float(zeta),
                "peak_mode": "both",
                "ers_kind": "pseudo_velocity_shock_spectrum",
                "provenance": {
                    "source": "compute_pvss_time",
                    "detrend": detrend,
                    "engine": "recursive_iir",
                },
            },
        )

    return _build_shock_result(
        f0=f0,
        response=response,
        metric="pv",
        q=sdof.q,
        zeta=zeta,
        peak_mode=peak_mode,
        ers_kind="pseudo_velocity_shock_spectrum",
        source="compute_pvss_time",
        detrend=detrend,
    )
