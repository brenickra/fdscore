"""Shock-spectrum wrappers built on the recursive IIR engine.

This module exposes the public time-domain APIs for computing shock
response spectra (SRS) and pseudo-velocity shock spectra (PVSS) from
base-acceleration signals. It validates user inputs, applies simple
signal preprocessing, and packages one-sided or two-sided outputs with
compatibility metadata.
"""

from __future__ import annotations

import numpy as np

from ._shock_iir import _shock_response_spectrum_iir
from ._shock_signal import preprocess_shock_signal
from .grid import build_frequency_grid
from .types import ERSResult, SDOFParams, ShockSpectrumPair
from .validate import (
    ValidationError,
    _finite_positive_float_or_raise,
    _validate_nyquist_with_info,
    ers_compat_dict,
    validate_sdof,
)


def _validate_shock_wrapper_inputs(
    *,
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    expected_metric: str,
    detrend: str,
    peak_mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
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

    fs = _finite_positive_float_or_raise(fs, field="fs")

    f0 = build_frequency_grid(sdof)
    zeta = 1.0 / (2.0 * float(sdof.q))

    preprocess_shock_signal(np.zeros(4, dtype=float), detrend=detrend)
    return x, f0, zeta, fs


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
    nyquist_info: dict[str, object],
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
            **nyquist_info,
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

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional base-acceleration time history.
    fs : float
        Sampling rate in Hz.
    sdof : SDOFParams
        Oscillator-grid definition. For this wrapper,
        ``sdof.metric`` must be ``"acc"``.
    detrend : {"linear", "mean", "median", "none"}, optional
        Detrending mode applied before the shock-spectrum calculation.
    strict_nyquist : bool, optional
        Whether oscillator frequencies at or above Nyquist should raise
        an error instead of being clipped.
    peak_mode : {"abs", "pos", "neg", "both"}, optional
        Requested peak convention. ``"both"`` returns separate negative
        and positive spectra.

    Returns
    -------
    object
        One-sided results are returned as ``ERSResult``. When
        ``peak_mode="both"``, the function returns a
        ``ShockSpectrumPair`` containing the negative and positive sides.

    Notes
    -----
    This wrapper uses the dedicated recursive shock engine implemented in
    :mod:`fdscore._shock_iir`.

    The returned spectrum is tagged with
    ``ers_kind="shock_response_spectrum"`` so that downstream envelope
    and compatibility operations can distinguish it from generic ERS or
    PVSS results.
    """
    x, f0, zeta, fs = _validate_shock_wrapper_inputs(
        x=x,
        fs=fs,
        sdof=sdof,
        expected_metric="acc",
        detrend=detrend,
        peak_mode=peak_mode,
    )
    f0, nyquist_info = _validate_nyquist_with_info(f0, fs=fs, strict=strict_nyquist)
    y = preprocess_shock_signal(x, detrend=detrend)

    response = _shock_response_spectrum_iir(
        y,
        fs=fs,
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
            nyquist_info=nyquist_info,
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
            nyquist_info=nyquist_info,
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
                    **nyquist_info,
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
        nyquist_info=nyquist_info,
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

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional base-acceleration time history.
    fs : float
        Sampling rate in Hz.
    sdof : SDOFParams
        Oscillator-grid definition. For this wrapper,
        ``sdof.metric`` must be ``"pv"``.
    detrend : {"linear", "mean", "median", "none"}, optional
        Detrending mode applied before the shock-spectrum calculation.
    strict_nyquist : bool, optional
        Whether oscillator frequencies at or above Nyquist should raise
        an error instead of being clipped.
    peak_mode : {"abs", "pos", "neg", "both"}, optional
        Requested peak convention. ``"both"`` returns separate negative
        and positive spectra.

    Returns
    -------
    object
        One-sided results are returned as ``ERSResult``. When
        ``peak_mode="both"``, the function returns a
        ``ShockSpectrumPair`` containing the negative and positive sides.

    Notes
    -----
    This wrapper uses the dedicated recursive shock engine implemented in
    :mod:`fdscore._shock_iir`.

    The returned spectrum is tagged with
    ``ers_kind="pseudo_velocity_shock_spectrum"`` so that downstream
    tooling can distinguish PVSS results from generic ERS and classical
    acceleration-based SRS.
    """
    x, f0, zeta, fs = _validate_shock_wrapper_inputs(
        x=x,
        fs=fs,
        sdof=sdof,
        expected_metric="pv",
        detrend=detrend,
        peak_mode=peak_mode,
    )
    f0, nyquist_info = _validate_nyquist_with_info(f0, fs=fs, strict=strict_nyquist)
    y = preprocess_shock_signal(x, detrend=detrend)

    response = _shock_response_spectrum_iir(
        y,
        fs=fs,
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
            nyquist_info=nyquist_info,
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
            nyquist_info=nyquist_info,
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
                    **nyquist_info,
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
        nyquist_info=nyquist_info,
    )
