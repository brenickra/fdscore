"""Time-domain extreme-response spectra via FFT or incremental SDOF evaluation.

This module computes extreme-response spectra from one-dimensional base-
acceleration histories. The public API supports the original FFT-domain
engine as well as an incremental ZOH engine that integrates the
oscillator bank sample-by-sample and tracks peaks online.
"""

from __future__ import annotations

import numpy as np

from .types import SDOFParams, ERSResult, FDSTimePlan
from .grid import build_frequency_grid
from ._time_plan import validate_time_plan_compatibility
from .validate import (
    ValidationError,
    _finite_positive_float_or_raise,
    _validate_nyquist_with_info,
    ers_compat_dict,
    validate_sdof,
)
from .preprocess import preprocess_signal
from .sdof_transfer import build_transfer_matrix
from ._ers_incremental import ers_incremental


def _ers_from_signal_fft(
    y: np.ndarray,
    *,
    fs: float,
    f0: np.ndarray,
    zeta: float,
    metric: str,
    peak_mode: str,
    batch_size: int = 64,
    H: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct oscillator responses in FFT batches and extract peaks."""
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    yf = np.fft.rfft(y)

    if H is None:
        H = build_transfer_matrix(fs=float(fs), n=n, f0_hz=f0, zeta=float(zeta), metric=metric)

    out = np.zeros(H.shape[0], dtype=float)
    bs = max(1, int(batch_size))

    for i0 in range(0, H.shape[0], bs):
        i1 = min(i0 + bs, H.shape[0])
        resp = np.fft.irfft(H[i0:i1] * yf[None, :], n=n, axis=1)

        if peak_mode == "abs":
            out[i0:i1] = np.max(np.abs(resp), axis=1)
        else:  # pragma: no cover
            raise ValidationError("peak_mode must be 'abs'.")

    return out


def compute_ers_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    *,
    detrend: str = "linear",
    strict_nyquist: bool = True,
    batch_size: int = 64,
    peak_mode: str = "abs",
    plan: FDSTimePlan | None = None,
    engine: str = "incremental",
    zoh_r_max: float = 0.2,
) -> ERSResult:
    """Compute time-domain ERS by evaluating an SDOF bank on an input history.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional input time history.
    fs : float
        Sampling rate in Hz.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    detrend : {"linear", "mean", "none"}, optional
        Preprocessing mode applied before the FFT.
    strict_nyquist : bool, optional
        Whether oscillator frequencies at or above Nyquist should raise
        an error instead of being clipped.
    batch_size : int, optional
        Number of oscillator responses reconstructed simultaneously in
        each FFT batch. Only used when ``engine="fft"``.
    peak_mode : {"abs"}, optional
        Peak convention for the ERS. The current implementation supports
        only absolute peaks.
    plan : FDSTimePlan or None, optional
        Optional reusable transfer plan matching the current sampling and
        oscillator configuration. Only used when ``engine="fft"``.
    engine : {"fft", "incremental"}, optional
        Internal computation engine.

        ``"fft"``
            Original FFT-domain engine. Applies the continuous SDOF
            transfer function to ``rfft(x)`` and reconstructs each
            oscillator response with ``irfft`` in batches.

        ``"incremental"`` (default)
            Sample-by-sample SDOF integration using exact ZOH state-
            transition matrices. Peaks are tracked online, so the full
            ``(n_osc, n_samples)`` response matrix is never materialized.
            Oscillators near Nyquist are integrated with adaptive
            upsampling controlled by ``zoh_r_max``.
    zoh_r_max : float, optional
        Maximum tolerated ``f0 / Nyquist_effective`` ratio for the
        incremental ZOH engine. Smaller values reduce the high-frequency
        ZOH attenuation error at the cost of larger upsample factors.
        Ignored when ``engine="fft"``.

    Returns
    -------
    ERSResult
        Extreme-response spectrum evaluated on the validated oscillator
        grid.

    Notes
    -----
    The ERS is tied to the selected ``sdof.metric`` and therefore may
    represent absolute acceleration, relative displacement, relative
    velocity, or pseudo-velocity depending on the chosen SDOF setup.

    A compatible ``FDSTimePlan`` can be reused by the FFT engine because
    it stores only the transfer data for a fixed
    ``(fs, n_samples, metric, q, f-grid)`` configuration. Reuse avoids
    rebuilding the transfer matrix on repeated calls with the same
    sampling contract.

    The computational pipeline is:

    1. Validate the SDOF definition and requested frequency grid.
    2. Enforce or apply Nyquist handling.
    3. Preprocess the input signal.
    4. Evaluate the oscillator bank with the selected engine.
    5. Extract the peak absolute response for each oscillator.

    The incremental engine is designed to match the FFT baseline while
    avoiding the cost of inverse FFT reconstruction for every oscillator.
    Small differences near the top of the frequency grid remain possible
    because the two engines use different discretization schemes
    (continuous FRF on discrete FFT bins versus causal ZOH state
    integration).
    """
    validate_sdof(sdof)

    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")
    if detrend not in ("linear", "mean", "none"):
        raise ValidationError("detrend must be one of: 'linear', 'mean', 'none'.")
    if peak_mode != "abs":
        raise ValidationError("peak_mode must be 'abs'.")
    if engine not in ("fft", "incremental"):
        raise ValidationError("engine must be one of: 'fft', 'incremental'.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError("batch_size must be an int > 0.")
    if not np.isfinite(zoh_r_max) or float(zoh_r_max) <= 0.0 or float(zoh_r_max) >= 1.0:
        raise ValidationError("zoh_r_max must be finite and satisfy 0 < zoh_r_max < 1.")
    fs = _finite_positive_float_or_raise(fs, field="fs")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValidationError("x must be a 1D array with length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("x must contain only finite values.")

    f_requested = build_frequency_grid(sdof)
    f0, nyquist_info = _validate_nyquist_with_info(f_requested, fs=fs, strict=strict_nyquist)
    zeta = 1.0 / (2.0 * float(sdof.q))
    y = preprocess_signal(x, mode=detrend)

    if engine == "incremental":
        response = ers_incremental(
            y,
            fs=fs,
            f0=f0,
            zeta=float(zeta),
            metric=sdof.metric,
            zoh_r_max=float(zoh_r_max),
        )
        engine_tag = "time_peak_incremental_zoh_numba"
        provenance_extra = {
            "engine": engine,
            "zoh_r_max": float(zoh_r_max),
        }
    else:
        if plan is None:
            H = build_transfer_matrix(
                fs=fs,
                n=int(y.size),
                f0_hz=f0,
                zeta=float(zeta),
                metric=sdof.metric,
            )
        else:
            H = validate_time_plan_compatibility(
                plan=plan,
                fs=fs,
                n_samples=int(y.size),
                f0=f0,
                zeta=float(zeta),
                metric=sdof.metric,
            )

        response = _ers_from_signal_fft(
            y,
            fs=fs,
            f0=f0,
            zeta=float(zeta),
            metric=sdof.metric,
            peak_mode=peak_mode,
            batch_size=int(batch_size),
            H=H,
        )
        engine_tag = "time_peak_fft"
        provenance_extra = {
            "engine": engine,
            "batch_size": int(batch_size),
            "transfer_plan": bool(plan is not None),
        }

    meta = {
        "compat": ers_compat_dict(
            metric=sdof.metric,
            q=sdof.q,
            peak_mode=peak_mode,
            engine=engine_tag,
        ),
        "metric": sdof.metric,
        "q": float(sdof.q),
        "zeta": float(zeta),
        "peak_mode": peak_mode,
        "provenance": {
            "source": "compute_ers_time",
            "detrend": detrend,
            **provenance_extra,
            **nyquist_info,
        },
    }
    return ERSResult(f=f0, response=np.asarray(response, dtype=float), meta=meta)
