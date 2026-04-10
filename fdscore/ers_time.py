from __future__ import annotations

import numpy as np

from .types import SDOFParams, ERSResult, FDSTimePlan
from .grid import build_frequency_grid
from .validate import (
    ValidationError,
    _finite_positive_float_or_raise,
    _validate_nyquist_with_info,
    ers_compat_dict,
    validate_sdof,
)
from .preprocess import preprocess_signal
from .sdof_transfer import build_transfer_matrix
from .fds_time import _validate_plan_compatibility


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
) -> ERSResult:
    """Compute time-domain ERS by reconstructing SDOF responses in the FFT domain.

    Notes
    -----
    - The ERS is tied to the selected `sdof.metric`.
    - The current implementation supports `peak_mode="abs"` only.
    - A compatible `FDSTimePlan` can be reused because it stores only transfer data
      for a fixed `(fs, n_samples, sdof.metric, q, f-grid)` configuration.
    """
    validate_sdof(sdof)

    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")
    if detrend not in ("linear", "mean", "none"):
        raise ValidationError("detrend must be one of: 'linear', 'mean', 'none'.")
    if peak_mode != "abs":
        raise ValidationError("peak_mode must be 'abs'.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError("batch_size must be an int > 0.")
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

    if plan is None:
        H = build_transfer_matrix(fs=fs, n=int(y.size), f0_hz=f0, zeta=float(zeta), metric=sdof.metric)
    else:
        H = _validate_plan_compatibility(
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

    meta = {
        "compat": ers_compat_dict(metric=sdof.metric, q=sdof.q, peak_mode=peak_mode, engine="time_peak_fft"),
        "metric": sdof.metric,
        "q": float(sdof.q),
        "zeta": float(zeta),
        "peak_mode": peak_mode,
        "provenance": {
            "source": "compute_ers_time",
            "detrend": detrend,
            "batch_size": int(batch_size),
            "transfer_plan": bool(plan is not None),
            **nyquist_info,
        },
    }
    return ERSResult(f=f0, response=np.asarray(response, dtype=float), meta=meta)
