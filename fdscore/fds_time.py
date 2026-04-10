from __future__ import annotations

import numpy as np

from .types import SNParams, SDOFParams, FDSResult, FDSTimePlan
from .grid import build_frequency_grid
from ._time_plan import validate_time_plan_compatibility
from .validate import (
    ValidationError,
    _finite_positive_float_or_raise,
    _validate_nyquist_with_info,
    compat_dict,
    resolve_p_scale,
    validate_nyquist,
    validate_sdof,
    validate_sn,
)
from .preprocess import preprocess_signal
from .sdof_transfer import build_transfer_matrix
from .rainflow_damage import miner_damage_from_matrix


def _fds_from_signal_fft(
    y: np.ndarray,
    *,
    fs: float,
    f0: np.ndarray,
    zeta: float,
    metric: str,
    k: float,
    c: float,
    p_scale: float,
    batch_size: int = 64,
    amplitude_from_range: bool = True,
    H: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    yf = np.fft.rfft(y)

    if H is None:
        H = build_transfer_matrix(fs=float(fs), n=n, f0_hz=f0, zeta=float(zeta), metric=metric)  # type: ignore[arg-type]

    out = np.zeros(H.shape[0], dtype=float)
    bs = max(1, int(batch_size))

    for i0 in range(0, H.shape[0], bs):
        i1 = min(i0 + bs, H.shape[0])
        resp = np.fft.irfft(H[i0:i1] * yf[None, :], n=n, axis=1)
        resp *= float(p_scale)

        dmg_batch = miner_damage_from_matrix(
            resp,
            k=float(k),
            c=float(c),
            amplitude_from_range=bool(amplitude_from_range),
        )
        out[i0:i1] = np.asarray(dmg_batch, dtype=float)

    return out


def prepare_fds_time_plan(
    *,
    fs: float,
    n_samples: int,
    sdof: SDOFParams,
    strict_nyquist: bool = True,
) -> FDSTimePlan:
    """Precompute and store transfer data for repeated `compute_fds_time` calls.

    A plan avoids rebuilding the FFT-domain transfer matrix for every repeated call
    with the same `(fs, n_samples, sdof)` configuration.

    Memory tradeoff
    ---------------
    The full transfer matrix `H` is stored explicitly as `complex128` with shape
    `(len(f0), n_fft_bins)`. Memory therefore scales approximately as:

        len(f0) * n_fft_bins * 16 bytes

    For example, 400 oscillators and a 4 s signal at 1 kHz correspond to about
    12 MB for the plan matrix alone.
    """
    validate_sdof(sdof)

    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")
    fs = _finite_positive_float_or_raise(fs, field="fs")
    if not isinstance(n_samples, (int, np.integer)) or int(n_samples) < 4:
        raise ValidationError("n_samples must be an integer >= 4.")

    f0 = build_frequency_grid(sdof)
    f0 = validate_nyquist(f0, fs=fs, strict=bool(strict_nyquist))
    zeta = 1.0 / (2.0 * float(sdof.q))

    H = build_transfer_matrix(
        fs=fs,
        n=int(n_samples),
        f0_hz=f0,
        zeta=float(zeta),
        metric=sdof.metric,
    )
    return FDSTimePlan(
        fs=fs,
        n_samples=int(n_samples),
        f=np.asarray(f0, dtype=float),
        zeta=float(zeta),
        metric=sdof.metric,
        H=np.asarray(H),
    )


def compute_fds_time(
    x: np.ndarray,
    fs: float,
    sn: SNParams,
    sdof: SDOFParams,
    *,
    p_scale: float | None = None,
    detrend: str = "linear",
    strict_nyquist: bool = True,
    batch_size: int = 64,
    plan: FDSTimePlan | None = None,
) -> FDSResult:
    """Compute time-domain FDS (Miner damage spectrum) for SDOF responses.

    The result embeds a compatibility signature in `meta["compat"]`
    and accepts an optional precomputed transfer `plan` for repeated calls.

    Notes
    -----
    `p_scale` multiplies the oscillator response before rainflow/Miner damage
    counting. For fixed `slope_k`, the absolute damage level scales globally with:

        p_scale**k / (ref_cycles * ref_stress**k)

    As a consequence:
    - `p_scale`, `ref_stress`, and `ref_cycles` change the magnitude of `damage(f)`
      but not its shape.
    - When only relative FDS shape and equivalent inverted PSD are of interest,
      use `SNParams.normalized(...)` together with `p_scale=1.0`.

    If `p_scale` is omitted, `p_scale=1.0` is assumed only for normalized S-N
    parameters (`ref_stress=1`, `ref_cycles=1`). Physical S-N workflows must pass
    `p_scale` explicitly.
    """
    validate_sn(sn)
    validate_sdof(sdof)

    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")

    p_scale_resolved = resolve_p_scale(p_scale=p_scale, sn=sn)
    if detrend not in ("linear", "mean", "none"):
        raise ValidationError("detrend must be one of: 'linear', 'mean', 'none'.")
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
    k = float(sn.slope_k)
    c = float(sn.ref_cycles) * (float(sn.ref_stress) ** k)

    y = preprocess_signal(x, mode=detrend)

    if plan is None:
        H = build_transfer_matrix(fs=fs, n=int(y.size), f0_hz=f0, zeta=float(zeta), metric=sdof.metric)  # type: ignore[arg-type]
    else:
        H = validate_time_plan_compatibility(
            plan=plan,
            fs=fs,
            n_samples=int(y.size),
            f0=f0,
            zeta=float(zeta),
            metric=sdof.metric,
        )

    damage = _fds_from_signal_fft(
        y,
        fs=fs,
        f0=f0,
        zeta=float(zeta),
        metric=sdof.metric,
        k=float(k),
        c=float(c),
        p_scale=float(p_scale_resolved),
        batch_size=int(batch_size),
        amplitude_from_range=bool(sn.amplitude_from_range),
        H=H,
    )

    meta = {
        "compat": compat_dict(sn=sn, metric=sdof.metric, q=sdof.q, p_scale=p_scale_resolved, engine="time_rainflow_fft_numba"),
        "provenance": {
            "source": "compute_fds_time",
            "detrend": detrend,
            "batch_size": int(batch_size),
            "transfer_plan": bool(plan is not None),
            **nyquist_info,
        },
    }
    return FDSResult(f=f0, damage=damage, meta=meta)
