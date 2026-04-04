from __future__ import annotations

import numpy as np

from .types import SNParams, SDOFParams, PSDParams, FDSResult
from .grid import build_frequency_grid
from .validate import ValidationError, validate_sn, validate_sdof, compat_dict, resolve_p_scale
from .sdof_transfer import build_transfer_psd
from .psd_welch import compute_psd_welch
from ._psd_utils import clip_tiny_negative_psd_or_raise


def _require_flife():
    try:
        import FLife  # type: ignore
        return FLife
    except Exception as e:  # pragma: no cover
        raise ValidationError(
            "Spectral FDS requires the external dependency 'FLife' (for Dirlik). "
            "Install it in your environment, then retry."
        ) from e


def compute_fds_spectral_psd(
    f_psd_hz: np.ndarray,
    psd_baseacc: np.ndarray,
    *,
    duration_s: float,
    sn: SNParams,
    sdof: SDOFParams,
    p_scale: float | None = None,
) -> FDSResult:
    """Compute spectral FDS using Dirlik (FLife) from an input base-acceleration PSD.

    Steps
    -----
    1) Build oscillator grid (sdof)
    2) Build transfer H(f) from base acceleration PSD to chosen metric response PSD
    3) For each oscillator, compute response PSD:
         P_resp = (p_scale^2) * |H|^2 * P_base
    4) Use FLife.Dirlik to compute life, then Miner damage as:
         damage = duration_s / life

    Parameters
    ----------
    f_psd_hz, psd_baseacc:
        Input PSD grid and values (one-sided). Must be same shape.
    duration_s:
        Exposure duration associated with the PSD.
    p_scale:
        Additional scale applied to the response time series/PSD before damage counting.
        This value must be consistent with the FDS and inversion workflow being used.

    Notes
    -----
    For fixed `slope_k`, `p_scale`, `ref_stress`, and `ref_cycles` act as a global
    damage scaling factor. They affect absolute damage magnitude, but not the shape
    of the FDS. Use `SNParams.normalized(...)` with `p_scale=1.0` when a normalized
    workflow is sufficient.

    Dirlik is a spectral fatigue approximation. It is not the same algorithm as
    time-domain rainflow counting on a realized signal, so absolute FDS levels from
    spectral and time-domain routes should not be expected to match exactly.

    Input PSD values are expected to be non-negative. Tiny negative values consistent
    with numerical noise are clamped to zero. Materially negative values raise
    `ValidationError`.

    Returns
    -------
    FDSResult
        Damage spectrum on the oscillator grid.
    """
    validate_sn(sn)
    validate_sdof(sdof)

    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValidationError("duration_s must be finite and > 0.")
    p_scale_resolved = resolve_p_scale(p_scale=p_scale, sn=sn)

    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    Pyy = np.asarray(psd_baseacc, dtype=float).reshape(-1)
    if f_psd.size < 2 or Pyy.size < 2 or f_psd.shape != Pyy.shape:
        raise ValidationError("f_psd_hz and psd_baseacc must be 1D arrays of the same length >= 2.")
    if not (np.all(np.isfinite(f_psd)) and np.all(np.isfinite(Pyy))):
        raise ValidationError("PSD inputs must be finite.")
    Pyy = clip_tiny_negative_psd_or_raise(Pyy, label="psd_baseacc")
    if not np.all(np.diff(f_psd) > 0):
        raise ValidationError("f_psd_hz must be strictly increasing.")
    if f_psd[0] < 0:
        raise ValidationError("f_psd_hz must be >= 0.")

    f0 = build_frequency_grid(sdof)
    zeta = 1.0 / (2.0 * float(sdof.q))
    H = build_transfer_psd(f_psd_hz=f_psd, f0_hz=f0, zeta=zeta, metric=sdof.metric)

    FLife = _require_flife()
    k = float(sn.slope_k)
    C = float(sn.C())

    dmg = np.zeros_like(f0, dtype=float)
    scale2 = float(p_scale_resolved) ** 2

    # Loop: FLife Dirlik is object-based; vectorization brings little benefit here
    for i in range(f0.size):
        P_resp = scale2 * (np.abs(H[i]) ** 2) * Pyy
        sd = FLife.SpectralData(input={"PSD": P_resp, "f": f_psd})
        life = float(FLife.Dirlik(sd).get_life(C=C, k=k))
        if not np.isfinite(life) or life <= 0.0:
            raise ValidationError(
                f"FLife returned invalid life for oscillator f0={float(f0[i])} Hz: {life}"
            )
        dmg[i] = float(duration_s) / life

    meta = {
        "compat": compat_dict(sn=sn, metric=sdof.metric, q=sdof.q, p_scale=p_scale_resolved, engine="spectral_dirlik_flife"),
        "provenance": {
            "source": "compute_fds_spectral_psd",
            "duration_s": float(duration_s),
        },
    }
    return FDSResult(f=f0, damage=dmg, meta=meta)


def compute_fds_spectral_time(
    x: np.ndarray,
    fs: float,
    *,
    sn: SNParams,
    sdof: SDOFParams,
    psd: PSDParams,
    duration_s: float | None = None,
    p_scale: float | None = None,
) -> FDSResult:
    """Compute spectral FDS from a time history by estimating PSD internally (Welch) then using Dirlik.

    If `duration_s` is None, uses `len(x)/fs`.

    This route combines two approximation layers:
    - PSD estimation through Welch
    - spectral fatigue damage through Dirlik

    Differences relative to `compute_fds_time(...)` or to `compute_fds_spectral_psd(...)`
    with an explicit reference PSD are therefore expected for finite-length signals.
    """
    if duration_s is None:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValidationError("x must be 1D when duration_s is None.")
        if not np.isfinite(fs) or float(fs) <= 0:
            raise ValidationError("fs must be > 0.")
        duration_s = float(x.size) / float(fs)

    f_psd, Pyy = compute_psd_welch(x, fs=float(fs), psd=psd)
    return compute_fds_spectral_psd(
        f_psd_hz=f_psd,
        psd_baseacc=Pyy,
        duration_s=float(duration_s),
        sn=sn,
        sdof=sdof,
        p_scale=p_scale,
    )
