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
    r"""Compute a spectral Fatigue Damage Spectrum from an acceleration PSD.

    This routine evaluates fatigue damage oscillator by oscillator using a
    PSD-domain response model and Dirlik's spectral fatigue approximation as
    implemented by ``FLife``.

    Pipeline
    --------
    The computation first builds the oscillator grid defined by ``sdof`` and
    the transfer matrix from base acceleration PSD to the selected response
    metric. For each oscillator, the response PSD is then computed as

    .. math::

       P_{resp}(f; f_0) =
       p_{scale}^2 \left| H(f; f_0) \right|^2 P_{base}(f)

    Dirlik's method is applied to that response PSD to estimate life, and
    Miner damage is recovered through

    .. math::

       D(f_0) = \frac{T}{life(f_0)}

    Parameters
    ----------
    f_psd_hz : numpy.ndarray
        One-sided input PSD frequency grid in Hz.
    psd_baseacc : numpy.ndarray
        One-sided base-acceleration PSD defined on ``f_psd_hz``.
    duration_s : float
        Exposure duration associated with the PSD.
    sn : SNParams
        S-N curve definition used by the Dirlik life calculation.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    p_scale : float or None
        Additional scale factor applied to the response quantity before damage
        evaluation. This must match the fatigue convention used elsewhere in
        the workflow.

    Returns
    -------
    FDSResult
        Damage spectrum on the oscillator grid defined by ``sdof``.

    Notes
    -----
    Dirlik is a spectral fatigue approximation derived from response-PSD
    moments. It is not the same algorithm as rainflow counting on a realized
    time history, so absolute levels from ``compute_fds_spectral_psd(...)`` and
    ``compute_fds_time(...)`` should not be expected to match exactly.

    Agreement with time-domain rainflow tends to improve when the response is
    well represented as a stationary Gaussian process and the record is long
    enough that the PSD is a stable descriptor. Differences grow for short
    records, strongly non-stationary environments, transient content, and
    non-Gaussian responses.

    For fixed ``slope_k``, ``p_scale``, ``ref_stress``, and ``ref_cycles`` act
    only as a global damage scaling factor. They change the magnitude of
    ``damage(f)`` but not its relative shape. Use
    ``SNParams.normalized(...)`` with ``p_scale=1.0`` when a normalized
    workflow is sufficient.

    Input PSD values are expected to be non-negative. Tiny negative values
    consistent with numerical noise are clamped to zero; materially negative
    values raise ``ValidationError``.

    References
    ----------
    Dirlik, T. (1985). Application of computers in fatigue analysis.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied Mechanics, 12(3), A159-A164.
    """
    validate_sn(sn)
    validate_sdof(sdof)
    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")

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
    r"""Compute a spectral FDS from a time history through Welch plus Dirlik.

    This convenience route first estimates a one-sided acceleration PSD from
    the input time history and then delegates to
    :func:`fdscore.compute_fds_spectral_psd`.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional base-acceleration time history.
    fs : float
        Sampling rate in Hz.
    sn : SNParams
        S-N curve definition used for the fatigue calculation.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    psd : PSDParams
        PSD-estimation configuration passed to ``compute_psd_welch(...)``.
    duration_s : float or None
        Exposure duration associated with the damage estimate. If ``None``,
        uses ``len(x) / fs``.
    p_scale : float or None
        Optional scale factor applied to the response quantity before damage
        evaluation.

    Returns
    -------
    FDSResult
        Spectral damage spectrum evaluated from the internally estimated PSD.

    Notes
    -----
    This route combines two approximation layers:

    1. Welch estimation of the PSD from a finite realization.
    2. Dirlik spectral fatigue damage from the estimated PSD.

    Differences relative to ``compute_fds_time(...)`` or to
    ``compute_fds_spectral_psd(...)`` with a reference PSD are therefore
    expected for finite-length signals.

    References
    ----------
    Dirlik, T. (1985). Application of computers in fatigue analysis.
    """
    if not bool(psd.onesided):
        raise ValidationError("compute_fds_spectral_time requires PSDParams.onesided=True.")

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
