"""Shared numerical helpers for iterative PSD inversion engines.

This module groups low-level array operations reused by the spectral and
time-domain iterative inversion routes. The helpers here implement
smoothing, log-domain blending, edge weighting, and metadata reporting
for iterative-parameter usage.

Although these functions are internal, they define important numerical
policies that shape convergence behavior, especially around PSD floors,
odd-length smoothing windows, and high- or low-frequency regularization.
"""

from __future__ import annotations

import numpy as np

#: Iterative-inversion parameter names interpreted by both engines.
ITERATIVE_COMMON_PARAM_FIELDS: tuple[str, ...] = (
    "iters",
    "gamma",
    "gain_min",
    "gain_max",
    "alpha_sharpness",
    "floor",
    "smooth_enabled",
    "smooth_window_bins",
    "smooth_every_n_iters",
    "prior_blend",
    "prior_power",
    "edge_anchor_hz",
    "edge_anchor_blend",
)

#: Additional iterative-inversion parameter names used only by the spectral engine.
ITERATIVE_SPECTRAL_ONLY_PARAM_FIELDS: tuple[str, ...] = (
    "tail_cap_start_hz",
    "tail_cap_ratio",
    "low_cap_ratio",
    "post_smooth_window_bins",
    "post_smooth_blend",
    "post_refine_iters",
    "post_refine_gamma",
    "post_refine_min",
    "post_refine_max",
)


def effective_smoothing_window_bins(win: int) -> int:
    """Return the effective smoothing-window length used by the engine.

    Parameters
    ----------
    win : int
        Requested smoothing-window length in bins.

    Returns
    -------
    int
        Effective smoothing-window length. Even values greater than one
        are promoted to the next odd integer.

    Notes
    -----
    Reflection-based moving averages are centered most naturally when the
    kernel length is odd. This helper therefore enforces the library
    policy that smoothing windows should be odd whenever actual smoothing
    is performed.
    """
    win = int(win)
    if win <= 1:
        return win
    if win % 2 == 0:
        return win + 1
    return win


def moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    """Apply a reflection-padded moving average to a one-dimensional array.

    Parameters
    ----------
    x : numpy.ndarray
        Input one-dimensional array to smooth.
    win : int
        Requested moving-average window length in bins.

    Returns
    -------
    numpy.ndarray
        Smoothed array with the same shape as ``x``.

    Notes
    -----
    Boundary handling uses reflection padding rather than zero padding to
    reduce edge bias near the ends of the PSD vector. The effective
    window length is normalized through
    :func:`effective_smoothing_window_bins`.
    """
    win = effective_smoothing_window_bins(win)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(xp, kernel, mode="valid")


def smooth_psd_log10(P: np.ndarray, win: int, floor: float) -> np.ndarray:
    """Smooth a PSD in logarithmic amplitude space.

    Parameters
    ----------
    P : numpy.ndarray
        Input PSD values.
    win : int
        Requested smoothing-window length in bins.
    floor : float
        Strictly positive floor applied before taking ``log10``.

    Returns
    -------
    numpy.ndarray
        Smoothed PSD with the same shape as ``P``.

    Notes
    -----
    Log-domain smoothing is often preferable for PSD inversion because it
    regularizes multiplicative variation while preserving positivity.
    Values are clipped to ``floor`` before the logarithm is taken.
    """
    P = np.clip(P, floor, None)
    logP = np.log10(P)
    logP_s = moving_average_reflect(logP, win)
    return np.power(10.0, logP_s)


def blend_log_curves(cur: np.ndarray, ref: np.ndarray, weight: np.ndarray, floor: float) -> np.ndarray:
    """Blend two positive curves in logarithmic space.

    Parameters
    ----------
    cur : numpy.ndarray
        Current curve, typically the evolving PSD estimate.
    ref : numpy.ndarray
        Reference curve, typically a seed or prior PSD.
    weight : numpy.ndarray
        Pointwise blend weight clipped to the interval ``[0, 1]``.
    floor : float
        Strictly positive floor applied before the logarithm.

    Returns
    -------
    numpy.ndarray
        Log-domain blend of ``cur`` and ``ref``.

    Notes
    -----
    A zero weight returns the current curve, while a unit weight returns
    the reference curve. Intermediate values interpolate geometrically,
    which is consistent with multiplicative PSD updates.
    """
    w = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    return np.exp(
        (1.0 - w) * np.log(np.clip(cur, floor, None))
        + w * np.log(np.clip(ref, floor, None))
    )


def build_edge_taper_weights(f_psd: np.ndarray, edge_hz: float) -> np.ndarray:
    """Build taper weights near the low- and high-frequency edges of a PSD.

    Parameters
    ----------
    f_psd : numpy.ndarray
        PSD frequency grid in Hz.
    edge_hz : float
        Frequency span over which edge anchoring should ramp from ``1``
        at the boundary to ``0`` in the interior.

    Returns
    -------
    numpy.ndarray
        Edge-weight vector with values in the interval ``[0, 1]``.

    Notes
    -----
    The taper is applied symmetrically to the lower and upper edges of
    the PSD grid. When ``edge_hz <= 0``, the function returns zeros and
    edge anchoring is effectively disabled.
    """
    edge = float(edge_hz)
    w = np.zeros_like(f_psd, dtype=float)
    if edge <= 0.0:
        return w
    fmin = float(np.min(f_psd))
    fmax = float(np.max(f_psd))
    low = f_psd <= (fmin + edge)
    high = f_psd >= (fmax - edge)
    if np.any(low):
        w[low] = np.maximum(w[low], 1.0 - (f_psd[low] - fmin) / edge)
    if np.any(high):
        w[high] = np.maximum(w[high], 1.0 - (fmax - f_psd[high]) / edge)
    return np.clip(w, 0.0, 1.0)


def apply_edge_caps(
    P: np.ndarray,
    f_psd: np.ndarray,
    *,
    tail_cap_start_hz: float,
    tail_cap_ratio: float,
    low_cap_ratio: float,
    floor: float,
) -> np.ndarray:
    """Apply monotonic edge caps to a PSD estimate.

    Parameters
    ----------
    P : numpy.ndarray
        PSD estimate to regularize.
    f_psd : numpy.ndarray
        PSD frequency grid in Hz.
    tail_cap_start_hz : float
        Frequency above which the high-frequency tail cap becomes active.
    tail_cap_ratio : float
        Maximum allowed ratio between adjacent PSD bins in the capped
        high-frequency tail.
    low_cap_ratio : float
        Maximum allowed ratio between the first two PSD bins.
    floor : float
        Strictly positive PSD floor.

    Returns
    -------
    numpy.ndarray
        Regularized PSD after edge-cap enforcement.

    Notes
    -----
    The low-frequency cap limits the first bin relative to the second,
    while the tail cap limits geometric growth in the high-frequency
    region. Both mechanisms are intended as numerical stabilizers for
    iterative inversion rather than as a physical PSD model.
    """
    p = np.clip(np.asarray(P, dtype=float).copy(), floor, None)

    low_cap = float(low_cap_ratio)
    if low_cap > 0.0 and p.size > 1:
        p[0] = min(p[0], p[1] * low_cap)

    t_ratio = float(tail_cap_ratio)
    if t_ratio > 0.0 and p.size > 1:
        idx = int(np.searchsorted(f_psd, float(tail_cap_start_hz)))
        idx = max(1, min(idx, p.size - 1))
        for i in range(idx + 1, p.size):
            lim = p[i - 1] * t_ratio
            if p[i] > lim:
                p[i] = lim

    return np.clip(p, floor, None)


def iterative_param_usage(engine: str, params: object) -> dict[str, object]:
    """Report how iterative-inversion parameters are interpreted by an engine.

    Parameters
    ----------
    engine : {"spectral", "time"}
        Iterative inversion engine for which parameter usage should be
        summarized.
    params : object
        Parameter container exposing the fields used by
        :class:`fdscore.types.IterativeInversionParams`.

    Returns
    -------
    dict
        Dictionary containing used fields, ignored fields, effective
        values, and notes suitable for storage in inversion metadata.

    Notes
    -----
    The spectral engine uses both the common parameter subset and the
    spectral-only controls, while the time-domain engine currently uses
    only the common subset. Effective smoothing-window lengths are
    normalized through :func:`effective_smoothing_window_bins` and
    reported explicitly in the returned payload.
    """
    if engine == "spectral":
        used_fields = ITERATIVE_COMMON_PARAM_FIELDS + ITERATIVE_SPECTRAL_ONLY_PARAM_FIELDS
        ignored_fields: tuple[str, ...] = ()
    elif engine == "time":
        used_fields = ITERATIVE_COMMON_PARAM_FIELDS
        ignored_fields = ITERATIVE_SPECTRAL_ONLY_PARAM_FIELDS
    else:
        raise ValueError(f"Unsupported iterative inversion engine: {engine}")

    effective = {
        "smooth_window_bins": effective_smoothing_window_bins(getattr(params, "smooth_window_bins")),
    }
    if engine == "spectral":
        effective["post_smooth_window_bins"] = effective_smoothing_window_bins(getattr(params, "post_smooth_window_bins"))

    return {
        "engine": engine,
        "used_fields": list(used_fields),
        "ignored_fields": list(ignored_fields),
        "used": {name: getattr(params, name) for name in used_fields},
        "ignored": {name: getattr(params, name) for name in ignored_fields},
        "effective": effective,
        "notes": {
            "smoothing_window_policy": "Even smoothing windows are promoted to the next odd value.",
        },
    }
