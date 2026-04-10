from __future__ import annotations

import numpy as np

from ._inversion_utils import blend_log_curves, build_edge_taper_weights, iterative_param_usage, smooth_psd_log10
from .types import FDSResult, PSDResult, IterativeInversionParams, SDOFParams, SNParams
from .validate import ValidationError, ensure_compat_inversion
from .sdof_transfer import build_transfer_psd
from .fds_time import compute_fds_time
from .synth_time import synthesize_time_from_psd

_TIME_ITERATIVE_PREDICTOR_DETREND = "none"
_TIME_ITERATIVE_PREDICTOR_BATCH_SIZE = 64
_TIME_ITERATIVE_SYNTH_REMOVE_MEAN = True


def invert_fds_iterative_time(
    target: FDSResult,
    *,
    f_psd_hz: np.ndarray,
    psd_seed: np.ndarray,
    fs: float,
    duration_s: float,
    sn: SNParams,
    sdof: SDOFParams,
    p_scale: float,
    params: IterativeInversionParams = IterativeInversionParams(),
    n_realizations: int = 1,
    seed: int | None = 0,
    nfft: int | None = None,
    target_duration_s: float | None = None,
) -> PSDResult:
    r"""Iteratively synthesize a PSD that matches a target FDS with a time predictor.

    This routine solves the inverse problem "find an acceleration PSD whose
    time-domain FDS matches ``target``" by repeatedly synthesizing time
    histories from the candidate PSD and re-evaluating
    :func:`fdscore.compute_fds_time`.

    Algorithm
    ---------
    Let :math:`F_{target}` be the target damage spectrum and let
    :math:`F(P)` be the average time-domain predictor response for candidate
    PSD :math:`P`. The method builds a PSD-to-oscillator influence matrix from
    the SDOF transfer model and converts it to a normalized redistribution
    matrix :math:`\alpha`.

    For each predictor call, the routine synthesizes ``n_realizations``
    random-phase time histories from :math:`P`, evaluates their FDS with
    :func:`fdscore.compute_fds_time`, and averages the resulting damage
    spectra.

    Oscillator-wise multiplicative gains are computed as

    .. math::

       s_i = \left(\frac{F_{target, i}}{F_i(P)}\right)^{2 / k}

    and clipped to the interval defined by ``gain_min`` and ``gain_max``.
    Those gains are projected back to PSD bins through

    .. math::

       u = \exp\left(\alpha^T \log(s)\right)

    The candidate PSD is then updated multiplicatively as

    .. math::

       P \leftarrow P \, u^{\gamma}

    followed by optional smoothing and prior blending. The updated PSD is
    re-evaluated and scored by the median absolute log10-domain mismatch.

    Parameters
    ----------
    target : FDSResult
        Target FDS result. It must carry compatibility metadata.
    f_psd_hz : numpy.ndarray
        Frequency grid in Hz for the synthesized acceleration PSD.
    psd_seed : numpy.ndarray
        Strictly positive seed PSD defined on ``f_psd_hz``.
    fs : float
        Sampling rate in Hz used during time synthesis and FDS evaluation.
    duration_s : float
        Synthetic duration in seconds used for each predictor realization.
    sn : SNParams
        S-N curve definition used by the time-domain predictor.
    sdof : SDOFParams
        Oscillator-grid definition and response metric used by the predictor.
    p_scale : float
        Response scale factor used by the predictor. It must be compatible with
        the way ``target`` was computed.
    params : IterativeInversionParams
        Iteration and regularization parameters.
    n_realizations : int
        Number of random-phase synthesized histories averaged per predictor
        call.
    seed : int or None
        Seed for reproducible synthesis. Realization ``r`` uses ``seed + r``.
    nfft : int or None
        FFT length used during synthesis. If ``None``, the synthesis routine
        chooses the next power of two.
    target_duration_s : float or None
        Optional duration to which the predictor damage is rescaled through

        .. math::

           D_{scaled} = D_{synth} \frac{T_{target}}{T_{synth}}

        If ``None``, no duration scaling is applied.

    Returns
    -------
    PSDResult
        Synthesized acceleration PSD on ``f_psd_hz``. The result metadata
        includes convergence diagnostics, predictor configuration, and
        explicit per-engine parameter usage.

    Notes
    -----
    The inversion heuristic implemented here is library-specific. It combines a
    stochastic time-history predictor with a multiplicative PSD update and does
    not correspond to a published closed-form inversion formula.

    The convergence metric stored in
    ``meta["diagnostics"]["best_err"]`` is the median absolute log10-domain
    mismatch over bins where both target and predicted damage are positive.

    The main loop performs two predictor evaluations per iteration: one on the
    current PSD to derive oscillator-wise gains, and one after the update to
    score the candidate that may become the best solution.

    Because the predictor is stochastic, convergence depends on
    ``n_realizations``, ``seed``, ``duration_s``, and ``nfft``. The fixed
    internal predictor policy is recorded in
    ``meta["diagnostics"]["predictor_config"]`` and currently uses
    ``synthesize_time_from_psd(remove_mean=True)`` together with
    ``compute_fds_time(detrend="none", batch_size=64)``.

    References
    ----------
    ASTM E1049-85(2017). Standard Practices for Cycle Counting in Fatigue Analysis.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied Mechanics, 12(3), A159-A164.
    """
    if not np.isfinite(fs) or float(fs) <= 0:
        raise ValidationError("fs must be finite and > 0.")
    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValidationError("duration_s must be finite and > 0.")
    if target_duration_s is not None and (not np.isfinite(target_duration_s) or float(target_duration_s) <= 0):
        raise ValidationError("target_duration_s must be finite and > 0 when provided.")
    if not np.isfinite(p_scale) or float(p_scale) <= 0:
        raise ValidationError("p_scale must be finite and > 0.")
    if not np.isfinite(sdof.q) or float(sdof.q) <= 0:
        raise ValidationError("sdof.q must be finite and > 0.")
    n_realizations = int(n_realizations)
    if n_realizations < 1:
        raise ValidationError("n_realizations must be >= 1.")

    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    P0 = np.asarray(psd_seed, dtype=float).reshape(-1)
    if f_psd.size < 2 or P0.size < 2 or f_psd.shape != P0.shape:
        raise ValidationError("f_psd_hz and psd_seed must be 1D arrays with same shape >= 2.")
    if not np.all(np.diff(f_psd) > 0):
        raise ValidationError("f_psd_hz must be strictly increasing.")
    if np.any(P0 <= 0) or not np.all(np.isfinite(P0)):
        raise ValidationError("psd_seed must be finite and strictly positive.")

    q = float(sdof.q)
    ensure_compat_inversion(target=target, metric=sdof.metric, q=q, p_scale=p_scale, sn=sn, sdof=sdof)

    f0 = np.asarray(target.f, dtype=float).reshape(-1)
    zeta = 1.0 / (2.0 * q)
    t_syn = float(duration_s)
    t_target = float(target_duration_s) if target_duration_s is not None else t_syn
    duration_scale = float(t_target / t_syn)

    H = build_transfer_psd(f_psd_hz=f_psd, f0_hz=f0, zeta=zeta, metric=sdof.metric)
    B = np.abs(H) ** 2
    B_eff = np.clip(B, 1e-300, None)
    if float(params.alpha_sharpness) != 1.0:
        B_eff = B_eff ** float(params.alpha_sharpness)
    alpha = B_eff / (B_eff.sum(axis=0) + 1e-30)

    sens = B_eff.sum(axis=0)
    sens_n = sens / (np.max(sens) + 1e-30)
    prior_w_sens = np.clip(1.0 - sens_n, 0.0, 1.0) ** float(max(params.prior_power, 0.0))
    prior_w = np.clip(float(params.prior_blend), 0.0, 1.0) * prior_w_sens
    edge_w = build_edge_taper_weights(f_psd=f_psd, edge_hz=params.edge_anchor_hz)
    prior_w = np.clip(prior_w + np.clip(float(params.edge_anchor_blend), 0.0, 1.0) * edge_w, 0.0, 1.0)
    use_prior = bool(np.any(prior_w > 0.0))

    floor = float(params.floor)
    P = np.clip(P0.copy(), floor, None)

    target_fds = np.asarray(target.damage, dtype=float)
    hist_err: list[float] = []
    bestP = P.copy()
    bestErr = float("inf")
    bestF = None

    def predictor(Pyy: np.ndarray) -> np.ndarray:
        acc_dmg = None
        for r in range(n_realizations):
            s = None if seed is None else int(seed) + int(r)
            x = synthesize_time_from_psd(
                f_psd_hz=f_psd,
                psd=Pyy,
                fs=float(fs),
                duration_s=float(t_syn),
                seed=s,
                nfft=nfft,
                remove_mean=_TIME_ITERATIVE_SYNTH_REMOVE_MEAN,
            )
            fds = compute_fds_time(
                x,
                float(fs),
                sn=sn,
                sdof=sdof,
                p_scale=float(p_scale),
                detrend=_TIME_ITERATIVE_PREDICTOR_DETREND,
                batch_size=_TIME_ITERATIVE_PREDICTOR_BATCH_SIZE,
            )
            if acc_dmg is None:
                acc_dmg = np.asarray(fds.damage, dtype=float).copy()
            else:
                acc_dmg += np.asarray(fds.damage, dtype=float)
        dmg_mean = (acc_dmg / float(n_realizations)).astype(float, copy=False)
        return dmg_mean * float(duration_scale)

    iters = int(params.iters)
    if iters <= 0:
        raise ValidationError("params.iters must be >= 1.")

    for it in range(iters):
        pred = predictor(P)
        safe = (target_fds > 0) & (pred > 0)

        s = np.ones_like(target_fds)
        if np.any(safe):
            s[safe] = (target_fds[safe] / pred[safe]) ** (2.0 / float(sn.slope_k))
        s = np.clip(s, float(params.gain_min), float(params.gain_max))

        u = np.exp(alpha.T @ np.log(s + 1e-30))
        P *= u ** float(params.gamma)
        P = np.clip(P, floor, None)

        do_smooth = bool(params.smooth_enabled) and int(params.smooth_window_bins) > 1
        if do_smooth:
            every = int(params.smooth_every_n_iters)
            if every > 0:
                do_smooth = ((it + 1) % every) == 0
            if do_smooth:
                P = smooth_psd_log10(P, win=int(params.smooth_window_bins), floor=floor)
                P = np.clip(P, floor, None)

        if use_prior:
            P = blend_log_curves(cur=P, ref=P0, weight=prior_w, floor=floor)

        pred_eval = predictor(P)
        safe_eval = (target_fds > 0) & (pred_eval > 0)
        if np.any(safe_eval):
            err = float(np.median(np.abs(np.log10(pred_eval[safe_eval]) - np.log10(target_fds[safe_eval]))))
        else:
            err = float("inf")
        hist_err.append(err)

        if err < bestErr:
            bestErr = err
            bestP = P.copy()
            bestF = pred_eval.copy()

    meta = {
        "diagnostics": {
            "best_err": float(bestErr),
            "best_stage": "main_loop",
            "err_history": hist_err,
            "err_history_scope": "main_loop_only",
            "best_recon_fds": bestF,
            "iters": int(params.iters),
            "predictor_evals_per_iteration": 2,
            "predictor_config": {
                "synthesize_time_from_psd_remove_mean": _TIME_ITERATIVE_SYNTH_REMOVE_MEAN,
                "compute_fds_time_detrend": _TIME_ITERATIVE_PREDICTOR_DETREND,
                "compute_fds_time_batch_size": _TIME_ITERATIVE_PREDICTOR_BATCH_SIZE,
                "nfft": None if nfft is None else int(nfft),
            },
            "n_realizations": int(n_realizations),
            "fs": float(fs),
            "duration_s": float(t_syn),
            "target_duration_s": float(t_target),
            "duration_scale": float(duration_scale),
        },
        "param_usage": iterative_param_usage(engine="time", params=params),
        "compat": target.meta.get("compat", {}),
        "provenance": {"source": "invert_fds_iterative_time"},
    }
    return PSDResult(f=f_psd, psd=bestP, meta=meta)
