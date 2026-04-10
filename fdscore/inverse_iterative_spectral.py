from __future__ import annotations

import numpy as np

from ._inversion_utils import (
    apply_edge_caps,
    blend_log_curves,
    build_edge_taper_weights,
    effective_smoothing_window_bins,
    iterative_param_usage,
    smooth_psd_log10,
)
from .types import FDSResult, PSDResult, IterativeInversionParams, SDOFParams, SNParams
from .validate import ValidationError, ensure_compat_inversion
from .sdof_transfer import build_transfer_psd
from .fds_spectral import compute_fds_spectral_psd


def invert_fds_iterative_spectral(
    target: FDSResult,
    *,
    f_psd_hz: np.ndarray,
    psd_seed: np.ndarray,
    duration_s: float,
    sn: SNParams,
    sdof: SDOFParams,
    p_scale: float,
    params: IterativeInversionParams = IterativeInversionParams(),
) -> PSDResult:
    r"""Iteratively synthesize a PSD that matches a target FDS with a spectral predictor.

    This routine solves the inverse problem "find an acceleration PSD whose
    predicted FDS matches ``target``" by repeatedly calling the spectral
    predictor :func:`fdscore.fds_spectral.compute_fds_spectral_psd`.

    Algorithm
    ---------
    Let :math:`F_{target}` be the target damage spectrum and let
    :math:`F(P)` be the spectral predictor evaluated at candidate PSD
    :math:`P`. The method builds a PSD-to-oscillator influence matrix from
    the SDOF transfer model and converts it to a normalized redistribution
    matrix :math:`\alpha`.

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

    followed by optional smoothing, prior blending, and edge capping. The
    updated PSD is re-evaluated and scored by the median absolute log10-domain
    mismatch.

    The function always outputs a one-sided acceleration PSD on the user
    supplied ``f_psd_hz`` grid.

    Parameters
    ----------
    target : FDSResult
        Target FDS result, expressed as damage versus oscillator frequency.
    f_psd_hz : numpy.ndarray
        Frequency grid in Hz for the synthesized acceleration PSD.
    psd_seed : numpy.ndarray
        Strictly positive seed PSD defined on ``f_psd_hz``.
    duration_s : float
        Exposure duration used by the spectral predictor.
    sn : SNParams
        S-N curve definition used by the predictor.
    sdof : SDOFParams
        Oscillator-grid definition and response metric used by the predictor.
    p_scale : float
        Response scale factor used by the predictor. It must be compatible with
        the way ``target`` was computed.
    params : IterativeInversionParams
        Iteration and regularization parameters.

    Returns
    -------
    PSDResult
        Synthesized acceleration PSD on ``f_psd_hz``. The result metadata
        includes convergence diagnostics, reconstructed FDS traces, and
        explicit per-engine parameter usage.

    Notes
    -----
    The update strategy implemented here is a library-specific multiplicative
    inversion heuristic. It is not claimed as a closed-form method from the
    fatigue literature.

    The convergence metric stored in
    ``meta["diagnostics"]["best_err"]`` is the median absolute log10-domain
    mismatch over bins where both target and predicted damage are positive:

    .. math::

       median\left(\left|\log_{10} F_{pred} - \log_{10} F_{target}\right|\right)

    The main loop performs two predictor evaluations per iteration: one on the
    current PSD to derive oscillator-wise gains, and one after the multiplicative
    update to score the candidate that may become the new best solution.

    If post-smoothing and post-refinement are enabled, an additional spectral
    inversion pass is launched from the smoothed candidate using a reduced
    parameter set. The metadata distinguishes whether the best result came from
    the main loop, the post-smooth stage, or the post-refine stage.

    References
    ----------
    Dirlik, T. (1985). Application of computers in fatigue analysis.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied Mechanics, 12(3), A159-A164.
    """
    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValidationError("duration_s must be finite and > 0.")
    if not np.isfinite(p_scale) or float(p_scale) <= 0:
        raise ValidationError("p_scale must be finite and > 0.")
    if not np.isfinite(sdof.q) or float(sdof.q) <= 0:
        raise ValidationError("sdof.q must be finite and > 0.")

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

    hist_err: list[float] = []

    bestP = P.copy()
    bestErr = float("inf")
    bestF = None
    best_stage = "main_loop"

    def predictor(Pyy: np.ndarray) -> np.ndarray:
        fds = compute_fds_spectral_psd(
            f_psd_hz=f_psd,
            psd_baseacc=Pyy,
            duration_s=float(duration_s),
            sn=sn,
            sdof=sdof,
            p_scale=float(p_scale),
        )
        return np.asarray(fds.damage, dtype=float)

    target_fds = np.asarray(target.damage, dtype=float)

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

        if (float(params.tail_cap_ratio) > 0.0) or (float(params.low_cap_ratio) > 0.0):
            P = apply_edge_caps(
                P=P,
                f_psd=f_psd,
                tail_cap_start_hz=float(params.tail_cap_start_hz),
                tail_cap_ratio=float(params.tail_cap_ratio),
                low_cap_ratio=float(params.low_cap_ratio),
                floor=floor,
            )

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
            best_stage = "main_loop"

    Pfin = bestP
    Ffin = bestF if bestF is not None else predictor(Pfin)
    err_fin = bestErr
    post_smooth_err: float | None = None
    post_refine_err: float | None = None
    post_refine_err_history: list[float] = []
    post_refine_params_meta: dict[str, float | int] | None = None

    if int(params.post_smooth_window_bins) > 1:
        Ps = smooth_psd_log10(Pfin, win=int(params.post_smooth_window_bins), floor=floor)
        blend = float(np.clip(params.post_smooth_blend, 0.0, 1.0))
        if blend >= 1.0:
            Pp = Ps
        elif blend <= 0.0:
            Pp = np.clip(Pfin, floor, None)
        else:
            Pp = np.exp((1.0 - blend) * np.log(np.clip(Pfin, floor, None)) + blend * np.log(np.clip(Ps, floor, None)))
        Fp = predictor(Pp)
        safe = (target_fds > 0) & (Fp > 0)
        if np.any(safe):
            err_p = float(np.median(np.abs(np.log10(Fp[safe]) - np.log10(target_fds[safe]))))
        else:
            err_p = float("inf")
        post_smooth_err = err_p
        if err_p <= err_fin:
            Pfin, Ffin, err_fin = Pp, Fp, err_p
            best_stage = "post_smooth"

        if int(params.post_refine_iters) > 0:
            refine_params = IterativeInversionParams(
                iters=int(params.post_refine_iters),
                gamma=float(params.post_refine_gamma),
                gain_min=float(params.post_refine_min),
                gain_max=float(params.post_refine_max),
                alpha_sharpness=float(params.alpha_sharpness),
                floor=floor,
                smooth_enabled=True,
                smooth_window_bins=int(params.post_smooth_window_bins),
                smooth_every_n_iters=1,
                prior_blend=float(params.prior_blend),
                prior_power=float(params.prior_power),
                edge_anchor_hz=float(params.edge_anchor_hz),
                edge_anchor_blend=float(params.edge_anchor_blend),
                tail_cap_start_hz=float(params.tail_cap_start_hz),
                tail_cap_ratio=float(params.tail_cap_ratio),
                low_cap_ratio=float(params.low_cap_ratio),
                post_smooth_window_bins=0,
                post_smooth_blend=0.0,
                post_refine_iters=0,
                post_refine_gamma=float(params.post_refine_gamma),
                post_refine_min=float(params.post_refine_min),
                post_refine_max=float(params.post_refine_max),
            )
            post_refine_params_meta = {
                "iters": int(refine_params.iters),
                "gamma": float(refine_params.gamma),
                "gain_min": float(refine_params.gain_min),
                "gain_max": float(refine_params.gain_max),
                "smooth_window_bins": int(effective_smoothing_window_bins(refine_params.smooth_window_bins)),
                "post_refine_iters": int(refine_params.post_refine_iters),
            }
            Pref = invert_fds_iterative_spectral(
                target,
                f_psd_hz=f_psd,
                psd_seed=Pp,
                duration_s=float(duration_s),
                sn=sn,
                sdof=sdof,
                p_scale=float(p_scale),
                params=refine_params,
            )
            pref_diag = Pref.meta.get("diagnostics", {})
            post_refine_err = float(pref_diag.get("best_err", float("inf")))
            post_refine_err_history = [float(x) for x in pref_diag.get("err_history", [])]
            if post_refine_err <= err_fin:
                Pfin = np.asarray(Pref.psd, dtype=float)
                Ffin = np.asarray(pref_diag.get("best_recon_fds", Ffin), dtype=float)
                err_fin = post_refine_err
                best_stage = "post_refine"

    meta = {
        "diagnostics": {
            "best_err": float(err_fin),
            "best_stage": best_stage,
            "err_history": hist_err,
            "err_history_scope": "main_loop_only",
            "post_smooth_err": post_smooth_err,
            "post_refine_err": post_refine_err,
            "post_refine_err_history": post_refine_err_history,
            "post_refine_params": post_refine_params_meta,
            "best_recon_fds": Ffin,
            "iters": int(params.iters),
            "predictor_evals_per_iteration": 2,
        },
        "param_usage": iterative_param_usage(engine="spectral", params=params),
        "compat": target.meta.get("compat", {}),
        "provenance": {"source": "invert_fds_iterative_spectral"},
    }
    return PSDResult(f=f_psd, psd=Pfin, meta=meta)

