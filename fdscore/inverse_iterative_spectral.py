from __future__ import annotations

import numpy as np

from .types import FDSResult, PSDResult, IterativeInversionParams, SDOFParams, SNParams
from .validate import ValidationError, ensure_compat_inversion
from .sdof_transfer import build_transfer_psd
from .fds_spectral import compute_fds_spectral_psd


def _moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    win = int(win)
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(xp, kernel, mode="valid")


def _smooth_psd_log10(P: np.ndarray, win: int, floor: float) -> np.ndarray:
    P = np.clip(P, floor, None)
    logP = np.log10(P)
    logP_s = _moving_average_reflect(logP, win)
    return np.power(10.0, logP_s)


def _blend_log_curves(cur: np.ndarray, ref: np.ndarray, weight: np.ndarray, floor: float) -> np.ndarray:
    w = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    return np.exp(
        (1.0 - w) * np.log(np.clip(cur, floor, None))
        + w * np.log(np.clip(ref, floor, None))
    )


def _build_edge_taper_weights(f_psd: np.ndarray, edge_hz: float) -> np.ndarray:
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


def _apply_edge_caps(
    P: np.ndarray,
    f_psd: np.ndarray,
    *,
    tail_cap_start_hz: float,
    tail_cap_ratio: float,
    low_cap_ratio: float,
    floor: float,
) -> np.ndarray:
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
    """Iteratively synthesize an acceleration PSD that matches a target FDS using the *spectral* Dirlik predictor.

    The predictor used internally is `compute_fds_spectral_psd(...)` with the
    provided `sn`, `sdof`, `p_scale`, and `duration_s`.
    This function outputs an **acceleration PSD** on the user-provided `f_psd_hz` grid.

    Parameters
    ----------
    target:
        Target FDS (damage vs oscillator frequency f0).
    f_psd_hz:
        Frequency grid for the synthesized PSD.
    psd_seed:
        Seed acceleration PSD on `f_psd_hz` (must be positive).
    duration_s:
        Benchmark/test duration for damage evaluation in the predictor.
    sn, sdof, p_scale:
        Must match the way `target` was computed (for meaningful inversion).
    params:
        Iteration and regularization knobs.

    Returns
    -------
    PSDResult
        `psd` is the synthesized acceleration PSD on `f_psd_hz`.
        `meta["diagnostics"]` includes best error and per-iteration history.
    """
    # Validate core inputs
    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValidationError("duration_s must be finite and > 0.")
    if not np.isfinite(p_scale) or float(p_scale) <= 0:
        raise ValidationError("p_scale must be finite and > 0.")

    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    P0 = np.asarray(psd_seed, dtype=float).reshape(-1)
    if f_psd.size < 2 or P0.size < 2 or f_psd.shape != P0.shape:
        raise ValidationError("f_psd_hz and psd_seed must be 1D arrays with same shape >= 2.")
    if not np.all(np.diff(f_psd) > 0):
        raise ValidationError("f_psd_hz must be strictly increasing.")
    if np.any(P0 <= 0) or not np.all(np.isfinite(P0)):
        raise ValidationError("psd_seed must be finite and strictly positive.")

    # Ensure compat between target and inversion config (strict)
    ensure_compat_inversion(target=target, metric=sdof.metric, q=sdof.q, p_scale=p_scale, sn=sn)

    f0 = np.asarray(target.f, dtype=float).reshape(-1)
    zeta = 1.0 / (2.0 * float(sdof.q))

    # Influence matrix alpha uses the same transfer for the chosen metric (generalized)
    H = build_transfer_psd(f_psd_hz=f_psd, f0_hz=f0, zeta=zeta, metric=sdof.metric)
    B = np.abs(H) ** 2  # (No, Nf)

    B_eff = np.clip(B, 1e-300, None)
    if float(params.alpha_sharpness) != 1.0:
        B_eff = B_eff ** float(params.alpha_sharpness)
    alpha = B_eff / (B_eff.sum(axis=0) + 1e-30)  # (No, Nf)

    # Prior weights derived from sensitivity (following 03b_inversao_psd_espectral.py)
    sens = B_eff.sum(axis=0)
    sens_n = sens / (np.max(sens) + 1e-30)
    prior_w_sens = np.clip(1.0 - sens_n, 0.0, 1.0) ** float(max(params.prior_power, 0.0))
    prior_w = np.clip(float(params.prior_blend), 0.0, 1.0) * prior_w_sens
    edge_w = _build_edge_taper_weights(f_psd=f_psd, edge_hz=params.edge_anchor_hz)
    prior_w = np.clip(prior_w + np.clip(float(params.edge_anchor_blend), 0.0, 1.0) * edge_w, 0.0, 1.0)
    use_prior = bool(np.any(prior_w > 0.0))

    floor = float(params.floor)
    P = np.clip(P0.copy(), floor, None)

    hist_err = []

    bestP = P.copy()
    bestErr = float("inf")
    bestF = None

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

        u = np.exp(alpha.T @ np.log(s + 1e-30))  # (Nf,)
        P *= u ** float(params.gamma)
        P = np.clip(P, floor, None)

        do_smooth = bool(params.smooth_enabled) and int(params.smooth_window_bins) > 1
        if do_smooth:
            every = int(params.smooth_every_n_iters)
            if every > 0:
                do_smooth = ((it + 1) % every) == 0
            if do_smooth:
                P = _smooth_psd_log10(P, win=int(params.smooth_window_bins), floor=floor)
                P = np.clip(P, floor, None)

        if use_prior:
            P = _blend_log_curves(cur=P, ref=P0, weight=prior_w, floor=floor)

        if (float(params.tail_cap_ratio) > 0.0) or (float(params.low_cap_ratio) > 0.0):
            P = _apply_edge_caps(
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

    # Optional post-smooth + refine
    Pfin = bestP
    Ffin = bestF if bestF is not None else predictor(Pfin)
    err_fin = bestErr

    if int(params.post_smooth_window_bins) > 1:
        Ps = _smooth_psd_log10(Pfin, win=int(params.post_smooth_window_bins), floor=floor)
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
            if err_p <= err_fin:
                Pfin, Ffin, err_fin = Pp, Fp, err_p

        if int(params.post_refine_iters) > 0:
            # light refine without additional priors change; reuse same update but fewer iters
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
            )
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
            # Compare err using FDS reconstruction stored in meta
            err_ref = float(Pref.meta.get("diagnostics", {}).get("best_err", float("inf")))
            if err_ref <= err_fin:
                Pfin = np.asarray(Pref.psd, dtype=float)
                Ffin = np.asarray(Pref.meta.get("diagnostics", {}).get("best_recon_fds", Ffin), dtype=float)
                err_fin = err_ref

    meta = {
        "diagnostics": {
            "best_err": float(err_fin),
            "err_history": hist_err,
            "best_recon_fds": Ffin,
            "iters": int(params.iters),
        },
        "compat": target.meta.get("compat", {}),
        "provenance": {"source": "invert_fds_iterative_spectral"},
    }
    return PSDResult(f=f_psd, psd=Pfin, meta=meta)
