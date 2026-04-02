from __future__ import annotations

import numpy as np

from ._inversion_utils import blend_log_curves, build_edge_taper_weights, smooth_psd_log10
from .types import FDSResult, PSDResult, IterativeInversionParams, SDOFParams, SNParams
from .validate import ValidationError, ensure_compat_inversion
from .sdof_transfer import build_transfer_psd
from .fds_time import compute_fds_time
from .synth_time import synthesize_time_from_psd


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
    """Iteratively synthesize an acceleration PSD that matches a target FDS using the **time-domain** predictor.

    Predictor
    ---------
    Each iteration synthesizes one or more time histories from the candidate PSD, computes FDS via
    `compute_fds_time(...)`, then updates the PSD multiplicatively using an influence matrix derived from the SDOF transfer.

    Output is an **acceleration PSD** on the provided `f_psd_hz` grid.

    Parameters
    ----------
    target:
        Target FDS result (damage vs f0). Must carry `meta["compat"]`.
    f_psd_hz, psd_seed:
        Grid and seed for the synthesized acceleration PSD.
    fs, duration_s:
        Sampling rate and synthetic duration used in time synthesis for the predictor.
    target_duration_s:
        Optional duration to which predictor damage is scaled, using
        `damage_scaled = damage_synth * (target_duration_s / duration_s)`.
        If None, uses `duration_s` (no scaling).
    n_realizations:
        Number of random-phase realizations per iteration (averaged in damage space).
    seed:
        Seed for reproducible synthesis. Different realizations use `seed + r`.
    nfft:
        FFT length for synthesis. If None, uses next power-of-two >= N.

    Returns
    -------
    PSDResult
        Contains `meta["diagnostics"]` with convergence history and reconstruction.
    """
    if not np.isfinite(fs) or float(fs) <= 0:
        raise ValidationError("fs must be finite and > 0.")
    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValidationError("duration_s must be finite and > 0.")
    if target_duration_s is not None and (not np.isfinite(target_duration_s) or float(target_duration_s) <= 0):
        raise ValidationError("target_duration_s must be finite and > 0 when provided.")
    if not np.isfinite(p_scale) or float(p_scale) <= 0:
        raise ValidationError("p_scale must be finite and > 0.")
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

    ensure_compat_inversion(target=target, metric=sdof.metric, q=sdof.q, p_scale=p_scale, sn=sn)

    f0 = np.asarray(target.f, dtype=float).reshape(-1)
    zeta = 1.0 / (2.0 * float(sdof.q))
    t_syn = float(duration_s)
    t_target = float(target_duration_s) if target_duration_s is not None else t_syn
    duration_scale = float(t_target / t_syn)

    # Influence matrix alpha from PSD-domain transfer (metric-consistent)
    H = build_transfer_psd(f_psd_hz=f_psd, f0_hz=f0, zeta=zeta, metric=sdof.metric)
    B = np.abs(H) ** 2
    B_eff = np.clip(B, 1e-300, None)
    if float(params.alpha_sharpness) != 1.0:
        B_eff = B_eff ** float(params.alpha_sharpness)
    alpha = B_eff / (B_eff.sum(axis=0) + 1e-30)

    # Prior weights (same logic as spectral iterative)
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
    hist_err = []
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
                remove_mean=True,
            )
            fds = compute_fds_time(
                x,
                float(fs),
                sn=sn,
                sdof=sdof,
                p_scale=float(p_scale),
                detrend="none",
                batch_size=64,
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

        # Evaluate error
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
            "err_history": hist_err,
            "best_recon_fds": bestF,
            "iters": int(params.iters),
            "n_realizations": int(n_realizations),
            "fs": float(fs),
            "duration_s": float(t_syn),
            "target_duration_s": float(t_target),
            "duration_scale": float(duration_scale),
        },
        "compat": target.meta.get("compat", {}),
        "provenance": {"source": "invert_fds_iterative_time"},
    }
    return PSDResult(f=f_psd, psd=bestP, meta=meta)
