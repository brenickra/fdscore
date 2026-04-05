import numpy as np
import pytest

from fdscore import (
    FDSResult,
    IterativeInversionParams,
    SDOFParams,
    SNParams,
    compute_fds_time,
    invert_fds_iterative_spectral,
    invert_fds_iterative_time,
)
from fdscore.validate import compat_dict
import fdscore.inverse_iterative_spectral as spectral_mod


def test_iterative_time_diagnostics_expose_main_loop_metadata():
    fs = 500.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 30 * t) + 0.05 * np.sin(2 * np.pi * 80 * t)

    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="pv")
    fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", batch_size=16)

    f_psd = np.linspace(0.0, fs / 2.0, int(fs / 2) + 1)
    p0 = np.ones_like(f_psd) * 1e-6
    params = IterativeInversionParams(iters=2, smooth_enabled=False, prior_blend=0.0)

    out = invert_fds_iterative_time(
        fds,
        f_psd_hz=f_psd,
        psd_seed=p0,
        fs=fs,
        duration_s=2.0,
        sn=sn,
        sdof=sdof,
        p_scale=1.0,
        params=params,
        n_realizations=1,
        seed=0,
    )

    diag = out.meta["diagnostics"]
    assert diag["predictor_evals_per_iteration"] == 2
    assert diag["best_stage"] == "main_loop"
    assert diag["err_history_scope"] == "main_loop_only"
    assert len(diag["err_history"]) == params.iters


def test_iterative_spectral_diagnostics_record_post_stages_without_flife(monkeypatch):
    target_f = np.array([10.0, 20.0, 30.0])
    target_damage = np.array([1.0, 1.5, 2.0])
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, f=target_f, metric="pv")
    target = FDSResult(
        f=target_f,
        damage=target_damage,
        meta={"compat": compat_dict(metric="pv", q=10.0, p_scale=1.0, sn=sn, engine="spectral")},
    )

    def fake_compute_fds_spectral_psd(*, f_psd_hz, psd_baseacc, duration_s, sn, sdof, p_scale):
        scale = float(np.mean(np.asarray(psd_baseacc, dtype=float)) / 1e-6)
        return FDSResult(f=np.asarray(sdof.f, dtype=float), damage=target_damage * scale, meta={})

    monkeypatch.setattr(spectral_mod, "compute_fds_spectral_psd", fake_compute_fds_spectral_psd)

    f_psd = np.linspace(5.0, 100.0, 64)
    p0 = np.ones_like(f_psd) * 1e-6
    params = IterativeInversionParams(
        iters=2,
        smooth_enabled=True,
        smooth_window_bins=4,
        smooth_every_n_iters=1,
        prior_blend=0.0,
        post_smooth_window_bins=6,
        post_smooth_blend=0.5,
        post_refine_iters=2,
        post_refine_gamma=0.4,
        post_refine_min=0.8,
        post_refine_max=1.2,
    )

    out = invert_fds_iterative_spectral(
        target,
        f_psd_hz=f_psd,
        psd_seed=p0,
        duration_s=2.0,
        sn=sn,
        sdof=sdof,
        p_scale=1.0,
        params=params,
    )

    diag = out.meta["diagnostics"]
    assert diag["predictor_evals_per_iteration"] == 2
    assert diag["err_history_scope"] == "main_loop_only"
    assert len(diag["err_history"]) == params.iters
    assert diag["post_smooth_err"] == pytest.approx(0.0)
    assert diag["post_refine_err"] == pytest.approx(0.0)
    assert len(diag["post_refine_err_history"]) == params.post_refine_iters
    assert diag["post_refine_params"]["post_refine_iters"] == 0
    assert diag["post_refine_params"]["smooth_window_bins"] == 7
    assert diag["best_stage"] in {"main_loop", "post_smooth", "post_refine"}


