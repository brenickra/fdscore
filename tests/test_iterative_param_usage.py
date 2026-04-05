import numpy as np
import pytest

from fdscore import (
    IterativeInversionParams,
    PSDParams,
    SDOFParams,
    SNParams,
    compute_fds_spectral_time,
    compute_fds_time,
    invert_fds_iterative_spectral,
    invert_fds_iterative_time,
)


def _build_params() -> IterativeInversionParams:
    return IterativeInversionParams(
        iters=2,
        smooth_enabled=True,
        smooth_window_bins=10,
        smooth_every_n_iters=1,
        prior_blend=0.0,
        tail_cap_start_hz=120.0,
        tail_cap_ratio=1.1,
        low_cap_ratio=1.2,
        post_smooth_window_bins=6,
        post_smooth_blend=0.4,
        post_refine_iters=1,
        post_refine_gamma=0.4,
        post_refine_min=0.8,
        post_refine_max=1.5,
    )


def test_iterative_time_reports_ignored_param_subset():
    fs = 500.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 30 * t) + 0.05 * np.sin(2 * np.pi * 80 * t)

    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="pv")
    fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", batch_size=16)

    f_psd = np.linspace(0.0, fs / 2.0, int(fs / 2) + 1)
    p0 = np.ones_like(f_psd) * 1e-6
    params = _build_params()

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

    usage = out.meta["param_usage"]
    assert usage["engine"] == "time"
    assert "iters" in usage["used_fields"]
    assert "tail_cap_ratio" in usage["ignored_fields"]
    assert usage["ignored"]["tail_cap_ratio"] == pytest.approx(params.tail_cap_ratio)
    assert usage["ignored"]["post_smooth_window_bins"] == params.post_smooth_window_bins
    assert usage["effective"]["smooth_window_bins"] == 11
    assert "post_smooth_window_bins" not in usage["effective"]
    assert "Even smoothing windows" in usage["notes"]["smoothing_window_policy"]


def test_iterative_spectral_reports_full_param_usage():
    pytest.importorskip("FLife")

    fs = 1000.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 30 * t) + 0.05 * np.sin(2 * np.pi * 80 * t)

    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="disp")
    psd_params = PSDParams(method="welch", window="hann", detrend="constant")
    fds = compute_fds_spectral_time(x, fs, sn=sn, sdof=sdof, psd=psd_params, p_scale=1.0)

    f_psd = np.linspace(0.0, fs / 2.0, int(fs / 2) + 1)
    p0 = np.ones_like(f_psd) * 1e-6
    params = _build_params()

    out = invert_fds_iterative_spectral(
        fds,
        f_psd_hz=f_psd,
        psd_seed=p0,
        duration_s=2.0,
        sn=sn,
        sdof=sdof,
        p_scale=1.0,
        params=params,
    )

    usage = out.meta["param_usage"]
    assert usage["engine"] == "spectral"
    assert "tail_cap_ratio" in usage["used_fields"]
    assert usage["ignored_fields"] == []
    assert usage["used"]["tail_cap_ratio"] == pytest.approx(params.tail_cap_ratio)
    assert usage["used"]["post_refine_iters"] == params.post_refine_iters
    assert usage["effective"]["smooth_window_bins"] == 11
    assert usage["effective"]["post_smooth_window_bins"] == 7
