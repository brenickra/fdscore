import numpy as np
import pytest

from fdscore import (
    SNParams,
    SDOFParams,
    compute_fds_time,
    compute_fds_spectral_time,
    PSDParams,
    invert_fds_iterative_spectral,
    IterativeInversionParams,
    ValidationError,
)

def test_iterative_inversion_smoke():
    pytest.importorskip("FLife")

    fs = 1000.0
    t = np.arange(0, 2.0, 1.0/fs)
    x = 0.1*np.sin(2*np.pi*30*t) + 0.05*np.sin(2*np.pi*80*t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="disp")
    psd_params = PSDParams(method="welch", window="hann", detrend="constant")

    fds = compute_fds_spectral_time(x, fs, sn=sn, sdof=sdof, psd=psd_params, p_scale=1.0)

    # Seed PSD: flat low-level curve on f grid
    f_psd = np.linspace(0.0, 500.0, 501)
    P0 = np.ones_like(f_psd) * 1e-6

    params = IterativeInversionParams(iters=3, smooth_enabled=False, prior_blend=0.0)
    out = invert_fds_iterative_spectral(
        fds,
        f_psd_hz=f_psd,
        psd_seed=P0,
        duration_s=2.0,
        sn=sn,
        sdof=sdof,
        p_scale=1.0,
        params=params,
    )
    assert out.f.shape == out.psd.shape
    assert np.all(np.isfinite(out.psd))
    assert np.all(out.psd >= 0)
    assert "diagnostics" in (out.meta or {})


def test_iterative_inversion_spectral_rejects_nonpositive_q():
    fs = 500.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 30 * t) + 0.05 * np.sin(2 * np.pi * 80 * t)

    sn = SNParams(slope_k=3.0)
    target_sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="pv")
    bad_sdof = SDOFParams(q=0.0, fmin=10.0, fmax=100.0, df=10.0, metric="pv")
    target = compute_fds_time(x, fs, sn=sn, sdof=target_sdof, detrend="none", p_scale=1.0, batch_size=16)

    f_psd = np.linspace(0.0, fs / 2.0, int(fs / 2) + 1)
    P0 = np.ones_like(f_psd) * 1e-6
    params = IterativeInversionParams(iters=1, smooth_enabled=False, prior_blend=0.0)

    with pytest.raises(ValidationError, match=r"sdof\.q must be finite and > 0"):
        invert_fds_iterative_spectral(
            target,
            f_psd_hz=f_psd,
            psd_seed=P0,
            duration_s=2.0,
            sn=sn,
            sdof=bad_sdof,
            p_scale=1.0,
            params=params,
        )
