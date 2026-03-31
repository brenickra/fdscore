import numpy as np

from fdscore import (
    SNParams,
    SDOFParams,
    compute_fds_time,
    invert_fds_iterative_time,
    IterativeInversionParams,
)

def test_iterative_inversion_time_smoke():
    fs = 500.0
    t = np.arange(0, 2.0, 1.0/fs)
    x = 0.1*np.sin(2*np.pi*30*t) + 0.05*np.sin(2*np.pi*80*t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="pv")

    fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", p_scale=1.0, batch_size=16)

    f_psd = np.linspace(0.0, fs/2.0, int(fs/2)+1)
    P0 = np.ones_like(f_psd) * 1e-6

    params = IterativeInversionParams(iters=2, smooth_enabled=False, prior_blend=0.0)
    out = invert_fds_iterative_time(
        fds,
        f_psd_hz=f_psd,
        psd_seed=P0,
        fs=fs,
        duration_s=2.0,
        sn=sn,
        sdof=sdof,
        p_scale=1.0,
        params=params,
        n_realizations=1,
        seed=0,
    )
    assert out.f.shape == out.psd.shape
    assert np.all(np.isfinite(out.psd))
    assert np.all(out.psd >= 0)
    assert "diagnostics" in (out.meta or {})
