import numpy as np
from fdscore import SNParams, SDOFParams, compute_fds_time

def test_metrics_run():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0/fs)
    x = 0.05*np.sin(2*np.pi*30*t) + 0.02*np.sin(2*np.pi*80*t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)

    for metric in ["pv", "disp", "vel", "acc"]:
        sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric=metric)
        fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", p_scale=6500.0, batch_size=8)
        assert fds.f.shape == fds.damage.shape
        assert np.all(np.isfinite(fds.damage))
        assert np.all(fds.damage >= 0)
