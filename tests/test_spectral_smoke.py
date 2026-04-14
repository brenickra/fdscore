import numpy as np

from fdscore import SNParams, SDOFParams, PSDParams, compute_fds_spectral_time

def test_spectral_smoke():
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0/fs)
    x = 0.1*np.sin(2*np.pi*30*t) + 0.05*np.sin(2*np.pi*80*t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=100.0, df=10.0, metric="disp")
    psd = PSDParams(method="welch", window="hann", detrend="constant")

    fds = compute_fds_spectral_time(x, fs, sn=sn, sdof=sdof, psd=psd, p_scale=1.0)
    assert fds.f.shape == fds.damage.shape
    assert np.all(np.isfinite(fds.damage))
    assert np.all(fds.damage >= 0)
