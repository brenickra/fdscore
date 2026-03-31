import numpy as np
from fdscore import SNParams, SDOFParams, compute_fds_time

def test_compute_fds_time_runs_and_shapes():
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0/fs)
    x = 0.1*np.sin(2*np.pi*20*t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", p_scale=6500.0)
    assert fds.f.shape == fds.damage.shape
    assert np.all(fds.f > 0)
    assert np.all(fds.damage >= 0)
