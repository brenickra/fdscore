import numpy as np
import pytest
from fdscore import SNParams, SDOFParams, compute_fds_time, sum_fds, ValidationError

def test_compat_signature_and_sum_guard():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0/fs)
    x = 0.1*np.sin(2*np.pi*20*t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6)
    sdof1 = SDOFParams(q=10.0, fmin=10.0, fmax=50.0, df=10.0, metric="pv")
    sdof2 = SDOFParams(q=20.0, fmin=10.0, fmax=50.0, df=10.0, metric="pv")

    fds1 = compute_fds_time(x, fs, sn=sn, sdof=sdof1, detrend="none", p_scale=6500.0, batch_size=8)
    fds2 = compute_fds_time(x, fs, sn=sn, sdof=sdof2, detrend="none", p_scale=6500.0, batch_size=8)

    assert "compat" in (fds1.meta or {})
    with pytest.raises(ValidationError):
        sum_fds([fds1, fds2])

def test_compat_metadata_does_not_embed_frequency_grid():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(256)
    fs = 512.0
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=40.0, df=5.0, metric="pv")

    fds = compute_fds_time(x, fs, sn=sn, sdof=sdof)
    compat = fds.meta["compat"]

    assert "f" not in compat
    assert compat["metric"] == "pv"
    assert compat["fds_kind"] == "damage_spectrum"

