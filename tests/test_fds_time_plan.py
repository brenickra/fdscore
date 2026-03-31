import numpy as np
import pytest

from fdscore import (
    SNParams,
    SDOFParams,
    ValidationError,
    compute_fds_time,
    prepare_fds_time_plan,
)


def test_compute_fds_time_plan_matches_no_plan():
    fs = 1000.0
    t = np.arange(0, 3.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 20 * t) + 0.05 * np.sin(2 * np.pi * 70 * t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=100.0, df=5.0, metric="pv")

    plan = prepare_fds_time_plan(fs=fs, n_samples=x.size, sdof=sdof)
    fds_ref = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", p_scale=6500.0, batch_size=16)
    fds_plan = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", p_scale=6500.0, batch_size=16, plan=plan)

    assert np.allclose(fds_ref.f, fds_plan.f, rtol=0.0, atol=0.0)
    assert np.allclose(fds_ref.damage, fds_plan.damage, rtol=1e-12, atol=1e-15)


def test_compute_fds_time_plan_validation():
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = 0.08 * np.sin(2 * np.pi * 30 * t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=120.0, df=10.0, metric="pv")

    plan_wrong_n = prepare_fds_time_plan(fs=fs, n_samples=x.size + 1, sdof=sdof)
    with pytest.raises(ValidationError):
        compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none", p_scale=6500.0, plan=plan_wrong_n)
