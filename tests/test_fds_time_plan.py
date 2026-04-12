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
    fds_ref = compute_fds_time(
        x,
        fs,
        sn=sn,
        sdof=sdof,
        detrend="none",
        p_scale=6500.0,
        batch_size=16,
        engine="fft",
    )
    fds_plan = compute_fds_time(
        x,
        fs,
        sn=sn,
        sdof=sdof,
        detrend="none",
        p_scale=6500.0,
        batch_size=16,
        plan=plan,
        engine="fft",
    )

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
        compute_fds_time(
            x,
            fs,
            sn=sn,
            sdof=sdof,
            detrend="none",
            p_scale=6500.0,
            plan=plan_wrong_n,
            engine="fft",
        )

def test_prepare_fds_time_plan_accepts_numpy_integer_scalars():
    fs = 1000.0
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    plan = prepare_fds_time_plan(fs=fs, n_samples=np.int64(2048), sdof=sdof)

    assert plan.n_samples == 2048
    assert np.isclose(plan.fs, fs)


def test_prepare_fds_time_plan_rejects_non_numeric_fs_with_validation_error():
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    with pytest.raises(ValidationError, match=r"fs must be finite and > 0"):
        prepare_fds_time_plan(fs="abc", n_samples=2048, sdof=sdof)

def test_compute_fds_time_plan_accepts_small_zeta_drift():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 0.05 * np.sin(2 * np.pi * 20 * t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    plan = prepare_fds_time_plan(fs=fs, n_samples=x.size, sdof=sdof)
    object.__setattr__(plan, "zeta", float(plan.zeta) + 5e-13)

    fds = compute_fds_time(
        x,
        fs,
        sn=sn,
        sdof=sdof,
        detrend="none",
        p_scale=6500.0,
        plan=plan,
        engine="fft",
    )

    assert np.all(np.isfinite(fds.damage))


def test_compute_fds_time_plan_grid_mismatch_message_mentions_nyquist_clipping():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 0.05 * np.sin(2 * np.pi * 20 * t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    plan = prepare_fds_time_plan(fs=fs, n_samples=x.size, sdof=sdof)
    object.__setattr__(plan, "f", np.asarray(plan.f[:-1], dtype=float))
    object.__setattr__(plan, "H", np.asarray(plan.H[:-1], dtype=complex))

    with pytest.raises(ValidationError, match="strict_nyquist"):
        compute_fds_time(
            x,
            fs,
            sn=sn,
            sdof=sdof,
            detrend="none",
            p_scale=6500.0,
            plan=plan,
            engine="fft",
        )


def test_compute_fds_time_records_nyquist_clipping_in_provenance():
    x = np.zeros(256, dtype=float)
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, metric="pv", f=np.array([10.0, 20.0, 40.0, 60.0]))

    fds = compute_fds_time(x, 100.0, sn=sn, sdof=sdof, detrend="none", strict_nyquist=False)

    prov = fds.meta["provenance"]
    assert prov["strict_nyquist"] is False
    assert prov["nyquist_clipped"] is True
    assert prov["nyquist_hz"] == pytest.approx(50.0)
    assert prov["requested_frequency_count"] == 4
    assert prov["returned_frequency_count"] == 3
    assert prov["requested_fmax_hz"] == pytest.approx(60.0)
    assert prov["returned_fmax_hz"] == pytest.approx(40.0)


def test_compute_fds_time_plan_rejects_nonfinite_transfer_matrix():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 0.05 * np.sin(2 * np.pi * 20 * t)

    sn = SNParams(slope_k=3.0, ref_stress=86.0, ref_cycles=1e6, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    plan = prepare_fds_time_plan(fs=fs, n_samples=x.size, sdof=sdof)
    h_bad = np.asarray(plan.H).copy()
    h_bad[0, 0] = np.nan
    object.__setattr__(plan, "H", h_bad)

    with pytest.raises(ValidationError, match=r"FDSTimePlan\.H must contain only finite values"):
        compute_fds_time(
            x,
            fs,
            sn=sn,
            sdof=sdof,
            detrend="none",
            p_scale=6500.0,
            plan=plan,
            engine="fft",
        )

