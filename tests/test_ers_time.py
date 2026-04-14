import numpy as np
import pytest

from fdscore import ERSResult, SDOFParams, ValidationError, compute_ers_time, envelope_ers, prepare_fds_time_plan


def test_compute_ers_time_zero_signal_is_zero():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    x = np.zeros(1024, dtype=float)
    ers = compute_ers_time(x, 512.0, sdof, detrend="none")

    assert np.allclose(ers.response, 0.0)
    assert ers.meta["provenance"]["source"] == "compute_ers_time"


def test_compute_ers_time_scales_linearly_with_signal_amplitude():
    rng = np.random.default_rng(123)
    x = rng.normal(size=2048)
    sdof = SDOFParams(q=8.0, metric="pv", fmin=20.0, fmax=120.0, df=20.0)

    ers_1 = compute_ers_time(x, 512.0, sdof, detrend="none")
    ers_3 = compute_ers_time(3.0 * x, 512.0, sdof, detrend="none")

    assert np.allclose(ers_3.response, 3.0 * ers_1.response)


def test_compute_ers_time_incremental_zero_signal_is_zero():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    x = np.zeros(1024, dtype=float)
    ers = compute_ers_time(x, 512.0, sdof, detrend="none", engine="incremental")

    assert np.allclose(ers.response, 0.0)
    assert ers.meta["provenance"]["engine"] == "incremental"


def test_compute_ers_time_incremental_scales_linearly_with_signal_amplitude():
    rng = np.random.default_rng(321)
    x = rng.normal(size=2048)
    sdof = SDOFParams(q=8.0, metric="pv", fmin=20.0, fmax=120.0, df=20.0)

    ers_1 = compute_ers_time(x, 512.0, sdof, detrend="none", engine="incremental")
    ers_3 = compute_ers_time(3.0 * x, 512.0, sdof, detrend="none", engine="incremental")

    assert np.allclose(ers_3.response, 3.0 * ers_1.response)


def test_compute_ers_time_can_reuse_compatible_fds_time_plan():
    rng = np.random.default_rng(7)
    x = rng.normal(size=2048)
    sdof = SDOFParams(q=12.0, metric="acc", fmin=10.0, fmax=90.0, df=10.0)
    plan = prepare_fds_time_plan(fs=512.0, n_samples=x.size, sdof=sdof)

    ers_no_plan = compute_ers_time(x, 512.0, sdof, detrend="mean", engine="fft")
    ers_plan = compute_ers_time(x, 512.0, sdof, detrend="mean", plan=plan, engine="fft")

    assert np.allclose(ers_no_plan.response, ers_plan.response)
    assert ers_plan.meta["provenance"]["transfer_plan"] is True


def test_compute_ers_time_rejects_non_numeric_fs_with_validation_error():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match=r"fs must be finite and > 0"):
        compute_ers_time(np.zeros(8, dtype=float), "abc", sdof, detrend="none")


def test_compute_ers_time_rejects_invalid_engine():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match="engine must be one of"):
        compute_ers_time(np.zeros(8, dtype=float), 512.0, sdof, detrend="none", engine="bad")


def test_compute_ers_time_records_nyquist_clipping_in_provenance():
    sdof = SDOFParams(q=10.0, metric="acc", f=np.array([10.0, 20.0, 40.0, 60.0]))
    x = np.zeros(256, dtype=float)

    ers = compute_ers_time(x, 100.0, sdof, detrend="none", strict_nyquist=False)

    prov = ers.meta["provenance"]
    assert prov["strict_nyquist"] is False
    assert prov["nyquist_clipped"] is True
    assert prov["nyquist_hz"] == pytest.approx(50.0)
    assert prov["requested_frequency_count"] == 4
    assert prov["returned_frequency_count"] == 3
    assert prov["requested_fmax_hz"] == pytest.approx(60.0)
    assert prov["returned_fmax_hz"] == pytest.approx(40.0)


def test_compute_ers_time_incremental_records_engine_settings():
    rng = np.random.default_rng(11)
    x = rng.normal(size=2048)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=120.0, df=10.0)

    ers = compute_ers_time(
        x,
        512.0,
        sdof,
        detrend="mean",
        engine="incremental",
        zoh_r_max=0.25,
    )

    prov = ers.meta["provenance"]
    assert prov["engine"] == "incremental"
    assert prov["zoh_r_max"] == pytest.approx(0.25)


def test_compute_ers_time_incremental_matches_fft_for_bandlimited_signal():
    fs = 1000.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    x = (
        0.40 * np.sin(2.0 * np.pi * 18.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 65.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 110.0 * t)
    )
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=140.0, df=5.0)

    ers_fft = compute_ers_time(x, fs, sdof, detrend="none", engine="fft")
    ers_inc = compute_ers_time(x, fs, sdof, detrend="none", engine="incremental")

    assert np.allclose(ers_fft.f, ers_inc.f, rtol=0.0, atol=0.0)
    assert np.allclose(ers_inc.response, ers_fft.response, rtol=0.03, atol=1e-10)


def test_compute_ers_time_incremental_matches_fft_with_adaptive_upsampling():
    fs = 1000.0
    t = np.arange(0.0, 3.0, 1.0 / fs)
    x = (
        0.18 * np.sin(2.0 * np.pi * 35.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 240.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 320.0 * t)
    )
    sdof = SDOFParams(q=10.0, metric="pv", fmin=20.0, fmax=360.0, df=10.0)

    ers_fft = compute_ers_time(x, fs, sdof, detrend="none", engine="fft")
    ers_inc = compute_ers_time(x, fs, sdof, detrend="none", engine="incremental", zoh_r_max=0.2)

    assert np.allclose(ers_fft.f, ers_inc.f, rtol=0.0, atol=0.0)
    assert np.allclose(ers_inc.response, ers_fft.response, rtol=0.12, atol=1e-10)


def test_envelope_ers_with_time_histories_matches_pointwise_max():
    fs = 512.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    x1 = 0.6 * np.sin(2.0 * np.pi * 25.0 * t)
    x2 = 1.2 * np.sin(2.0 * np.pi * 60.0 * t)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    ers1 = compute_ers_time(x1, fs, sdof, detrend="none")
    ers2 = compute_ers_time(x2, fs, sdof, detrend="none")
    mission = envelope_ers([ers1, ers2])

    expected = np.maximum(ers1.response, ers2.response)
    assert np.allclose(mission.response, expected)


def test_compute_ers_time_rejects_nonfinite_reused_plan_matrix():
    rng = np.random.default_rng(9)
    x = rng.normal(size=2048)
    sdof = SDOFParams(q=12.0, metric="acc", fmin=10.0, fmax=90.0, df=10.0)
    plan = prepare_fds_time_plan(fs=512.0, n_samples=x.size, sdof=sdof)
    h_bad = np.asarray(plan.H).copy()
    h_bad[0, 0] = np.inf
    object.__setattr__(plan, "H", h_bad)

    with pytest.raises(ValidationError, match=r"FDSTimePlan\.H must contain only finite values"):
        compute_ers_time(x, 512.0, sdof, detrend="mean", plan=plan, engine="fft")


def test_envelope_ers_rejects_response_shape_broadcast():
    fs = 512.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    x1 = 0.6 * np.sin(2.0 * np.pi * 25.0 * t)
    x2 = 1.2 * np.sin(2.0 * np.pi * 60.0 * t)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    ers1 = compute_ers_time(x1, fs, sdof, detrend="none")
    ers2 = compute_ers_time(x2, fs, sdof, detrend="none")
    malformed = ERSResult(
        f=np.asarray(ers2.f, dtype=float),
        response=np.asarray(ers2.response, dtype=float).reshape(1, -1),
        meta=dict(ers2.meta),
    )

    with pytest.raises(ValidationError, match="response arrays must match the reference shape"):
        envelope_ers([ers1, malformed])
