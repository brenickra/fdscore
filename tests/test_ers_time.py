import numpy as np

from fdscore import SDOFParams, compute_ers_time, envelope_ers, prepare_fds_time_plan


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


def test_compute_ers_time_can_reuse_compatible_fds_time_plan():
    rng = np.random.default_rng(7)
    x = rng.normal(size=2048)
    sdof = SDOFParams(q=12.0, metric="acc", fmin=10.0, fmax=90.0, df=10.0)
    plan = prepare_fds_time_plan(fs=512.0, n_samples=x.size, sdof=sdof)

    ers_no_plan = compute_ers_time(x, 512.0, sdof, detrend="mean")
    ers_plan = compute_ers_time(x, 512.0, sdof, detrend="mean", plan=plan)

    assert np.allclose(ers_no_plan.response, ers_plan.response)
    assert ers_plan.meta["provenance"]["transfer_plan"] is True


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
