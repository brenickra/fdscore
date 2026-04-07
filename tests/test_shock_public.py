import numpy as np
import pytest

from fdscore import SDOFParams, compute_pvss_time, compute_srs_time
from fdscore._shock_iir import _shock_response_spectrum_iir
from fdscore.validate import ValidationError


def _pulse_signal(fs: float = 4096.0, duration_s: float = 1.0) -> tuple[np.ndarray, float]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = np.zeros_like(t)
    mask = (t >= 0.2) & (t <= 0.21)
    x[mask] = np.sin(np.pi * (t[mask] - 0.2) / 0.01)
    return x, fs


def test_compute_srs_time_zero_signal_is_zero():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    x = np.zeros(2048, dtype=float)

    srs = compute_srs_time(x, 2048.0, sdof, detrend="none")

    assert np.allclose(srs.response, 0.0)
    assert srs.meta["provenance"]["source"] == "compute_srs_time"
    assert srs.meta["compat"]["ers_kind"] == "shock_response_spectrum"


def test_compute_pvss_time_zero_signal_is_zero():
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=100.0, df=10.0)
    x = np.zeros(2048, dtype=float)

    pvss = compute_pvss_time(x, 2048.0, sdof, detrend="none")

    assert np.allclose(pvss.response, 0.0)
    assert pvss.meta["provenance"]["source"] == "compute_pvss_time"
    assert pvss.meta["compat"]["ers_kind"] == "pseudo_velocity_shock_spectrum"


def test_shock_wrappers_require_expected_metric():
    x, fs = _pulse_signal()
    sdof_pv = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=100.0, df=10.0)
    sdof_acc = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match="sdof.metric must be 'acc'"):
        compute_srs_time(x, fs, sdof_pv)

    with pytest.raises(ValidationError, match="sdof.metric must be 'pv'"):
        compute_pvss_time(x, fs, sdof_acc)


def test_shock_wrappers_reject_public_peak_mode_both():
    x, fs = _pulse_signal()
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match="peak_mode must be one of"):
        compute_srs_time(x, fs, sdof, peak_mode="both")


def test_compute_srs_time_pos_neg_swap_under_signal_inversion():
    x, fs = _pulse_signal()
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=200.0, df=10.0)

    pos = compute_srs_time(x, fs, sdof, detrend="none", peak_mode="pos")
    neg = compute_srs_time(-x, fs, sdof, detrend="none", peak_mode="neg")

    assert np.allclose(pos.response, neg.response, rtol=1e-10, atol=1e-12)


def test_public_shock_wrappers_match_private_engine_exactly():
    x, fs = _pulse_signal(fs=8192.0, duration_s=1.5)
    sdof_srs = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=300.0, df=10.0)
    sdof_pvss = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=300.0, df=10.0)
    zeta = 1.0 / (2.0 * sdof_srs.q)
    f0 = np.arange(10.0, 301.0, 10.0)

    srs = compute_srs_time(x, fs, sdof_srs, detrend="median", peak_mode="abs")
    pvss = compute_pvss_time(x, fs, sdof_pvss, detrend="median", peak_mode="abs")

    x_centered = x - np.median(x)
    expected_srs = _shock_response_spectrum_iir(x_centered, fs=fs, f0_hz=f0, zeta=zeta, metric="acc", peak_mode="abs")
    expected_pvss = _shock_response_spectrum_iir(x_centered, fs=fs, f0_hz=f0, zeta=zeta, metric="pv", peak_mode="abs")

    assert np.allclose(srs.f, f0)
    assert np.allclose(pvss.f, f0)
    assert np.allclose(srs.response, expected_srs)
    assert np.allclose(pvss.response, expected_pvss)
