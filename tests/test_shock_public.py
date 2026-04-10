import numpy as np
import pytest

from fdscore import SDOFParams, ShockSpectrumPair, compute_pvss_time, compute_srs_time
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


def test_compute_srs_time_rejects_non_numeric_fs_with_validation_error():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match=r"fs must be finite and > 0"):
        compute_srs_time(np.zeros(8, dtype=float), "abc", sdof, detrend="none")


def test_compute_srs_time_records_nyquist_clipping_in_provenance():
    x = np.zeros(256, dtype=float)
    sdof = SDOFParams(q=10.0, metric="acc", f=np.array([10.0, 20.0, 40.0, 60.0]))

    srs = compute_srs_time(x, 100.0, sdof, detrend="none", strict_nyquist=False)

    prov = srs.meta["provenance"]
    assert prov["strict_nyquist"] is False
    assert prov["nyquist_clipped"] is True
    assert prov["nyquist_hz"] == pytest.approx(50.0)
    assert prov["requested_frequency_count"] == 4
    assert prov["returned_frequency_count"] == 3
    assert prov["requested_fmax_hz"] == pytest.approx(60.0)
    assert prov["returned_fmax_hz"] == pytest.approx(40.0)


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

def test_compute_srs_time_both_returns_sided_pair():
    x, fs = _pulse_signal()
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=200.0, df=10.0)

    pair = compute_srs_time(x, fs, sdof, detrend="none", peak_mode="both")

    assert isinstance(pair, ShockSpectrumPair)
    assert pair.meta["peak_mode"] == "both"
    assert pair.meta["ers_kind"] == "shock_response_spectrum"
    assert pair.neg.meta["peak_mode"] == "neg"
    assert pair.pos.meta["peak_mode"] == "pos"
    assert np.allclose(pair.neg.f, pair.pos.f)


def test_compute_pvss_time_both_returns_sided_pair():
    x, fs = _pulse_signal()
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=10.0)

    pair = compute_pvss_time(x, fs, sdof, detrend="none", peak_mode="both")

    assert isinstance(pair, ShockSpectrumPair)
    assert pair.meta["peak_mode"] == "both"
    assert pair.meta["ers_kind"] == "pseudo_velocity_shock_spectrum"
    assert pair.neg.meta["peak_mode"] == "neg"
    assert pair.pos.meta["peak_mode"] == "pos"
    assert np.allclose(pair.neg.f, pair.pos.f)


def test_public_shock_wrappers_both_match_private_engine_exactly():
    x, fs = _pulse_signal(fs=8192.0, duration_s=1.5)
    sdof_srs = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=300.0, df=10.0)
    sdof_pvss = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=300.0, df=10.0)
    zeta = 1.0 / (2.0 * sdof_srs.q)
    f0 = np.arange(10.0, 301.0, 10.0)
    x_centered = x - np.median(x)

    srs_pair = compute_srs_time(x, fs, sdof_srs, detrend="median", peak_mode="both")
    pvss_pair = compute_pvss_time(x, fs, sdof_pvss, detrend="median", peak_mode="both")

    expected_srs = _shock_response_spectrum_iir(x_centered, fs=fs, f0_hz=f0, zeta=zeta, metric="acc", peak_mode="both")
    expected_pvss = _shock_response_spectrum_iir(x_centered, fs=fs, f0_hz=f0, zeta=zeta, metric="pv", peak_mode="both")

    assert np.allclose(srs_pair.neg.f, f0)
    assert np.allclose(srs_pair.pos.f, f0)
    assert np.allclose(pvss_pair.neg.f, f0)
    assert np.allclose(pvss_pair.pos.f, f0)
    assert np.allclose(srs_pair.neg.response, expected_srs[0])
    assert np.allclose(srs_pair.pos.response, expected_srs[1])
    assert np.allclose(pvss_pair.neg.response, expected_pvss[0])
    assert np.allclose(pvss_pair.pos.response, expected_pvss[1])

