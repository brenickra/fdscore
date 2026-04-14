import numpy as np
import pytest

from fdscore import (
    PSDParams,
    SDOFParams,
    ValidationError,
    compute_ers_spectral_psd,
    compute_ers_spectral_time,
    compute_psd_welch,
)


def test_compute_ers_spectral_psd_zero_input_returns_zero():
    f = np.arange(5.0, 405.0, 5.0)
    p = np.zeros_like(f)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=400.0, df=5.0)

    ers = compute_ers_spectral_psd(f, p, duration_s=120.0, sdof=sdof)

    assert np.allclose(ers.response, 0.0)
    assert ers.meta["provenance"]["source"] == "compute_ers_spectral_psd"
    assert ers.meta["compat"]["ers_kind"] == "random_extreme_response_spectrum"


@pytest.mark.parametrize("metric", ["acc", "vel", "disp", "pv"])
def test_compute_ers_spectral_psd_scales_with_square_root_of_psd(metric: str):
    f = np.arange(5.0, 405.0, 5.0)
    p = 0.02 + 0.0001 * f
    alpha = 9.0
    sdof = SDOFParams(q=10.0, metric=metric, fmin=5.0, fmax=400.0, df=5.0)

    ers_1 = compute_ers_spectral_psd(f, p, duration_s=300.0, sdof=sdof)
    ers_9 = compute_ers_spectral_psd(f, alpha * p, duration_s=300.0, sdof=sdof)

    assert np.allclose(ers_9.response, np.sqrt(alpha) * ers_1.response, rtol=1e-10, atol=1e-12)


def test_compute_ers_spectral_psd_rejects_invalid_inputs():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=100.0, df=5.0)

    with pytest.raises(ValidationError, match="duration_s must be finite and > 0"):
        compute_ers_spectral_psd(np.array([5.0, 10.0]), np.array([1.0, 1.0]), duration_s=0.0, sdof=sdof)

    with pytest.raises(ValidationError, match="strictly increasing"):
        compute_ers_spectral_psd(np.array([5.0, 5.0]), np.array([1.0, 1.0]), duration_s=10.0, sdof=sdof)

    with pytest.raises(ValidationError, match="negative values below numerical tolerance"):
        compute_ers_spectral_psd(np.array([5.0, 10.0]), np.array([1.0, -1.0]), duration_s=10.0, sdof=sdof)


def test_compute_ers_spectral_time_requires_onesided_psd():
    x = np.linspace(0.0, 1.0, 1024, dtype=float)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=100.0, df=5.0)
    psd = PSDParams(onesided=False)

    with pytest.raises(ValidationError, match="requires PSDParams.onesided=True"):
        compute_ers_spectral_time(x, 512.0, sdof=sdof, psd=psd)


def test_compute_ers_spectral_time_matches_direct_psd_route():
    rng = np.random.default_rng(123)
    x = rng.normal(size=4096)
    fs = 512.0
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=150.0, df=5.0)
    psd = PSDParams(window="hann", nperseg=512, noverlap=256, detrend="constant", onesided=True)

    ers_time = compute_ers_spectral_time(x, fs, sdof=sdof, psd=psd)
    f_psd, pyy = compute_psd_welch(x, fs=fs, psd=psd)
    ers_direct = compute_ers_spectral_psd(
        f_psd_hz=f_psd,
        psd_baseacc=pyy,
        duration_s=float(x.size) / fs,
        sdof=sdof,
        nyquist_hz=fs / 2.0,
        edge_correction=True,
    )

    assert np.allclose(ers_time.f, ers_direct.f, rtol=0.0, atol=0.0)
    assert np.allclose(ers_time.response, ers_direct.response, rtol=0.0, atol=1e-12)
    assert ers_time.meta["provenance"]["source"] == "compute_ers_spectral_time"


def test_compute_ers_spectral_psd_edge_correction_is_noop_when_psd_reaches_nyquist():
    f = np.arange(5.0, 505.0, 5.0)
    p = 0.01 + 0.00005 * f
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=400.0, df=5.0)

    ers_plain = compute_ers_spectral_psd(f, p, duration_s=120.0, sdof=sdof, nyquist_hz=500.0, edge_correction=False)
    ers_edge = compute_ers_spectral_psd(f, p, duration_s=120.0, sdof=sdof, nyquist_hz=500.0, edge_correction=True)

    assert np.allclose(ers_plain.response, ers_edge.response, rtol=0.0, atol=1e-12)
    assert ers_edge.meta["provenance"]["edge_correction_applied"] is False


def test_compute_ers_spectral_psd_edge_correction_lifts_only_high_frequency_tail():
    f = np.arange(5.0, 405.0, 5.0)
    p = np.full_like(f, 0.02)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=400.0, df=5.0)

    ers_plain = compute_ers_spectral_psd(f, p, duration_s=300.0, sdof=sdof, nyquist_hz=500.0, edge_correction=False)
    ers_edge = compute_ers_spectral_psd(f, p, duration_s=300.0, sdof=sdof, nyquist_hz=500.0, edge_correction=True)

    low = ers_edge.f <= 100.0
    high = ers_edge.f >= 350.0

    assert np.allclose(ers_edge.response[low], ers_plain.response[low], rtol=1e-3, atol=1e-12)
    assert np.all(ers_edge.response[high] >= ers_plain.response[high])
    assert np.median(ers_edge.response[high] / ers_plain.response[high]) > 1.0
    assert ers_edge.meta["provenance"]["edge_correction_applied"] is True
    assert ers_edge.meta["provenance"]["edge_corrected_oscillator_count"] > 0


def test_compute_ers_spectral_psd_records_expected_metadata():
    f = np.arange(5.0, 405.0, 5.0)
    p = 0.02 + 0.00002 * f
    sdof = SDOFParams(q=20.0, metric="pv", fmin=5.0, fmax=400.0, df=5.0)

    ers = compute_ers_spectral_psd(f, p, duration_s=60.0, sdof=sdof, nyquist_hz=500.0, edge_correction=True)

    compat = ers.meta["compat"]
    prov = ers.meta["provenance"]

    assert compat["engine"] == "spectral_random_psd"
    assert compat["metric"] == "pv"
    assert compat["peak_mode"] == "expected_gaussian_max"
    assert compat["ers_kind"] == "random_extreme_response_spectrum"
    assert prov["peak_model"] == "gaussian_davenport"
    assert prov["edge_correction_mode"] == "auto_bandwidth_taper"
    assert prov["edge_bandwidth_scale"] == pytest.approx(2.0)
    assert prov["nyquist_hz"] == pytest.approx(500.0)
