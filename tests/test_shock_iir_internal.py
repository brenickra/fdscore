import numpy as np
import pytest

from fdscore._shock_iir import (
    _extract_shock_peak,
    _shock_filter_coefficients,
    _shock_response_spectrum_iir,
    _shock_zero_padding_length,
)
from fdscore import SDOFParams, compute_ers_time
from fdscore.validate import ValidationError


def _half_sine(t: np.ndarray, start_s: float, duration_s: float, amp: float) -> np.ndarray:
    out = np.zeros_like(t, dtype=float)
    mask = (t >= start_s) & (t < start_s + duration_s)
    if np.any(mask):
        phase = np.pi * (t[mask] - start_s) / duration_s
        out[mask] = amp * np.sin(phase)
    return out


def test_shock_filter_coefficients_shape_and_finiteness() -> None:
    f0 = np.array([10.0, 25.0, 80.0])

    for metric in ("acc", "pv"):
        b, a = _shock_filter_coefficients(f0_hz=f0, zeta=0.05, dt=1.0 / 5000.0, metric=metric)
        assert b.shape == (3, 3)
        assert a.shape == (3, 3)
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(a))
        assert np.allclose(a[:, 0], 1.0)


def test_shock_filter_coefficients_reject_unsupported_metric() -> None:
    with pytest.raises(ValidationError, match="shock engine metric"):
        _shock_filter_coefficients(f0_hz=np.array([10.0]), zeta=0.05, dt=1.0 / 5000.0, metric="disp")  # type: ignore[arg-type]


def test_shock_zero_padding_length_depends_on_peak_mode() -> None:
    f0 = np.array([10.0, 20.0, 40.0])
    n_abs = _shock_zero_padding_length(fs=5000.0, f0_hz=f0, zeta=0.05, peak_mode="abs")
    n_pos = _shock_zero_padding_length(fs=5000.0, f0_hz=f0, zeta=0.05, peak_mode="pos")
    assert n_abs > 0
    assert n_pos > n_abs


def test_extract_shock_peak_contract() -> None:
    resp = np.array([0.0, 1.5, -0.4, 0.2])
    tail = np.array([-0.9, 0.7, 0.0])

    neg = _extract_shock_peak(resp, tail, peak_mode="neg")
    pos = _extract_shock_peak(resp, tail, peak_mode="pos")
    abs_peak = _extract_shock_peak(resp, tail, peak_mode="abs")
    both = _extract_shock_peak(resp, tail, peak_mode="both")

    assert np.isclose(float(neg), 0.9)
    assert np.isclose(float(pos), 1.5)
    assert np.isclose(float(abs_peak), 1.5)
    assert both.shape == (2,)
    assert np.allclose(both, np.array([0.9, 1.5]))


def test_shock_response_iir_zero_signal_returns_zero() -> None:
    x = np.zeros(1024)
    f0 = np.array([10.0, 25.0, 80.0])

    for metric in ("acc", "pv"):
        for peak_mode in ("abs", "pos", "neg"):
            out = _shock_response_spectrum_iir(x, fs=5000.0, f0_hz=f0, zeta=0.05, metric=metric, peak_mode=peak_mode)
            assert out.shape == (3,)
            assert np.allclose(out, 0.0)

        both = _shock_response_spectrum_iir(x, fs=5000.0, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="both")
        assert both.shape == (2, 3)
        assert np.allclose(both, 0.0)


def test_shock_response_iir_sidedness_and_polarity_symmetry() -> None:
    fs = 5000.0
    t = np.arange(int(fs * 0.8), dtype=float) / fs
    x = _half_sine(t, 0.18, 0.010, 8.0) + _half_sine(t, 0.44, 0.008, -5.0)
    f0 = np.geomspace(8.0, 300.0, 32)

    for metric in ("acc", "pv"):
        pos = _shock_response_spectrum_iir(x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="pos")
        neg = _shock_response_spectrum_iir(x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="neg")
        abs_peak = _shock_response_spectrum_iir(x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="abs")
        both = _shock_response_spectrum_iir(x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="both")

        assert np.allclose(abs_peak, np.maximum(pos, neg))
        assert np.allclose(both[0], neg)
        assert np.allclose(both[1], pos)

        pos_flipped = _shock_response_spectrum_iir(-x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="pos")
        neg_flipped = _shock_response_spectrum_iir(-x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="neg")
        assert np.allclose(pos_flipped, neg, rtol=1e-10, atol=1e-12)
        assert np.allclose(neg_flipped, pos, rtol=1e-10, atol=1e-12)


def test_shock_response_iir_validates_inputs() -> None:
    x = np.ones(64)
    f0 = np.array([10.0, 25.0])

    with pytest.raises(ValidationError, match="shock engine metric"):
        _shock_response_spectrum_iir(x, fs=5000.0, f0_hz=f0, zeta=0.05, metric="disp", peak_mode="abs")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="peak_mode"):
        _shock_response_spectrum_iir(x, fs=5000.0, f0_hz=f0, zeta=0.05, metric="acc", peak_mode="foo")
    with pytest.raises(ValidationError, match="zeta"):
        _shock_response_spectrum_iir(x, fs=5000.0, f0_hz=f0, zeta=1.0, metric="acc", peak_mode="abs")


def test_shock_response_iir_stays_close_to_current_ers_fft_on_clean_pulse() -> None:
    fs = 20000.0
    t = np.arange(int(fs * 0.8), dtype=float) / fs
    x = _half_sine(t, 0.15, 0.008, 25.0)
    f0 = np.geomspace(5.0, 1000.0, 80)

    for metric in ("acc", "pv"):
        shock = _shock_response_spectrum_iir(x, fs=fs, f0_hz=f0, zeta=0.05, metric=metric, peak_mode="abs")
        ers = compute_ers_time(
            x,
            fs,
            sdof=SDOFParams(q=10.0, metric=metric, f=f0),
            detrend="none",
            peak_mode="abs",
            batch_size=64,
        ).response
        rel = np.abs(shock - ers) / np.maximum(np.abs(ers), 1e-12)
        assert float(np.median(rel)) < 0.05
