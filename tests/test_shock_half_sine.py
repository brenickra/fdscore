import numpy as np
import pytest

from fdscore import (
    ERSResult,
    HalfSinePulse,
    SDOFParams,
    compute_pvss_time,
    fit_half_sine_to_pvss,
    synthesize_half_sine_pulse,
)
from fdscore.validate import ValidationError


def _shock_grid() -> np.ndarray:
    return np.geomspace(5.0, 2000.0, 240)


def test_synthesize_half_sine_pulse_generates_expected_waveform_and_polarity():
    pulse_pos = HalfSinePulse(amplitude=12.0, duration_s=0.010, polarity="pos")
    pulse_neg = HalfSinePulse(amplitude=12.0, duration_s=0.010, polarity="neg")

    x_pos = synthesize_half_sine_pulse(pulse_pos, 10000.0, total_duration_s=0.050, t_start_s=0.010)
    x_neg = synthesize_half_sine_pulse(pulse_neg, 10000.0, total_duration_s=0.050, t_start_s=0.010)

    assert x_pos.shape == x_neg.shape
    assert np.max(x_pos) > 0.0
    assert np.min(x_pos) >= -1e-12
    assert np.min(x_neg) < 0.0
    assert np.max(x_neg) <= 1e-12
    assert np.allclose(x_neg, -x_pos)
    assert np.allclose(x_pos[:100], 0.0)


def test_fit_half_sine_to_pvss_requires_pvss_kind():
    f = np.array([10.0, 20.0, 40.0])
    bad = ERSResult(
        f=f,
        response=np.ones_like(f),
        meta={"compat": {"engine": "x", "metric": "pv", "q": 10.0, "peak_mode": "abs", "ers_kind": "response_spectrum"}},
    )

    with pytest.raises(ValidationError, match="pseudo_velocity_shock_spectrum"):
        fit_half_sine_to_pvss(bad)


def test_fit_half_sine_to_pvss_returns_positive_finite_pulse():
    fs = 20000.0
    f = _shock_grid()
    pulse = HalfSinePulse(amplitude=18.0, duration_s=0.012, polarity="pos")
    x = synthesize_half_sine_pulse(pulse, fs, total_duration_s=0.400, t_start_s=0.080)
    pvss = compute_pvss_time(x, fs, SDOFParams(q=10.0, metric="pv", f=f), detrend="none")

    fitted = fit_half_sine_to_pvss(pvss)

    assert fitted.polarity == "pos"
    assert np.isfinite(fitted.amplitude) and fitted.amplitude > 0.0
    assert np.isfinite(fitted.duration_s) and fitted.duration_s > 0.0
    assert fitted.meta["source"] == "fit_half_sine_to_pvss"
    assert fitted.meta["fit_from"]["q"] == 10.0
    assert fitted.meta["fit_from"]["zeta"] == pytest.approx(0.05)


def test_fit_half_sine_to_pvss_fitted_pulse_tracks_target_pvss():
    fs = 20000.0
    f = _shock_grid()
    pulse = HalfSinePulse(amplitude=22.0, duration_s=0.008, polarity="pos")
    x = synthesize_half_sine_pulse(pulse, fs, total_duration_s=0.300, t_start_s=0.060)
    target = compute_pvss_time(x, fs, SDOFParams(q=10.0, metric="pv", f=f), detrend="none")

    fitted = fit_half_sine_to_pvss(target, polarity="neg")
    x_fit = synthesize_half_sine_pulse(fitted, fs, total_duration_s=0.300, t_start_s=0.060)
    fitted_pvss = compute_pvss_time(x_fit, fs, SDOFParams(q=10.0, metric="pv", f=f), detrend="none")

    ratio = fitted_pvss.response / np.maximum(target.response, 1e-12)
    assert fitted.polarity == "neg"
    assert np.min(ratio) > 0.94
    assert float(np.median(ratio)) > 0.98


def test_synthesize_half_sine_pulse_validates_inputs():
    with pytest.raises(ValidationError, match="pulse.amplitude"):
        synthesize_half_sine_pulse(HalfSinePulse(amplitude=0.0, duration_s=0.01), 1000.0)

    with pytest.raises(ValidationError, match="total_duration_s"):
        synthesize_half_sine_pulse(HalfSinePulse(amplitude=1.0, duration_s=0.01), 1000.0, total_duration_s=0.005)

    with pytest.raises(ValidationError, match="fs must be finite and > 0"):
        synthesize_half_sine_pulse(HalfSinePulse(amplitude=1.0, duration_s=0.01), 0.0)
