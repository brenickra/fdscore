import numpy as np
import pytest

from fdscore import (
    RollingERSResult,
    SDOFParams,
    compute_pvss_time,
    compute_rolling_pvss_time,
    compute_rolling_srs_time,
    compute_srs_time,
    detect_shock_events,
)
from fdscore.validate import ValidationError


def _three_pulse_signal(fs: float = 2000.0, duration_s: float = 1.0) -> tuple[np.ndarray, float]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = np.zeros_like(t)
    for center_s, amp in ((0.20, 2.0), (0.50, -3.0), (0.80, 1.5)):
        mask = np.abs(t - center_s) <= 0.005
        phase = (t[mask] - (center_s - 0.005)) / 0.01
        x[mask] += amp * np.sin(np.pi * phase)
    return x, fs


def test_compute_rolling_srs_time_matches_per_event_direct_calls():
    x, fs = _three_pulse_signal()
    events = detect_shock_events(x, fs, detrend="none", threshold_value=1.0, min_separation_s=0.10, window_s=0.04)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    rolling = compute_rolling_srs_time(x, fs, sdof, events, detrend="none", peak_mode="abs")

    assert isinstance(rolling, RollingERSResult)
    assert rolling.response.shape == (len(events.events), rolling.f.size)
    assert np.allclose(rolling.t_center_s, [ev.peak_time_s for ev in events.events])

    for i, ev in enumerate(events.events):
        seg = x[ev.start_index:ev.stop_index]
        direct = compute_srs_time(seg, fs, sdof, detrend="none", peak_mode="abs")
        assert np.allclose(rolling.f, direct.f)
        assert np.allclose(rolling.response[i], direct.response)


def test_compute_rolling_pvss_time_matches_per_event_direct_calls():
    x, fs = _three_pulse_signal()
    events = detect_shock_events(x, fs, detrend="none", threshold_value=1.0, min_separation_s=0.10, window_s=0.04)
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=100.0, df=10.0)

    rolling = compute_rolling_pvss_time(x, fs, sdof, events, detrend="none", peak_mode="neg")

    assert isinstance(rolling, RollingERSResult)
    assert rolling.meta["peak_mode"] == "neg"
    assert rolling.meta["ers_kind"] == "pseudo_velocity_shock_spectrum"

    for i, ev in enumerate(events.events):
        seg = x[ev.start_index:ev.stop_index]
        direct = compute_pvss_time(seg, fs, sdof, detrend="none", peak_mode="neg")
        assert np.allclose(rolling.response[i], direct.response)


def test_compute_rolling_shock_time_handles_empty_event_set():
    x = np.zeros(2048, dtype=float)
    fs = 2048.0
    events = detect_shock_events(x, fs, detrend="none", threshold_value=0.5)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    rolling = compute_rolling_srs_time(x, fs, sdof, events, detrend="none")

    assert rolling.response.shape == (0, rolling.f.size)
    assert rolling.t_center_s.shape == (0,)
    assert rolling.meta["n_windows"] == 0


def test_compute_rolling_shock_time_rejects_both_peak_mode():
    x, fs = _three_pulse_signal()
    events = detect_shock_events(x, fs, detrend="none", threshold_value=1.0)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match="rolling shock spectra currently support peak_mode"):
        compute_rolling_srs_time(x, fs, sdof, events, peak_mode="both")


def test_compute_rolling_shock_time_validates_event_set_alignment():
    x, fs = _three_pulse_signal()
    events = detect_shock_events(x, fs, detrend="none", threshold_value=1.0)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match="events.fs must match fs"):
        compute_rolling_srs_time(x, fs + 1.0, sdof, events)

    with pytest.raises(ValidationError, match=r"events.n_samples must match len\(x\)"):
        compute_rolling_srs_time(x[:-1], fs, sdof, events)


def test_compute_rolling_shock_time_rejects_event_windows_shorter_than_four_samples():
    x, fs = _three_pulse_signal(fs=1000.0)
    events = detect_shock_events(
        x,
        fs,
        detrend="none",
        threshold_value=1.0,
        min_separation_s=0.10,
        window_s=0.001,
    )
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)

    with pytest.raises(ValidationError, match="at least 4 samples"):
        compute_rolling_srs_time(x, fs, sdof, events, detrend="none")

