import numpy as np
import pytest

from fdscore import ShockEventSet, detect_shock_events
from fdscore.validate import ValidationError


def _three_pulse_signal(fs: float = 2000.0, duration_s: float = 1.0) -> tuple[np.ndarray, float]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = np.zeros_like(t)
    for center_s, amp in ((0.20, 2.0), (0.50, -3.0), (0.80, 1.5)):
        mask = np.abs(t - center_s) <= 0.005
        phase = (t[mask] - (center_s - 0.005)) / 0.01
        x[mask] += amp * np.sin(np.pi * phase)
    return x, fs


def test_detect_shock_events_returns_empty_set_for_zero_signal():
    x = np.zeros(2048, dtype=float)

    events = detect_shock_events(x, 2048.0, detrend="none", threshold_value=0.5)

    assert isinstance(events, ShockEventSet)
    assert events.events == ()
    assert events.meta["source"] == "detect_shock_events"


def test_detect_shock_events_finds_known_abs_events():
    x, fs = _three_pulse_signal()

    events = detect_shock_events(
        x,
        fs,
        detrend="none",
        polarity="abs",
        threshold_value=1.0,
        min_separation_s=0.10,
        window_s=0.04,
    )

    assert len(events.events) == 3
    peak_times = np.array([ev.peak_time_s for ev in events.events])
    assert np.allclose(peak_times, [0.2, 0.5, 0.8], atol=1.0 / fs)
    assert [ev.polarity for ev in events.events] == ["pos", "neg", "pos"]


def test_detect_shock_events_filters_by_requested_polarity():
    x, fs = _three_pulse_signal()

    pos_events = detect_shock_events(x, fs, detrend="none", polarity="pos", threshold_value=1.0, min_separation_s=0.10)
    neg_events = detect_shock_events(x, fs, detrend="none", polarity="neg", threshold_value=1.0, min_separation_s=0.10)

    assert len(pos_events.events) == 2
    assert all(ev.polarity == "pos" for ev in pos_events.events)
    assert len(neg_events.events) == 1
    assert neg_events.events[0].polarity == "neg"


def test_detect_shock_events_uses_threshold_reference_when_absolute_threshold_missing():
    x, fs = _three_pulse_signal()

    events = detect_shock_events(
        x,
        fs,
        detrend="none",
        polarity="abs",
        threshold_reference="rms",
        threshold_multiplier=2.5,
        min_separation_s=0.10,
    )

    assert len(events.events) >= 1
    assert events.meta["effective_threshold"] > 0.0


def test_detect_shock_events_window_is_clipped_at_signal_edges():
    fs = 1000.0
    t = np.arange(0.0, 0.5, 1.0 / fs)
    x = np.zeros_like(t)
    x[2] = 5.0

    events = detect_shock_events(x, fs, detrend="none", threshold_value=1.0, min_separation_s=0.05, window_s=0.10)

    assert len(events.events) == 1
    ev = events.events[0]
    assert ev.start_index == 0
    assert ev.stop_index <= x.size


def test_detect_shock_events_validates_inputs():
    x = np.zeros(16, dtype=float)

    with pytest.raises(ValidationError, match="polarity must be one of"):
        detect_shock_events(x, 1000.0, polarity="foo")

    with pytest.raises(ValidationError, match="threshold_reference must be one of"):
        detect_shock_events(x, 1000.0, threshold_reference="foo")

    with pytest.raises(ValidationError, match="threshold_multiplier must be finite and > 0"):
        detect_shock_events(x, 1000.0, threshold_multiplier=0.0)

    with pytest.raises(ValidationError, match="threshold_value must be finite and > 0"):
        detect_shock_events(x, 1000.0, threshold_value=0.0)
