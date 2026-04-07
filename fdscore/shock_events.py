from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks

from .shock import _preprocess_shock_signal
from .types import ShockEvent, ShockEventSet
from .validate import ValidationError


def _validate_event_detector_inputs(
    *,
    x: np.ndarray,
    fs: float,
    detrend: str,
    polarity: str,
    threshold_reference: str,
    threshold_multiplier: float,
    threshold_value: float | None,
    min_separation_s: float,
    window_s: float | None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValidationError("x must be a 1D array with length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("x must contain only finite values.")
    if not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValidationError("fs must be finite and > 0.")

    _preprocess_shock_signal(np.zeros(4, dtype=float), detrend=detrend)

    if polarity not in ("abs", "pos", "neg"):
        raise ValidationError("polarity must be one of: 'abs', 'pos', 'neg'.")
    if threshold_reference not in ("rms", "std", "peak"):
        raise ValidationError("threshold_reference must be one of: 'rms', 'std', 'peak'.")
    if not np.isfinite(threshold_multiplier) or float(threshold_multiplier) <= 0.0:
        raise ValidationError("threshold_multiplier must be finite and > 0.")
    if threshold_value is not None and (not np.isfinite(threshold_value) or float(threshold_value) <= 0.0):
        raise ValidationError("threshold_value must be finite and > 0 when provided.")
    if not np.isfinite(min_separation_s) or float(min_separation_s) < 0.0:
        raise ValidationError("min_separation_s must be finite and >= 0.")
    if window_s is not None and (not np.isfinite(window_s) or float(window_s) <= 0.0):
        raise ValidationError("window_s must be finite and > 0 when provided.")

    return x


def _event_threshold(y: np.ndarray, *, threshold_reference: str, threshold_multiplier: float, threshold_value: float | None) -> float:
    if threshold_value is not None:
        return float(threshold_value)

    if threshold_reference == "rms":
        ref = math.sqrt(float(np.mean(y * y)))
    elif threshold_reference == "std":
        ref = float(np.std(y))
    else:
        ref = float(np.max(np.abs(y)))

    return float(threshold_multiplier) * ref


def detect_shock_events(
    x: np.ndarray,
    fs: float,
    *,
    detrend: str = "median",
    polarity: str = "abs",
    threshold_reference: str = "rms",
    threshold_multiplier: float = 5.0,
    threshold_value: float | None = None,
    min_separation_s: float = 0.05,
    window_s: float | None = None,
) -> ShockEventSet:
    """Detect discrete shock events in a 1D acceleration time history.

    Notes
    -----
    - This is an axis-first detector for event-oriented shock workflows.
    - Detection is performed on a preprocessed 1D signal using `scipy.signal.find_peaks`.
    - `polarity` controls whether detection is based on absolute peaks, positive peaks,
      or negative peaks.
    """
    x = _validate_event_detector_inputs(
        x=x,
        fs=fs,
        detrend=detrend,
        polarity=polarity,
        threshold_reference=threshold_reference,
        threshold_multiplier=threshold_multiplier,
        threshold_value=threshold_value,
        min_separation_s=min_separation_s,
        window_s=window_s,
    )

    y = _preprocess_shock_signal(x, detrend=detrend)
    if polarity == "abs":
        detector = np.abs(y)
    elif polarity == "pos":
        detector = y
    else:
        detector = -y

    threshold = _event_threshold(
        y,
        threshold_reference=threshold_reference,
        threshold_multiplier=threshold_multiplier,
        threshold_value=threshold_value,
    )
    distance = max(1, int(math.floor(float(min_separation_s) * float(fs))) + 1)

    peaks, _ = find_peaks(detector, height=threshold, distance=distance)

    if window_s is None:
        half_window = max(0, int(math.floor(float(min_separation_s) * float(fs) / 2.0)))
    else:
        half_window = max(0, int(math.floor(float(window_s) * float(fs) / 2.0)))

    events: list[ShockEvent] = []
    for idx in peaks.tolist():
        start = max(0, idx - half_window)
        stop = min(int(y.size), idx + half_window + 1)
        peak_value = float(y[idx])
        events.append(
            ShockEvent(
                peak_index=int(idx),
                start_index=int(start),
                stop_index=int(stop),
                peak_time_s=float(idx / float(fs)),
                start_time_s=float(start / float(fs)),
                stop_time_s=float((stop - 1) / float(fs)),
                peak_value=peak_value,
                peak_abs=abs(peak_value),
                polarity="pos" if peak_value >= 0.0 else "neg",
            )
        )

    meta = {
        "source": "detect_shock_events",
        "detrend": detrend,
        "polarity": polarity,
        "threshold_reference": threshold_reference,
        "threshold_multiplier": float(threshold_multiplier),
        "threshold_value": None if threshold_value is None else float(threshold_value),
        "effective_threshold": float(threshold),
        "min_separation_s": float(min_separation_s),
        "window_s": float(window_s) if window_s is not None else float(min_separation_s),
    }
    return ShockEventSet(events=tuple(events), fs=float(fs), n_samples=int(y.size), meta=meta)
