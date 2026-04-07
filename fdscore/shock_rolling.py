from __future__ import annotations

import numpy as np

from .shock import compute_pvss_time, compute_srs_time
from .types import RollingERSResult, SDOFParams, ShockEventSet
from .validate import ValidationError


def _validate_rolling_inputs(
    *,
    x: np.ndarray,
    fs: float,
    events: ShockEventSet,
    peak_mode: str,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValidationError("x must be a 1D array with length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("x must contain only finite values.")
    if not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValidationError("fs must be finite and > 0.")
    if not isinstance(events, ShockEventSet):
        raise ValidationError("events must be a ShockEventSet.")
    if peak_mode not in ("abs", "pos", "neg"):
        raise ValidationError("rolling shock spectra currently support peak_mode 'abs', 'pos', or 'neg'.")
    if not np.isclose(float(events.fs), float(fs), rtol=0.0, atol=1e-12):
        raise ValidationError("events.fs must match fs.")
    if int(events.n_samples) != int(x.size):
        raise ValidationError("events.n_samples must match len(x).")
    return x


def _rolling_from_events(
    *,
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    events: ShockEventSet,
    detrend: str,
    strict_nyquist: bool,
    peak_mode: str,
    spectrum_fn,
    source: str,
    ers_kind: str,
) -> RollingERSResult:
    x = _validate_rolling_inputs(x=x, fs=fs, events=events, peak_mode=peak_mode)

    rows: list[np.ndarray] = []
    centers: list[float] = []
    f_ref: np.ndarray | None = None

    for ev in events.events:
        seg = x[int(ev.start_index):int(ev.stop_index)]
        spec = spectrum_fn(
            seg,
            fs,
            sdof,
            detrend=detrend,
            strict_nyquist=strict_nyquist,
            peak_mode=peak_mode,
        )
        if f_ref is None:
            f_ref = np.asarray(spec.f, dtype=float)
        else:
            if spec.f.shape != f_ref.shape or not np.allclose(spec.f, f_ref, rtol=0.0, atol=1e-12):
                raise ValidationError("Incompatible rolling frequency grid across event windows.")
        rows.append(np.asarray(spec.response, dtype=float))
        centers.append(float(ev.peak_time_s))

    if f_ref is None:
        # No events: still expose the validated oscillator grid shape via an empty stack.
        empty = spectrum_fn(
            np.zeros(max(4, min(8, int(x.size))), dtype=float),
            fs,
            sdof,
            detrend="none",
            strict_nyquist=strict_nyquist,
            peak_mode=peak_mode,
        )
        f_ref = np.asarray(empty.f, dtype=float)
        response = np.zeros((0, f_ref.size), dtype=float)
        t_center = np.zeros(0, dtype=float)
    else:
        response = np.vstack(rows)
        t_center = np.asarray(centers, dtype=float)

    meta = {
        "source": source,
        "metric": sdof.metric,
        "q": float(sdof.q),
        "peak_mode": peak_mode,
        "ers_kind": ers_kind,
        "n_windows": int(response.shape[0]),
        "event_detector": dict(events.meta or {}),
        "provenance": {
            "detrend": detrend,
            "strict_nyquist": bool(strict_nyquist),
        },
    }
    return RollingERSResult(f=f_ref, t_center_s=t_center, response=response, meta=meta)


def compute_rolling_srs_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    events: ShockEventSet,
    *,
    detrend: str = "median",
    strict_nyquist: bool = True,
    peak_mode: str = "abs",
) -> RollingERSResult:
    """Compute SRS on each detected shock-event window.

    Notes
    -----
    - Rolling shock spectra are currently event-window based.
    - `peak_mode` is limited to `abs`, `pos`, and `neg` in this API.
    - `sdof.metric` must be `"acc"`.
    """
    return _rolling_from_events(
        x=x,
        fs=fs,
        sdof=sdof,
        events=events,
        detrend=detrend,
        strict_nyquist=strict_nyquist,
        peak_mode=peak_mode,
        spectrum_fn=compute_srs_time,
        source="compute_rolling_srs_time",
        ers_kind="shock_response_spectrum",
    )


def compute_rolling_pvss_time(
    x: np.ndarray,
    fs: float,
    sdof: SDOFParams,
    events: ShockEventSet,
    *,
    detrend: str = "median",
    strict_nyquist: bool = True,
    peak_mode: str = "abs",
) -> RollingERSResult:
    """Compute PVSS on each detected shock-event window.

    Notes
    -----
    - Rolling shock spectra are currently event-window based.
    - `peak_mode` is limited to `abs`, `pos`, and `neg` in this API.
    - `sdof.metric` must be `"pv"`.
    """
    return _rolling_from_events(
        x=x,
        fs=fs,
        sdof=sdof,
        events=events,
        detrend=detrend,
        strict_nyquist=strict_nyquist,
        peak_mode=peak_mode,
        spectrum_fn=compute_pvss_time,
        source="compute_rolling_pvss_time",
        ers_kind="pseudo_velocity_shock_spectrum",
    )
