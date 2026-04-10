# Shock Workflows

`fdscore` provides dedicated transient shock workflows built on a recursive
IIR SDOF backend. These workflows are intentionally separate from the generic
FFT-domain ERS path.

## SRS and PVSS

Use `compute_srs_time(...)` for shock response spectra and
`compute_pvss_time(...)` for pseudo-velocity shock spectra.

```python
from fdscore import SDOFParams, compute_srs_time, compute_pvss_time

sdof_srs = SDOFParams(q=10.0, metric="acc", fmin=5.0, fmax=2000.0, df=5.0)
sdof_pvss = SDOFParams(q=10.0, metric="pv", fmin=5.0, fmax=2000.0, df=5.0)

srs = compute_srs_time(x, fs, sdof=sdof_srs, detrend="median", peak_mode="abs")
pvss = compute_pvss_time(x, fs, sdof=sdof_pvss, detrend="median", peak_mode="both")
```

Important constraints:

- `compute_srs_time(...)` requires `sdof.metric="acc"`.
- `compute_pvss_time(...)` requires `sdof.metric="pv"`.
- both support `peak_mode="abs"`, `"pos"`, `"neg"`, or `"both"`.
- `peak_mode="both"` returns `ShockSpectrumPair`.

For short extracted shock windows, `detrend="median"` is often a practical
starting point.

## Event Detection

Detected-event workflows are useful when one long signal contains multiple
transient shocks.

```python
from fdscore import detect_shock_events

events = detect_shock_events(
    x,
    fs,
    detrend="median",
    threshold_reference="rms",
    threshold_multiplier=5.0,
    min_separation_s=0.050,
    window_s=0.040,
)
```

The returned `ShockEventSet` stores the selected windows and the detector
settings used to produce them.

## Rolling SRS and PVSS

Once events are detected, each event window can be evaluated independently.

```python
from fdscore import compute_rolling_srs_time, compute_rolling_pvss_time

rolling_srs = compute_rolling_srs_time(
    x,
    fs,
    sdof=sdof_srs,
    events=events,
    detrend="none",
    peak_mode="abs",
)

rolling_pvss = compute_rolling_pvss_time(
    x,
    fs,
    sdof=sdof_pvss,
    events=events,
    detrend="none",
    peak_mode="abs",
)
```

The returned `RollingERSResult.response` matrix has shape
`(n_events, n_freq)`.

## Shock Envelope Helpers

Mission-style shock composition is handled explicitly through sided envelope
helpers.

```python
from fdscore import envelope_srs, envelope_pvss

srs_env = envelope_srs([srs_run_1, srs_run_2, srs_run_3])
pvss_env = envelope_pvss([pvss_run_1, pvss_run_2, pvss_run_3])
```

For `peak_mode="both"`, these helpers envelope the negative and positive sides
separately and return another `ShockSpectrumPair`.

## Half-Sine Reduction

PVSS can be reduced to an equivalent half-sine pulse for requirement
simplification or bench setup studies.

```python
from fdscore import fit_half_sine_to_pvss, synthesize_half_sine_pulse

pulse = fit_half_sine_to_pvss(pvss_abs)
x_half_sine = synthesize_half_sine_pulse(
    pulse,
    fs=20000.0,
    total_duration_s=0.100,
    t_start_s=0.010,
)
```

This fit is an enveloping approximation derived from the PVSS. It is not
intended to reconstruct the original measured pulse exactly.
