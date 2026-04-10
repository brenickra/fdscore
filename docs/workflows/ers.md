# ERS Workflows

This page covers the generic extreme response spectrum workflows in `fdscore`.

## Time-Domain ERS

`compute_ers_time(...)` is the generic FFT-domain ERS engine. It reconstructs
the SDOF responses on the requested oscillator grid and extracts the peak
response for the selected metric.

```python
from fdscore import SDOFParams, compute_ers_time

sdof_ers = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=200.0, df=5.0)
ers = compute_ers_time(x, fs, sdof=sdof_ers, detrend="linear", batch_size=64)
```

When the metric and sampling setup match, `compute_ers_time(...)` can reuse a
compatible `FDSTimePlan`.

## Deterministic ERS

The deterministic harmonic helpers parallel the deterministic FDS workflows.

### Single sine

```python
from fdscore import compute_ers_sine

ers = compute_ers_sine(freq_hz=80.0, amp=2.0, sdof=sdof_ers, input_motion="acc")
```

### Dwell profiles

```python
from fdscore import SineDwellSegment, compute_ers_dwell_profile

segments = [
    SineDwellSegment(freq_hz=40.0, amp=1.5, duration_s=600.0),
    SineDwellSegment(freq_hz=80.0, amp=2.0, duration_s=300.0),
]

ers_profile = compute_ers_dwell_profile(segments, sdof=sdof_ers)
```

### Dwell-discretized sine sweeps

```python
from fdscore import compute_ers_sine_sweep

ers_sweep = compute_ers_sine_sweep(
    f_start_hz=20.0,
    f_stop_hz=200.0,
    amp=2.0,
    duration_s=180.0,
    sdof=sdof_ers,
    spacing="log",
    n_steps=200,
)
```

## ERS Envelope Composition

ERS mission composition is based on pointwise envelope, not summation.

```python
from fdscore import envelope_ers

ers_env = envelope_ers([ers_run_1, ers_run_2, ers_run_3])
```

This differs intentionally from FDS, where mission composition follows damage
summation.

## ERS vs Shock Spectra

`compute_ers_time(...)` is the generic ERS engine. Shock-oriented workflows
such as `compute_srs_time(...)` and `compute_pvss_time(...)` use a separate
recursive backend tailored to transient shock analysis and are documented in
[workflows/shock.md](shock.md).
