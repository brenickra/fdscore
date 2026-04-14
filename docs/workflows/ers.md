# ERS Workflows

This page covers the generic extreme response spectrum workflows in `fdscore`.

## Time-Domain ERS

`compute_ers_time(...)` is the generic time-domain ERS engine. It evaluates
the oscillator bank on a realized input history and returns the maximum
observed response for the selected metric.

```python
from fdscore import SDOFParams, compute_ers_time

sdof_ers = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=200.0, df=5.0)
ers = compute_ers_time(x, fs, sdof=sdof_ers, detrend="linear")
```

When the metric and sampling setup match, `compute_ers_time(...)` can reuse a
compatible `FDSTimePlan`.

## Spectral Random ERS

`compute_ers_spectral_psd(...)` computes a random-vibration ERS directly from a
one-sided acceleration PSD. Unlike `compute_ers_time(...)`, this is not the
maximum observed in a single realized record. It is the expected extreme
response of a stationary Gaussian process over a specified duration.

```python
from fdscore import SDOFParams, compute_ers_spectral_psd

sdof_ers = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=200.0, df=5.0)
ers = compute_ers_spectral_psd(
    f_psd_hz=f_psd,
    psd_baseacc=Pyy,
    duration_s=24.0 * 3600.0,
    sdof=sdof_ers,
    nyquist_hz=fs / 2.0,
)
```

The spectral route builds the response PSD of each oscillator, computes its
response moments, estimates a Gaussian peak rate, and returns the expected
maximum over `duration_s`.

For `metric="acc"`, the current implementation uses a Lalanne-style
relative-displacement/random-peak backbone internally because it better matches
classical random-vibration acceleration ERS practice. Other metrics use exact
response-PSD moments of the selected response quantity.

### Spectral convenience wrapper

`compute_ers_spectral_time(...)` is a convenience wrapper for the same random
ERS concept. It first estimates a one-sided PSD with Welch and then delegates
to `compute_ers_spectral_psd(...)`.

```python
from fdscore import PSDParams, compute_ers_spectral_time

psd = PSDParams(window="hann", nperseg=2048, noverlap=1024, onesided=True)
ers = compute_ers_spectral_time(
    x,
    fs,
    sdof=sdof_ers,
    psd=psd,
    duration_s=24.0 * 3600.0,
)
```

### Edge correction

By default, the spectral ERS route applies an automatic high-frequency edge
correction when the PSD is cropped below the original Nyquist limit. This is
important near the top of the oscillator grid because truncated PSD exports can
artificially suppress the response moments.

The correction is oscillator-local and based on the damping bandwidth:

- `delta_f = 2 * zeta * f_n`
- a raised-cosine taper is used from the last available PSD value down to zero
  over that span

When the PSD already reaches Nyquist, the correction is a no-op.

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

`compute_ers_time(...)` and the spectral/random ERS APIs are the generic ERS
family.

Shock-oriented workflows such as `compute_srs_time(...)` and
`compute_pvss_time(...)` use a separate recursive backend tailored to transient
shock analysis and are documented in [workflows/shock.md](shock.md).
