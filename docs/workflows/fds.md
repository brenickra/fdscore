# FDS Workflows

This page summarizes the main ways `fdscore` computes and combines Fatigue
Damage Spectra.

## Time-Domain FDS

`compute_fds_time(...)` is the main time-domain route. It reconstructs SDOF
responses in the FFT domain, performs rainflow counting on each response, and
accumulates damage with Miner rule.

```python
from fdscore import SNParams, SDOFParams, compute_fds_time

sn = SNParams.normalized(slope_k=4.0)
sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=10.0)

fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="linear", batch_size=64)
```

Use this route when you have a measured or synthesized acceleration time
history and want fatigue damage based on actual time-domain cycle counting.

### Reusing `FDSTimePlan`

When multiple signals share the same sampling setup, precomputing the transfer
plan avoids rebuilding the FFT-domain transfer matrix for every call.

```python
from fdscore import prepare_fds_time_plan

plan = prepare_fds_time_plan(fs=fs, n_samples=len(x), sdof=sdof)
fds_x = compute_fds_time(x, fs, sn=sn, sdof=sdof, plan=plan)
fds_y = compute_fds_time(y, fs, sn=sn, sdof=sdof, plan=plan)
```

## Spectral FDS

The spectral route uses an internal Dirlik-based fatigue estimator.

### From an explicit PSD

```python
from fdscore import compute_fds_spectral_psd

fds = compute_fds_spectral_psd(
    f_psd_hz,
    psd_values,
    duration_s=3600.0,
    sn=sn,
    sdof=sdof,
)
```

### From a time history through Welch

```python
from fdscore import compute_fds_spectral_time

fds = compute_fds_spectral_time(
    x,
    fs,
    sn=sn,
    sdof=sdof,
    duration_s=3600.0,
)
```

This second route first estimates the PSD with Welch and then applies the same
spectral fatigue model.

## Deterministic Harmonic FDS

`fdscore` also supports deterministic harmonic workflows without requiring
time-domain simulation of the whole signal.

### Single sine

```python
from fdscore import compute_fds_sine

fds = compute_fds_sine(
    freq_hz=80.0,
    amp=2.0,
    duration_s=300.0,
    sn=sn,
    sdof=sdof,
)
```

### Dwell profiles

```python
from fdscore import SineDwellSegment, compute_fds_dwell_profile

segments = [
    SineDwellSegment(freq_hz=40.0, amp=1.5, duration_s=600.0),
    SineDwellSegment(freq_hz=80.0, amp=2.0, duration_s=300.0),
]

fds_mission = compute_fds_dwell_profile(segments, sn=sn, sdof=sdof)
```

### Dwell-discretized sine sweeps

```python
from fdscore import compute_fds_sine_sweep

fds_sweep = compute_fds_sine_sweep(
    f_start_hz=20.0,
    f_stop_hz=200.0,
    amp=2.0,
    duration_s=180.0,
    sn=sn,
    sdof=sdof,
    spacing="log",
    n_steps=200,
)
```

This sweep route is intentionally modeled as a dwell-discretized
approximation, not as a closed-form sweep fatigue equation.

## FDS Algebra

`fdscore` provides explicit helpers for scaling and summing compatible FDS
results.

```python
from fdscore import scale_fds, sum_fds

fds_scaled = scale_fds(fds, factor=1.5)
fds_total = sum_fds([fds_x, fds_y, fds_z], weights=[0.5, 0.3, 0.2])
```

These operations require compatible fatigue semantics and the same oscillator
grid. No implicit regridding is performed.

## Choosing a Route

Use these rules of thumb:

- `compute_fds_time(...)`: best when you trust the time history and want
  time-domain cycle counting.
- `compute_fds_spectral_psd(...)`: best when the PSD is already the natural
  input and the Dirlik approximation is acceptable.
- `compute_fds_spectral_time(...)`: useful when your input is a time history
  but you want a PSD-based fatigue workflow.
- deterministic helpers: use for sine, dwell, and sweep specifications rather
  than random vibration.
