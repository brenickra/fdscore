# PSD Analysis and Signal Synthesis

This page covers the utilities used to estimate PSDs, generate stationary
Gaussian time histories, and derive scalar metrics from PSDs.

## Welch PSD Estimation

`compute_psd_welch(...)` estimates a one-sided acceleration PSD from a time
history using the Welch method.

```python
from fdscore import compute_psd_welch, PSDParams

psd_params = PSDParams(df=1.0)
psd = compute_psd_welch(x, fs, psd=psd_params)
```

This function is useful both directly and as a building block inside
`compute_fds_spectral_time(...)`.

## Stationary Gaussian Time Synthesis

`synthesize_time_from_psd(...)` generates a random-phase, stationary Gaussian
time history consistent with a target one-sided acceleration PSD.

```python
from fdscore import synthesize_time_from_psd

x = synthesize_time_from_psd(
    f_psd_hz=f_psd,
    psd=psd_values,
    fs=1000.0,
    duration_s=12.0,
    seed=7,
)
```

This is especially useful for:

- controlled numerical studies;
- synthetic examples;
- iterative inversion predictors.

It is not intended as a general replacement for arbitrary non-stationary or
strongly non-Gaussian measured vibration signals.

## PSD Summary Metrics

`compute_psd_metrics(...)` converts an acceleration PSD into a compact set of
RMS, peak, velocity, displacement, and band-limited metrics.

```python
from fdscore import compute_psd_metrics

metrics = compute_psd_metrics(
    psd.psd,
    f_hz=psd.f,
    duration_s=3600.0,
    acc_unit="g",
    bands_hz=[(1, 10), (10, 100), (100, 400)],
)
```

The returned `PSDMetricsResult` includes:

- broadband RMS acceleration;
- Gaussian peak estimates when `duration_s` is provided;
- RMS velocity and RMS displacement;
- peak velocity and peak displacement estimates;
- per-band RMS values in `band_rms_g`;
- diagnostic metadata such as `band_coverage` and `peak_statistics`.

## Choosing the Right Utility

- `compute_psd_welch(...)`: use when the source is a time history and the next
  step expects a PSD.
- `synthesize_time_from_psd(...)`: use when the source is a PSD and the next
  step expects a time history.
- `compute_psd_metrics(...)`: use when the goal is summary reporting rather
  than another fatigue workflow.
