# fdscore

`fdscore` is a Python library for **Fatigue Damage Spectrum (FDS)** computation and **FDS-to-PSD inversion**.
It provides time-domain and spectral workflows for vibration fatigue analysis, with reusable numerical
building blocks for engineering applications.

## Main capabilities

- Time-domain FDS using FFT-domain SDOF response reconstruction and Numba-accelerated rainflow/Miner damage
- Spectral FDS using Dirlik through `FLife`
- Closed-form FDS-to-PSD inversion for pseudo-velocity (`pv`)
- Iterative PSD inversion with spectral and time-domain predictors
- Reusable transfer plans for repeated FDS evaluations
- PSD summary metrics including RMS, Gaussian peak estimates, and velocity/displacement metrics
- Normalized (`k`-only) and physical (`S-N + p_scale`) workflow support

## Installation

Install in editable mode during development:

```bash
pip install -e .
```

To enable spectral FDS and spectral iterative inversion:

```bash
pip install -e .[spectral]
```

## Quick start

```python
from fdscore import SNParams, SDOFParams, compute_fds_time, invert_fds_closed_form

sn = SNParams(slope_k=3.0, amplitude_from_range=True)
sdof = SDOFParams(q=10.0, fmin=1.0, fmax=400.0, df=1.0, metric="pv")

fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="linear", batch_size=64)
psd = invert_fds_closed_form(fds, test_duration_s=24 * 3600.0)
```

## Normalized and physical workflows

`fdscore` supports two equally valid ways of defining fatigue parameters:

- Normalized workflow:
  - use `SNParams(slope_k=...)` or `SNParams.normalized(slope_k=...)`
  - omit `p_scale` or use `p_scale=1.0`
  - recommended when the main goal is FDS shape analysis and FDS-to-PSD inversion

- Physical workflow:
  - provide `SNParams(slope_k, ref_stress, ref_cycles, ...)`
  - provide an application-specific `p_scale` explicitly
  - use when absolute Miner damage magnitude matters

For fixed `slope_k`, the combination of `ref_stress`, `ref_cycles`, and `p_scale`
acts as a global damage scaling factor. It changes the magnitude of the FDS, but not
its shape. In compatible inversion workflows, that global factor cancels when solving
for the equivalent PSD.

If `p_scale` is omitted, `fdscore` assumes `p_scale=1.0` only when the S-N definition
is normalized (`ref_stress=1`, `ref_cycles=1`). If non-unit S-N references are used,
`p_scale` must be provided explicitly.

Example of a physical setup:

```python
sn = SNParams(slope_k=6.0, ref_stress=120.0, ref_cycles=1e6)
p_scale_physical = 300.0  # application-specific response -> fatigue scaling
fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, p_scale=p_scale_physical, detrend="linear")
```

For repeated analyses with the same sampling setup:

```python
from fdscore import prepare_fds_time_plan

plan = prepare_fds_time_plan(fs=fs, n_samples=len(x), sdof=sdof)
fds_x = compute_fds_time(x, fs, sn=sn, sdof=sdof, plan=plan)
fds_y = compute_fds_time(y, fs, sn=sn, sdof=sdof, plan=plan)
fds_z = compute_fds_time(z, fs, sn=sn, sdof=sdof, plan=plan)
```

PSD summary metrics are available through `compute_psd_metrics(...)`:

```python
from fdscore import compute_psd_metrics

metrics = compute_psd_metrics(psd, duration_s=3600.0, acc_unit="g")
print(metrics.rms_acc_g, metrics.peak_acc_g, metrics.band_rms_g)
```

`compute_psd_metrics(...)` requires an explicit acceleration unit through `acc_unit` or `acc_to_m_s2`.

## Iterative inversion parameter usage

Both iterative inversion engines share the same `IterativeInversionParams` dataclass, but
they do not consume exactly the same subset of fields.

- `invert_fds_iterative_spectral(...)` uses the full regularization set, including tail caps
  and optional post-smooth/post-refine stages.
- `invert_fds_iterative_time(...)` uses the common update, smoothing, and prior controls, but
  currently ignores tail caps and post-processing/refine fields.

Each inversion result exposes `meta["param_usage"]` so callers can inspect which fields were
used and which were ignored by the selected engine.

## Examples

Minimal runnable workflows are available under [examples/README.md](examples/README.md):

- `python -m examples.minimal_fds_time`
- `python -m examples.minimal_fds_spectral`
- `python -m examples.minimal_inversion_and_metrics`

## Compatibility semantics

`fdscore` uses compatibility in two distinct ways:

- FDS algebra operations such as `sum_fds(...)` require matching damage semantics and the same oscillator frequency grid.
- Inversion operations require matching damage semantics stored in `meta["compat"]`, but do not require the candidate PSD grid to match the target FDS grid.

This distinction is intentional: FDS addition operates directly on spectra defined on the same oscillator grid, while inversion solves for a separate PSD representation.

## Method assumptions and limits

- `synthesize_time_from_psd(...)` generates stationary Gaussian random-phase realizations.
  It is useful for iterative predictors, controlled studies, and synthetic examples, but
  it is not a general substitute for arbitrary measured non-stationary vibration signals.
- Spectral FDS uses Dirlik through `FLife`. It is a spectral fatigue approximation and is
  not numerically identical to time-domain rainflow counting on a realized signal.
- `compute_fds_spectral_time(...)` first estimates a PSD with Welch and then applies Dirlik.
  Its result therefore depends both on the spectral model and on the PSD estimation settings.
- Closed-form inversion is implemented only for `metric="pv"`. For other metrics, use the
  iterative inversion engines.
- `FDSTimePlan` trades memory for speed by storing the full complex transfer matrix `H`.
  Memory scales roughly as `len(f0) * (n_fft_bins) * 16 bytes` for `complex128`. For example,
  400 oscillators and a 4 s signal at 1 kHz correspond to about 12 MB for the plan matrix alone.

## API reference

Public contracts and data structures are documented in `CONTRACTS.md`.

