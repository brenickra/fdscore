# fdscore contracts

This document defines the public API contracts of **fdscore**.

Results carry a `meta["compat"]` signature for compatibility checks across
aggregation and inversion workflows.

## Core objects

### `SNParams`
Defines the S-N curve used in Miner damage.

Fields
- `slope_k` (float, >0): fatigue exponent `k` (also called `b` in some derivations).
- `ref_stress` (float, >0): stress amplitude reference.
- `ref_cycles` (float, >0): cycles at `ref_stress`.
- `amplitude_from_range` (bool):
  - `True` (default): rainflow range is interpreted as `2*amplitude`.
  - `False`: rainflow range is interpreted as amplitude directly.

Derived constant
- `C = ref_cycles * ref_stress**slope_k`

### `SDOFParams`
Defines oscillator grid and response metric.

Fields
- `q` (float, >0): quality factor. Damping ratio is `zeta = 1/(2Q)`.
- `metric` (str): `"pv" | "disp" | "vel" | "acc"`.
- Frequency grid is either:
  - explicit: `f: np.ndarray` (1D, strictly increasing, >0), or
  - linear: `fmin,fmax,df` (all >0, `fmax>fmin`).

Notes
- Supported metrics are `"pv"`, `"disp"`, `"vel"`, and `"acc"`.
- Closed-form inversion is available for `"pv"`.

### `FDSResult`
- `f` (Hz): oscillator natural frequencies.
- `damage` (-): Miner damage per oscillator.
- `meta`: dictionary with at least `meta["compat"]` used for safe algebra/inversion.

### `FDSTimePlan`
Precomputed transfer plan for repeated time-domain FDS calls.

Fields
- `fs` (float): sampling rate used to build the plan.
- `n_samples` (int): signal length in samples.
- `f` (Hz): oscillator frequency grid used by the plan.
- `zeta` (float): damping ratio (`1/(2Q)`).
- `metric` (str): `"pv" | "disp" | "vel" | "acc"`.
- `H` (complex ndarray): FFT-domain transfer matrix with shape `(len(f), n_fft_bins)`.

### `PSDResult`
- `f` (Hz): frequency grid.
- `psd` (units^2/Hz): acceleration PSD.
- `meta`: additional diagnostics.

### `PSDMetricsResult`
Summary metrics derived from acceleration PSD.

Fields
- `rms_acc_g`, `rms_acc_m_s2`
- `peak_acc_g`, `peak_acc_m_s2`
- `peak_factor`, `zero_upcrossing_hz`, `effective_cycles`
- `rms_vel_m_s`, `peak_vel_m_s`
- `rms_disp_mm`, `peak_disp_mm`, `disp_pk_pk_mm`
- `band_rms_g` (dict with per-band RMS in g)
- `meta` (inputs and settings used for metric computation)

## Core API

### `compute_fds_time(x, fs, sn, sdof, ...) -> FDSResult`
Computes time-domain FDS using FFT-domain SDOF transfer and Numba-native rainflow/Miner damage.

Input
- `x`: 1D numpy array (finite values), base acceleration time history.
- `fs`: sampling rate [Hz] (finite, >0).
- `sn`, `sdof`: parameter objects.
- `p_scale`: scale applied to response time series before damage counting.
- `detrend`: `"linear"|"mean"|"none"` applied to `x`.
- `batch_size`: number of oscillators per FFT batch.
- `plan` (optional): `FDSTimePlan` built by `prepare_fds_time_plan(...)`.
  When provided and compatible, transfer matrix rebuild is skipped.

Validation
- Nyquist: by default errors if `max(f) >= fs/2`.

Output
- Miner damage spectrum `damage(f)` plus a `compat` signature embedding `metric`, `q`, `p_scale`, and S-N parameters.

### `prepare_fds_time_plan(fs, n_samples, sdof, ...) -> FDSTimePlan`
Precomputes and stores transfer data for repeated `compute_fds_time` calls that share
the same sampling setup (`fs`, `n_samples`, and `sdof`).

### `compute_psd_metrics(psd, ...) -> PSDMetricsResult`
Computes summary metrics from an acceleration PSD.

Input
- `psd`: `PSDResult` or 1D PSD array.
- `f_hz`: frequency grid when `psd` is a raw array.
- `duration_s` (optional): used for Gaussian peak estimates; peak fields are `nan` when omitted.
- `acc_unit` (`"g"` or `"m/s2"`) or `acc_to_m_s2`: required.
- `bands_hz`: frequency bands used for band-limited RMS in g.

Output
- RMS/peak metrics for acceleration, velocity, and displacement.
- Upcrossing/peak-factor statistics.
- Band RMS values in g.

### `scale_fds(fds, factor)`
Multiplies damage by `factor > 0` and records provenance.

### `sum_fds(list_of_fds, weights=None)`
Sums spectra only when compatible:
- same `metric`, `q`, `p_scale`, same S-N, and same frequency grid.

No implicit regridding is performed.

### `invert_fds_closed_form(fds, test_duration_s)`
Closed-form inversion based on the Henderson-Piersol / MIL-STD style formulation:
- Converts `Damage -> DP` using `p_scale`, S-N, and `gamma(1+b/2)`.
- Converts `DP -> PSD` with `G(f) = f*zeta*(DP/(f*T))^(2/b)`.

Requirements
- `fds.meta["compat"]` must contain `metric="pv"`, `q`, `p_scale`, and `sn`.
- `test_duration_s > 0`.

Output
- `PSDResult` with reconstruction diagnostics in `meta["reconstruction"]`.

## Available inversion engines

### Iterative inversion (spectral)
- `invert_fds_iterative_spectral(...)`: synthesizes acceleration PSD by matching a target FDS using a spectral Dirlik predictor.
- Requires `FLife`.
- Regularization options are defined in `IterativeInversionParams`.

### Iterative inversion (time-domain)
- `invert_fds_iterative_time(...)`: synthesizes acceleration PSD by matching a target FDS using random-phase time synthesis and `compute_fds_time`.
- Supports optional target-duration scaling through `target_duration_s`.

## External dependencies
- Spectral FDS and spectral iterative inversion require `FLife`.
- Time-domain FDS, PSD metrics, closed-form inversion, and time-domain iterative inversion do not require `FLife`.
