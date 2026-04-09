# fdscore contracts

This document defines the core public API contracts of **fdscore** used by the
main release workflows.

It is not a generated inventory of every symbol exported by `fdscore.__all__`.
Helper exports such as low-level builders and convenience wrappers may exist
outside the scope of this contract document.

Results carry a `meta["compat"]` signature for compatibility checks across
aggregation and inversion workflows.

Compatibility is used in two different ways:
- FDS algebra compatibility (`scale_fds`, `sum_fds`) requires matching damage semantics
  and the same oscillator frequency grid.
- Inversion compatibility requires matching damage semantics (`metric`, `q`, `p_scale`, S-N)
  but does not require a separate PSD grid to match the target FDS grid.

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

Factory helpers
- `SNParams.normalized(slope_k, amplitude_from_range=True)`:
  returns a normalized S-N definition with `ref_stress=1` and `ref_cycles=1`.
  Use this together with `p_scale=1.0` when only relative FDS shape and equivalent
  PSD inversion are of interest.

Defaults
- `SNParams(slope_k=...)` is itself a normalized definition because
  `ref_stress=1` and `ref_cycles=1` by default.

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
- `meta["compat"]` is a dict-based signature with the fields:
  - `engine`
  - `metric`
  - `q`
  - `p_scale`
  - `sn` (`slope_k`, `ref_stress`, `ref_cycles`, `amplitude_from_range`)
  - `fds_kind`

### `ERSResult`
- `f` (Hz): oscillator natural frequencies.
- `response`: extreme response per oscillator for the selected response metric.
- `meta`: dictionary carrying response-spectrum diagnostics.
- `meta["compat"]` is a dict-based signature with the fields:
  - `engine`
  - `metric`
  - `q`
  - `peak_mode`
  - `ers_kind`
- `ers_kind` distinguishes generic ERS (`"response_spectrum"`) from shock-specific wrappers such as
  `"shock_response_spectrum"` and `"pseudo_velocity_shock_spectrum"`.

### `ShockSpectrumPair`
Positive/negative sided shock-spectrum pair.

Fields
- `neg`: `ERSResult` for the negative-side magnitude spectrum.
- `pos`: `ERSResult` for the positive-side magnitude spectrum.
- `meta`: pair-level metadata, typically carrying `peak_mode="both"` and provenance.

Notes
- Used by `compute_srs_time(...)` and `compute_pvss_time(...)` when `peak_mode="both"`.
- The sides remain explicit; they are not collapsed into a single 2D `ERSResult`.

### `ShockEvent`
One detected event in a 1D shock time history.

Fields
- `peak_index`, `start_index`, `stop_index`
- `peak_time_s`, `start_time_s`, `stop_time_s`
- `peak_value`: signed peak value in the preprocessed signal.
- `peak_abs`: absolute peak magnitude.
- `polarity`: `"pos" | "neg"`

### `ShockEventSet`
Detected event collection and its detector settings.

Fields
- `events`: tuple of `ShockEvent`
- `fs`: sampling rate used during detection
- `n_samples`: source signal length used during detection
- `meta`: detector settings and provenance

### `RollingERSResult`
Event-window response spectra stacked over multiple windows.

Fields
- `f` (Hz): oscillator natural frequencies.
- `t_center_s` (s): center time for each event/window.
- `response`: 2D array with shape `(n_windows, n_freq)`.
- `meta`: rolling-workflow metadata.

### `HalfSinePulse`
Parameterized half-sine acceleration pulse.

Fields
- `amplitude` (float, >0): unsigned acceleration amplitude.
- `duration_s` (float, >0): pulse duration.
- `polarity`: `"pos" | "neg"`
- `meta`: fit/synthesis metadata.

Derived property
- `signed_amplitude`: signed amplitude implied by `polarity`.
### `FDSTimePlan`
Precomputed transfer plan for repeated time-domain FDS calls.

Fields
- `fs` (float): sampling rate used to build the plan.
- `n_samples` (int): signal length in samples.
- `f` (Hz): oscillator frequency grid used by the plan.
- `zeta` (float): damping ratio (`1/(2Q)`).
- `metric` (str): `"pv" | "disp" | "vel" | "acc"`.
- `H` (complex ndarray): FFT-domain transfer matrix with shape `(len(f), n_fft_bins)`.

Memory note
- `H` is stored explicitly as a `complex128` matrix.
- Memory scales approximately as `len(f) * n_fft_bins * 16 bytes`.
- Example: 400 oscillators and a 4 s signal at 1 kHz correspond to roughly
  `400 * 2001 * 16 ~= 12 MB` for the plan matrix alone.

### `PSDResult`
- `f` (Hz): frequency grid.
- `psd` (units^2/Hz): acceleration PSD.
- `meta`: additional diagnostics.

### `SineDwellSegment`
Defines one deterministic harmonic dwell segment.

Fields
- `freq_hz` (float, >0): dwell excitation frequency.
- `amp` (float, >=0): base-motion amplitude.
- `duration_s` (float, >0): dwell duration.
- `input_motion` (str): `"acc" | "vel" | "disp"` describing what `amp` represents.
- `label` (optional str): optional user label stored on the segment object itself.

Current note
- deterministic helpers currently do not propagate `label` into result provenance automatically.

### `PSDMetricsResult`
Summary metrics derived from acceleration PSD.

Fields
- `rms_acc_g`, `rms_acc_m_s2`
- `peak_acc_g`, `peak_acc_m_s2`
- `peak_factor`, `zero_upcrossing_hz`, `effective_cycles`
- `rms_vel_m_s`, `peak_vel_m_s`
- `rms_disp_mm`, `peak_disp_mm`, `disp_pk_pk_mm`
- `band_rms_g` (dict with per-band RMS in g)
- `meta` (inputs and settings used for metric computation), including:
  - `band_coverage` for requested vs effective band support on the PSD grid
  - `peak_statistics` for Gaussian peak diagnostic details such as effective-cycle floors

## Core API

### `compute_fds_time(x, fs, sn, sdof, ...) -> FDSResult`
Computes time-domain FDS using FFT-domain SDOF transfer and Numba-native rainflow/Miner damage.

Input
- `x`: 1D numpy array (finite values), base acceleration time history.
- `fs`: sampling rate [Hz] (finite, >0).
- `sn`, `sdof`: parameter objects.
- `p_scale`: scale applied to response time series before damage counting.
  For fixed `slope_k`, the absolute damage level scales globally with
  `p_scale**k / (ref_cycles * ref_stress**k)`.
- `detrend`: `"linear"|"mean"|"none"` applied to `x`.
- `batch_size`: number of oscillators per FFT batch.
- `plan` (optional): `FDSTimePlan` built by `prepare_fds_time_plan(...)`.
  When provided and compatible, transfer matrix rebuild is skipped.

Validation
- Nyquist: by default errors if `max(f) >= fs/2`.

Output
- Miner damage spectrum `damage(f)` plus a `compat` signature embedding `metric`, `q`, `p_scale`, and S-N parameters.

Usage notes
- Normalized workflow:
  - `sn = SNParams(slope_k=...)` or `SNParams.normalized(slope_k=...)`
  - omit `p_scale` or pass `p_scale = 1.0`
- Physical workflow:
  - use explicit `ref_stress`, `ref_cycles`, and application-specific `p_scale`
- If `p_scale` is omitted, `fdscore` assumes `p_scale=1.0` only for normalized
  S-N definitions (`ref_stress=1`, `ref_cycles=1`).
- If non-unit S-N references are provided, `p_scale` must be passed explicitly.

### `prepare_fds_time_plan(fs, n_samples, sdof, ...) -> FDSTimePlan`
Precomputes and stores transfer data for repeated `compute_fds_time` calls that share
the same sampling setup (`fs`, `n_samples`, and `sdof`).

Tradeoff
- Reusing a plan avoids rebuilding the transfer matrix for every call.
- The tradeoff is memory: the full `H` matrix is materialized and kept in the plan.
- Plan reuse also requires the same effective oscillator grid after Nyquist validation. If clipping is enabled (`strict_nyquist=False`), plan creation and later calls must be consistent about that choice.

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

### `compute_ers_sine(freq_hz, amp, sdof, ...) -> ERSResult`
Computes deterministic ERS for a single-frequency harmonic base excitation.

Notes
- ERS remains tied to the selected `sdof.metric`.
- `input_motion` specifies whether the provided amplitude is base acceleration, velocity, or displacement.
- The current implementation supports `peak_mode="abs"`.

### `compute_fds_sine(freq_hz, amp, duration_s, sn, sdof, ...) -> FDSResult`
Computes deterministic FDS for a single-frequency harmonic base excitation without time simulation.

Damage model
- response amplitude is computed from the SDOF transfer at the excitation frequency
- cycles are `freq_hz * duration_s`
- damage follows the existing `SNParams` and `p_scale` conventions

### `compute_ers_dwell_profile(segments, sdof, ...) -> ERSResult`
Computes mission-level ERS from multiple deterministic dwell segments.

Mission rule
- ERS composes by pointwise envelope, not summation.

### `compute_ers_time(x, fs, sdof, ...) -> ERSResult`
Computes time-domain ERS by reconstructing SDOF responses in the FFT domain and
extracting their peak response.

Current contract
- supports `peak_mode="abs"`
- uses the selected `sdof.metric`
- accepts an optional compatible `FDSTimePlan` to reuse transfer data

Interpretation
- if ERS and FDS use different metrics, their frequency grid may match, but the
  transfer matrix must still match the chosen ERS metric.

### `compute_srs_time(x, fs, sdof, ...) -> ERSResult | ShockSpectrumPair`
Computes a shock response spectrum using the dedicated recursive shock engine.

Current contract
- requires `sdof.metric="acc"`
- supports `peak_mode="abs"|"pos"|"neg"|"both"`
- supports `detrend="linear"|"mean"|"median"|"none"`
- returns `meta["compat"]["ers_kind"] == "shock_response_spectrum"`
- does not reuse `FDSTimePlan`

Interpretation
- this is the dedicated time-domain SRS path for base-acceleration shock histories
- unlike `compute_ers_time(...)`, it is intended specifically for transient shock analysis

### `compute_pvss_time(x, fs, sdof, ...) -> ERSResult | ShockSpectrumPair`
Computes a pseudo-velocity shock spectrum using the dedicated recursive shock engine.

Current contract
- requires `sdof.metric="pv"`
- supports `peak_mode="abs"|"pos"|"neg"|"both"`
- supports `detrend="linear"|"mean"|"median"|"none"`
- returns `meta["compat"]["ers_kind"] == "pseudo_velocity_shock_spectrum"`
- does not reuse `FDSTimePlan`

Interpretation
- this is the dedicated time-domain PVSS path for base-acceleration shock histories
- it shares the same recursive backend philosophy as `compute_srs_time(...)`

### `detect_shock_events(x, fs, ...) -> ShockEventSet`
Detects shock-like events in a 1D time history.

Current contract
- 1D only
- supports `detrend="linear"|"mean"|"median"|"none"`
- supports `polarity="abs"|"pos"|"neg"`
- threshold can be defined either by:
  - explicit `threshold_value`, or
  - `threshold_reference="rms"|"std"|"peak"` with `threshold_multiplier`
- `min_separation_s` controls the minimum distance between accepted events
- `window_s` defines the event window used later by rolling shock calculations

Interpretation
- this is intended as the event-selection stage for shock-window workflows
- it does not classify event type; it only locates and windows candidate events

### `compute_rolling_srs_time(x, fs, sdof, events, ...) -> RollingERSResult`
Computes SRS on each detected event window.

Current contract
- requires `sdof.metric="acc"`
- requires `events.fs == fs` and `events.n_samples == len(x)`
- currently supports `peak_mode="abs"|"pos"|"neg"`
- rolling is event-window based, not fixed-stride

### `compute_rolling_pvss_time(x, fs, sdof, events, ...) -> RollingERSResult`
Computes PVSS on each detected event window.

Current contract
- requires `sdof.metric="pv"`
- requires `events.fs == fs` and `events.n_samples == len(x)`
- currently supports `peak_mode="abs"|"pos"|"neg"`
- rolling is event-window based, not fixed-stride

### `fit_half_sine_to_pvss(pvss, ...) -> HalfSinePulse`
Fits an enveloping half-sine acceleration pulse from a PVSS result.

Current contract
- input must be an `ERSResult`
- requires `meta["compat"]["metric"] == "pv"`
- requires `meta["compat"]["ers_kind"] == "pseudo_velocity_shock_spectrum"`
- `polarity` selects the sign of the returned pulse, not the fitted amplitude magnitude

Interpretation
- this is an enveloping approximation derived from PVSS
- it is intended for simplification and comparative studies, not exact recovery of the original measured pulse

### `synthesize_half_sine_pulse(pulse, fs, ...) -> ndarray`
Synthesizes a 1D half-sine acceleration pulse.

Current contract
- `pulse` must be `HalfSinePulse`
- supports `total_duration_s` and `t_start_s`
- output is a 1D numpy array in the same acceleration units implied by `pulse.amplitude`

### `envelope_srs(results)`
Computes a pointwise envelope across compatible SRS results.

Current contract
- accepts either:
  - a sequence of `ERSResult` one-sided SRS values, or
  - a sequence of `ShockSpectrumPair` values for `peak_mode="both"`
- mixing the two forms is rejected
- requires `ers_kind == "shock_response_spectrum"`

### `envelope_pvss(results)`
Computes a pointwise envelope across compatible PVSS results.

Current contract
- accepts either:
  - a sequence of `ERSResult` one-sided PVSS values, or
  - a sequence of `ShockSpectrumPair` values for `peak_mode="both"`
- mixing the two forms is rejected
- requires `ers_kind == "pseudo_velocity_shock_spectrum"`
### `compute_ers_sine_sweep(...) -> ERSResult`
Computes an approximate deterministic ERS for a sine sweep by discretizing the
sweep path into `n_steps` dwell segments and taking their ERS envelope.

Parameters of interest
- `spacing`: `"linear" | "log"`
- `n_steps`: number of dwell segments in the discretized sweep

### `compute_fds_sine_sweep(...) -> FDSResult`
Computes an approximate deterministic FDS for a sine sweep by discretizing the
sweep path into `n_steps` dwell segments and summing their damage spectra.

Notes
- This is an approximation based on dwell discretization.
- It is not a closed-form sweep damage equation.

### `compute_fds_dwell_profile(segments, sn, sdof, ...) -> FDSResult`
Computes mission-level FDS from multiple deterministic dwell segments.

Mission rule
- FDS composes by damage summation.

### `envelope_ers(results)`
Computes a pointwise envelope across compatible ERS results.

Compatibility
- same response metric
- same `q` / damping
- same peak mode
- same oscillator grid

### `scale_fds(fds, factor)`
Multiplies damage by `factor > 0` and records structured provenance for the new result, including the scaled input provenance.

### `sum_fds(fds_list, weights=None)`
Sums spectra only when compatible under FDS algebra rules:
- same `metric`, `q`, `p_scale`, same S-N, and same frequency grid.

The output preserves the compatible metadata from the reference spectrum and records structured provenance for all inputs and weights.
No implicit regridding is performed.

### `invert_fds_closed_form(fds, *, test_duration_s, strict_metric=True)`
Closed-form inversion based on the Henderson-Piersol / MIL-STD style formulation:
- Converts `Damage -> DP` using `p_scale`, S-N, and `gamma(1+b/2)`.
- Converts `DP -> PSD` with `G(f) = f*zeta*(DP/(f*T))^(2/b)`.

Requirements
- `fds.meta["compat"]` must contain `metric="pv"`, `q`, `p_scale`, and `sn`.
- `test_duration_s` must be passed as a keyword argument and satisfy `> 0`.
- `strict_metric=True` enforces `metric="pv"`.

Output
- `PSDResult` with reconstruction diagnostics in `meta["reconstruction"]`.

Interpretation
- In a compatible workflow, `p_scale`, `ref_stress`, and `ref_cycles` cancel in the
  closed-form inversion. They affect absolute FDS magnitude, but not the equivalent PSD.

## Available inversion engines

### Iterative inversion (spectral)
- `invert_fds_iterative_spectral(...)`: synthesizes acceleration PSD by matching a target FDS using a spectral Dirlik predictor.
- Requires `FLife`.
- Uses the full `IterativeInversionParams` set, including:
  - core update/smoothing/prior knobs
  - tail/low caps
  - optional post-smooth and post-refine stages
- Returned `PSDResult.meta["param_usage"]` lists the fields consumed by the spectral engine.

### Iterative inversion (time-domain)
- `invert_fds_iterative_time(...)`: synthesizes acceleration PSD by matching a target FDS using random-phase time synthesis and `compute_fds_time`.
- Supports optional target-duration scaling through `target_duration_s`.
- Uses the common `IterativeInversionParams` subset:
  - `iters`, `gamma`, `gain_min`, `gain_max`, `alpha_sharpness`
  - `floor`
  - `smooth_enabled`, `smooth_window_bins`, `smooth_every_n_iters`
  - `prior_blend`, `prior_power`
  - `edge_anchor_hz`, `edge_anchor_blend`
- The time-domain engine currently ignores:
  - `tail_cap_start_hz`, `tail_cap_ratio`, `low_cap_ratio`
  - `post_smooth_window_bins`, `post_smooth_blend`
  - `post_refine_iters`, `post_refine_gamma`, `post_refine_min`, `post_refine_max`
- Returned `PSDResult.meta["param_usage"]` makes this usage explicit at runtime.
- Returned `PSDResult.meta["diagnostics"]["predictor_config"]` records the current fixed predictor policy used by the time-domain engine: `remove_mean=True` in synthesis and `detrend="none"`, `batch_size=64` in `compute_fds_time(...)`.

## Method assumptions and limits

- `synthesize_time_from_psd(...)` generates stationary Gaussian time histories through
  random-phase IFFT. It is a numerical synthesis helper, not a general representation
  of arbitrary measured vibration signals, transients, or strongly non-Gaussian processes.
- Spectral FDS uses Dirlik through `FLife`; it is not the same method as time-domain
  rainflow counting and can produce different absolute FDS levels.
- `compute_fds_spectral_time(...)` first estimates PSD with Welch and then applies Dirlik.
  Its result therefore depends on both the spectral fatigue model and the PSD estimation settings.
- Explicit spectral PSD inputs are expected to be non-negative. Tiny negative values consistent
  with numerical noise are clamped to zero; materially negative values raise `ValidationError`.
- Closed-form inversion is supported only for `metric="pv"`. Setting `strict_metric=False`
  only bypasses the guard; it does not generalize the closed-form derivation to other metrics.
- `FDSTimePlan` stores the full complex transfer matrix `H` with shape
  `(len(f), n_fft_bins)`. This improves reuse performance at the cost of memory.
- `compute_ers_time(...)` remains the generic FFT-domain ERS implementation.
- `compute_srs_time(...)` and `compute_pvss_time(...)` intentionally use a separate recursive
  shock backend so transient-shock behavior can evolve without changing the generic ERS engine.

## External dependencies
- Spectral FDS and spectral iterative inversion require `FLife`.
- Time-domain FDS, PSD metrics, closed-form inversion, and time-domain iterative inversion do not require `FLife`.







